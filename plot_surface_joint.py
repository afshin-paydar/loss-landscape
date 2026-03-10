"""
Calculate and visualize the loss surface for joint pruning + quantization models.

Extension of plot_surface.py that:
  - Loads PruningQuantizationWrapper models
  - Evaluates both task loss AND Lagrangian at each grid point
  - Stores multiple surfaces: task_loss, lagrangian_loss, model_size, sparsity
  - Supports subspace-specific perturbations (weights_only, alpha_only, beta_only)

Usage:
  python plot_surface_joint.py \
    --model resnet56 --joint \
    --model_file cifar10/trained_nets/resnet56_joint/final.pth \
    --x=-1:1:51 --y=-1:1:51 \
    --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn \
    --lambda_s 0.1 --gamma 0.01 --S_max_ratio 0.5 \
    --cuda --plot
"""

import argparse
import copy
import h5py
import torch
import time
import socket
import os
import sys
import numpy as np
import torchvision
import torch.nn as nn

import dataloader
import evaluation_joint
import net_plotter_joint
import plot_2D
import plot_1D
import scheduler
import mpi4pytorch as mpi

from joint_model import PruningQuantizationWrapper
import cifar10.model_loader as cifar10_loader


def load_joint_model(args):
    """Load model and wrap with PruningQuantizationWrapper if --joint."""
    # Load base model
    base_net = cifar10_loader.load(args.model, None)
    
    if args.joint:
        bit_widths = [int(b) for b in args.bit_widths.split(',')]
        net = PruningQuantizationWrapper(base_net, bit_widths=bit_widths)
    else:
        net = base_net
    
    # Load weights
    if args.model_file:
        checkpoint = torch.load(args.model_file, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            net.load_state_dict(checkpoint['state_dict'])
        else:
            net.load_state_dict(checkpoint)
        print(f"Loaded model from {args.model_file}")
    
    return net


def name_surface_file(args, dir_file):
    """Name the surface file."""
    if args.surf_file:
        return args.surf_file
    
    surf_file = dir_file
    surf_file += '_[%s,%s,%d]' % (str(args.xmin), str(args.xmax), int(args.xnum))
    if args.y:
        surf_file += 'x[%s,%s,%d]' % (str(args.ymin), str(args.ymax), int(args.ynum))
    
    if args.joint:
        surf_file += '_joint'
        surf_file += '_ls=%s_g=%s' % (str(args.lambda_s), str(args.gamma))
    
    return surf_file + ".h5"


def setup_surface_file(args, surf_file, dir_file):
    """Create the surface HDF5 file with coordinate grids."""
    if os.path.exists(surf_file):
        f = h5py.File(surf_file, 'r')
        if 'xcoordinates' in f.keys():
            f.close()
            print(f"{surf_file} already set up")
            return
        f.close()
    
    f = h5py.File(surf_file, 'a')
    f['dir_file'] = dir_file
    
    xcoordinates = np.linspace(args.xmin, args.xmax, num=int(args.xnum))
    f['xcoordinates'] = xcoordinates
    
    if args.y:
        ycoordinates = np.linspace(args.ymin, args.ymax, num=int(args.ynum))
        f['ycoordinates'] = ycoordinates
    
    f.close()


def crunch_joint(surf_file, net, weights, alphas, betas, d, 
                 dataloader, comm, rank, args):
    """
    Calculate loss values for joint model across the 2D grid.
    
    Stores: task_loss, task_acc, lagrangian_loss, model_size, sparsity
    """
    f = h5py.File(surf_file, 'r+' if rank == 0 else 'r')
    
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None
    
    shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates), len(ycoordinates))
    
    # Initialize arrays for all metrics
    metrics = {}
    for key in ['task_loss', 'task_acc', 'lagrangian_loss', 'model_size', 'sparsity']:
        if key not in f.keys():
            metrics[key] = -np.ones(shape=shape)
            if rank == 0:
                f[key] = metrics[key]
        else:
            metrics[key] = f[key][:]
    
    # Use task_loss to find unplotted indices
    inds, coords, inds_nums = scheduler.get_job_indices(
        metrics['task_loss'], xcoordinates, ycoordinates, comm
    )
    
    print(f'Computing {len(inds)} values for rank {rank}')
    start_time = time.time()
    total_sync = 0.0
    
    criterion = nn.CrossEntropyLoss()
    
    # Compute S_max
    if args.S_max_ratio > 0 and hasattr(net, 'conv_layer_names'):
        wrapper = net.module if hasattr(net, 'module') else net
        baseline_size = 0
        for ln in wrapper.conv_layer_names:
            info = wrapper.conv_layer_info[ln]
            baseline_size += 32 * info['out_channels'] * info['params_per_filter']
        S_max = args.S_max_ratio * baseline_size
    else:
        S_max = None
    
    for count, ind in enumerate(inds):
        coord = coords[count]
        
        # Set parameters with directional perturbation
        net_plotter_joint.set_joint_params(
            net, weights, alphas, betas, directions=d, step=coord
        )
        
        loss_start = time.time()
        task_loss, acc, lagrangian, model_size, sparsity = \
            evaluation_joint.eval_joint_loss(
                net, criterion, dataloader, args.cuda,
                lambda_s=args.lambda_s, gamma=args.gamma, S_max=S_max
            )
        loss_time = time.time() - loss_start
        
        # Record results
        metrics['task_loss'].ravel()[ind] = task_loss
        metrics['task_acc'].ravel()[ind] = acc
        metrics['lagrangian_loss'].ravel()[ind] = lagrangian
        metrics['model_size'].ravel()[ind] = model_size
        metrics['sparsity'].ravel()[ind] = sparsity
        
        # MPI sync
        sync_start = time.time()
        for key in metrics:
            metrics[key] = mpi.reduce_max(comm, metrics[key])
        sync_time = time.time() - sync_start
        total_sync += sync_time
        
        # Write to file (rank 0 only)
        if rank == 0:
            for key in metrics:
                f[key][:] = metrics[key]
            f.flush()
        
        print(f'Rank {rank}  {count}/{len(inds)} ({100.0*count/max(len(inds),1):.1f}%) '
              f'coord={coord} task_loss={task_loss:.3f} acc={acc:.1f}% '
              f'lagr={lagrangian:.3f} size={model_size:.0f} '
              f'time={loss_time:.1f}s sync={sync_time:.1f}s')
    
    # Keep MPI in sync
    for i in range(max(inds_nums) - len(inds)):
        for key in metrics:
            metrics[key] = mpi.reduce_max(comm, metrics[key])
    
    total_time = time.time() - start_time
    print(f'Rank {rank} done! Total: {total_time:.1f}s Sync: {total_sync:.1f}s')
    
    f.close()


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint loss surface plotting')
    parser.add_argument('--mpi', '-m', action='store_true', help='use mpi')
    parser.add_argument('--cuda', '-c', action='store_true', help='use cuda')
    parser.add_argument('--threads', default=2, type=int)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--batch_size', default=128, type=int)
    
    # Data
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--datapath', default='cifar10/data')
    parser.add_argument('--raw_data', action='store_true', default=False)
    parser.add_argument('--data_split', default=1, type=int)
    parser.add_argument('--split_idx', default=0, type=int)
    parser.add_argument('--trainloader', default='')
    parser.add_argument('--testloader', default='')
    
    # Model
    parser.add_argument('--model', default='resnet56')
    parser.add_argument('--model_file', default='', help='path to trained joint model')
    parser.add_argument('--joint', action='store_true', help='use PruningQuantizationWrapper')
    parser.add_argument('--bit_widths', default='2,4,8,16', type=str)
    
    # Direction
    parser.add_argument('--dir_file', default='')
    parser.add_argument('--x', default='-1:1:51')
    parser.add_argument('--y', default=None)
    parser.add_argument('--xnorm', default='filter')
    parser.add_argument('--ynorm', default='filter')
    parser.add_argument('--xignore', default='biasbn')
    parser.add_argument('--yignore', default='biasbn')
    parser.add_argument('--subspace', default='all',
                        choices=['all', 'weights_only', 'alpha_only', 'beta_only'],
                        help='which subspace to perturb')
    parser.add_argument('--idx', default=0, type=int)
    parser.add_argument('--surf_file', default='')
    
    # Joint Lagrangian parameters
    parser.add_argument('--lambda_s', default=0.1, type=float)
    parser.add_argument('--gamma', default=0.01, type=float)
    parser.add_argument('--S_max_ratio', default=0.5, type=float)
    
    # Plot
    parser.add_argument('--vmax', default=10, type=float)
    parser.add_argument('--vmin', default=0.1, type=float)
    parser.add_argument('--vlevel', default=0.5, type=float)
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--plot', action='store_true', default=False)
    
    args = parser.parse_args()
    
    torch.manual_seed(123)
    
    # ---- Environment ----
    if args.mpi:
        comm = mpi.setup_MPI()
        rank, nproc = comm.Get_rank(), comm.Get_size()
    else:
        comm, rank, nproc = None, 0, 1
    
    if args.cuda:
        if not torch.cuda.is_available():
            raise Exception('CUDA not available')
        gpu_count = torch.cuda.device_count()
        torch.cuda.set_device(rank % gpu_count)
        print(f'Rank {rank} using GPU {torch.cuda.current_device()} of {gpu_count}')
    
    # ---- Parse coordinates ----
    args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]
    args.ymin, args.ymax, args.ynum = (None, None, None)
    if args.y:
        args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]
    
    # ---- Load model ----
    net = load_joint_model(args)
    
    # Extract parameters
    if args.joint:
        weights, alphas, betas = net_plotter_joint.get_joint_params(net)
    else:
        weights = [p.data.clone() for p in net.parameters()]
        alphas, betas = [], []
    
    if args.ngpu > 1:
        net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    
    # ---- Setup directions ----
    dir_file = args.dir_file if args.dir_file else (args.model_file + '_joint_' + args.subspace + '.h5')
    
    if rank == 0:
        if args.joint:
            net_plotter_joint.setup_direction_joint(args, dir_file, net)
        else:
            import net_plotter
            net_plotter.setup_direction(args, dir_file, net)
    
    # ---- Setup surface file ----
    surf_file = name_surface_file(args, dir_file)
    if rank == 0:
        setup_surface_file(args, surf_file, dir_file)
    
    mpi.barrier(comm)
    
    # ---- Load directions ----
    if args.joint:
        d = net_plotter_joint.load_directions_joint(dir_file)
    else:
        import net_plotter
        d = net_plotter.load_directions(dir_file)
    
    # ---- Load data ----
    if rank == 0 and args.dataset == 'cifar10':
        torchvision.datasets.CIFAR10(root=args.datapath, train=True, download=True)
    mpi.barrier(comm)
    
    trainloader, testloader = dataloader.load_dataset(
        args.dataset, args.datapath, args.batch_size, args.threads,
        args.raw_data, args.data_split, args.split_idx,
        args.trainloader, args.testloader
    )
    
    # ---- Compute surface ----
    if args.joint:
        crunch_joint(surf_file, net, weights, alphas, betas, d,
                     trainloader, comm, rank, args)
    else:
        import evaluation
        import net_plotter
        s = copy.deepcopy(net.state_dict())
        # Fall back to original crunch for non-joint models
        from plot_surface import crunch
        crunch(surf_file, net, weights, s, d, trainloader,
               'train_loss', 'train_acc', comm, rank, args)
    
    # ---- Plot ----
    if args.plot and rank == 0:
        if args.y:
            # Plot task loss contour
            plot_2D.plot_2d_contour(surf_file, 'task_loss',
                                   args.vmin, args.vmax, args.vlevel, args.show)
            # Plot Lagrangian contour
            if args.joint:
                plot_2D.plot_2d_contour(surf_file, 'lagrangian_loss',
                                       args.vmin, args.vmax * 2, args.vlevel, args.show)
        else:
            plot_1D.plot_1d_loss_err(surf_file, args.xmin, args.xmax, 5, False, args.show)
    
    print("Done!")
