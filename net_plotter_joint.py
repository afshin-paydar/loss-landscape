"""
Extended direction generation and weight manipulation for joint pruning +
mixed-precision quantization models.

Handles the composite parameter space theta = [W; alpha; beta] with
sub-space-specific normalization:
  - W: filter normalization (Li et al. 2018)
  - alpha: layer-wise L2 normalization
  - beta: layer-wise L2 normalization
"""

import torch
import copy
import h5py
import os
import net_plotter
import h5_util
from joint_model import PruningQuantizationWrapper


def get_joint_params(net):
    """
    Extract parameters split into three groups: (weights, alphas, betas).
    
    Args:
        net: PruningQuantizationWrapper (or DataParallel wrapper around it)
    
    Returns:
        (weight_params, alpha_params, beta_params) - each a list of tensors
    """
    wrapper = net.module if hasattr(net, 'module') else net
    
    weight_params = []
    alpha_params = []
    beta_params = []
    
    for name, param in wrapper.named_parameters():
        if 'pruning_alphas' in name:
            alpha_params.append(param.data.clone())
        elif 'bitwidth_betas' in name:
            beta_params.append(param.data.clone())
        else:
            weight_params.append(param.data.clone())
    
    return weight_params, alpha_params, beta_params


def set_joint_params(net, weights, alphas, betas, directions=None, step=None):
    """
    Set the model parameters, optionally with directional perturbation.
    
    If directions is None: set params to (weights, alphas, betas).
    If directions is provided: set params to originals + step * directions.
    
    Args:
        net: PruningQuantizationWrapper
        weights, alphas, betas: original parameter lists
        directions: tuple of (w_dirs, a_dirs, b_dirs) lists, or None
        step: [step_x, step_y] for 2D, or scalar for 1D
    """
    wrapper = net.module if hasattr(net, 'module') else net
    
    # Separate parameters by name
    w_idx, a_idx, b_idx = 0, 0, 0
    
    for name, param in wrapper.named_parameters():
        if 'pruning_alphas' in name:
            if directions is None:
                param.data.copy_(alphas[a_idx])
            else:
                d_a = _compute_change(directions, 'alpha', a_idx, step)
                param.data.copy_(alphas[a_idx] + d_a)
            a_idx += 1
        elif 'bitwidth_betas' in name:
            if directions is None:
                param.data.copy_(betas[b_idx])
            else:
                d_b = _compute_change(directions, 'beta', b_idx, step)
                param.data.copy_(betas[b_idx] + d_b)
            b_idx += 1
        else:
            if directions is None:
                param.data.copy_(weights[w_idx])
            else:
                d_w = _compute_change(directions, 'weight', w_idx, step)
                param.data.copy_(weights[w_idx] + d_w)
            w_idx += 1


def _compute_change(directions, param_type, idx, step):
    """Compute the directional change for a single parameter."""
    # directions is a list of (w_dir, a_dir, b_dir) tuples, one per axis
    type_map = {'weight': 0, 'alpha': 1, 'beta': 2}
    t = type_map[param_type]
    
    if len(directions) == 2:
        # 2D: directions = [(w_dx, a_dx, b_dx), (w_dy, a_dy, b_dy)]
        dx = directions[0][t][idx]
        dy = directions[1][t][idx]
        return torch.tensor(dx).float() * step[0] + torch.tensor(dy).float() * step[1]
    else:
        # 1D
        d = directions[0][t][idx]
        return torch.tensor(d).float() * step


def create_random_direction_joint(net, subspace='all', ignore='biasbn', norm='filter'):
    """
    Create a random direction in the joint [W, alpha, beta] parameter space.
    
    Normalization:
      - W: filter normalization (reuses net_plotter.normalize_direction)
      - alpha: layer-wise L2 normalization
      - beta: layer-wise L2 normalization
    
    Args:
        net: PruningQuantizationWrapper
        subspace: 'all' | 'weights_only' | 'alpha_only' | 'beta_only'
        ignore: 'biasbn' to zero out bias/BN directions
        norm: normalization for weight directions ('filter', 'layer', etc.)
    
    Returns:
        (w_direction, a_direction, b_direction) - lists of tensors
    """
    weights, alphas, betas = get_joint_params(net)
    
    # Generate random directions for each sub-space
    if subspace in ('all', 'weights_only'):
        w_direction = [torch.randn_like(w) for w in weights]
        # Apply filter normalization to weight directions
        net_plotter.normalize_directions_for_weights(w_direction, weights, norm, ignore)
    else:
        w_direction = [torch.zeros_like(w) for w in weights]
    
    if subspace in ('all', 'alpha_only'):
        a_direction = [torch.randn_like(a) for a in alphas]
        # Layer-wise L2 normalization for alpha directions
        for d, ref in zip(a_direction, alphas):
            ref_norm = ref.norm()
            d_norm = d.norm()
            if d_norm > 1e-10:
                d.mul_(ref_norm / d_norm)
    else:
        a_direction = [torch.zeros_like(a) for a in alphas]
    
    if subspace in ('all', 'beta_only'):
        b_direction = [torch.randn_like(b) for b in betas]
        # Layer-wise L2 normalization for beta directions
        for d, ref in zip(b_direction, betas):
            ref_norm = ref.norm()
            d_norm = d.norm()
            if d_norm > 1e-10:
                d.mul_(ref_norm / d_norm)
    else:
        b_direction = [torch.zeros_like(b) for b in betas]
    
    return w_direction, a_direction, b_direction


def setup_direction_joint(args, dir_file, net):
    """
    Setup the HDF5 direction file for joint model.
    Stores xdirection and ydirection, each as (w_dir, a_dir, b_dir).
    """
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    
    if os.path.exists(dir_file):
        f = h5py.File(dir_file, 'r')
        if 'xdirection_w' in f.keys():
            f.close()
            print(f"{dir_file} already set up")
            return
        f.close()
    
    f = h5py.File(dir_file, 'w')
    
    subspace = getattr(args, 'subspace', 'all')
    
    # X direction
    w_dx, a_dx, b_dx = create_random_direction_joint(
        net, subspace=subspace, ignore=args.xignore, norm=args.xnorm
    )
    h5_util.write_list(f, 'xdirection_w', w_dx)
    h5_util.write_list(f, 'xdirection_a', a_dx)
    h5_util.write_list(f, 'xdirection_b', b_dx)
    
    # Y direction (if 2D)
    if args.y:
        w_dy, a_dy, b_dy = create_random_direction_joint(
            net, subspace=subspace, ignore=args.yignore, norm=args.ynorm
        )
        h5_util.write_list(f, 'ydirection_w', w_dy)
        h5_util.write_list(f, 'ydirection_a', a_dy)
        h5_util.write_list(f, 'ydirection_b', b_dy)
    
    f.close()
    print(f"Joint direction file created: {dir_file}")


def load_directions_joint(dir_file):
    """
    Load joint directions from HDF5.
    
    Returns:
        list of (w_dir, a_dir, b_dir) tuples, length 1 (1D) or 2 (2D)
    """
    f = h5py.File(dir_file, 'r')
    
    w_dx = h5_util.read_list(f, 'xdirection_w')
    a_dx = h5_util.read_list(f, 'xdirection_a')
    b_dx = h5_util.read_list(f, 'xdirection_b')
    
    directions = [(w_dx, a_dx, b_dx)]
    
    if 'ydirection_w' in f.keys():
        w_dy = h5_util.read_list(f, 'ydirection_w')
        a_dy = h5_util.read_list(f, 'ydirection_a')
        b_dy = h5_util.read_list(f, 'ydirection_b')
        directions.append((w_dy, a_dy, b_dy))
    
    f.close()
    return directions
