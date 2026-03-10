"""
4-Stage training script for joint pruning + mixed-precision quantization.

Stage 1 - Warm-up:       Train W only (freeze alpha, beta) — verify pretrained accuracy holds
Stage 2 - Joint search:  Train W + alpha + beta with Lagrangian
Stage 3 - Discretize:    Snap alpha to {0,1}, beta to argmax
Stage 4 - Fine-tune:     Retrain W with fixed structure (QAT)

Usage:
  python train_joint.py --model resnet56 \
    --pretrained cifar10/trained_nets/resnet56_sgd_lr=0.1_.../model_300.t7 \
    --lambda_s 0.01 --gamma 0.001 --S_max_ratio 0.5 \
    --cuda --save_dir cifar10/trained_nets/resnet56_joint
"""

import argparse
import os
import sys
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from joint_model import PruningQuantizationWrapper
import cifar10.model_loader as cifar10_loader


def get_cifar10_loaders(batch_size=128, num_workers=2):
    """Load CIFAR-10 train and test data with standard augmentation."""
    import torchvision
    import torchvision.transforms as transforms

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='cifar10/data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(
        root='cifar10/data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


def evaluate(net, testloader, criterion, use_cuda):
    """Evaluate accuracy and loss on test set."""
    net.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return total_loss / total, 100.0 * correct / total


def cosine_annealing(start_val, end_val, epoch, total_epochs):
    """Cosine annealing schedule."""
    return end_val + 0.5 * (start_val - end_val) * (1 + math.cos(math.pi * epoch / total_epochs))


def check_for_nan(net, epoch, stage_name):
    """Check for NaN in parameters and report."""
    for name, p in net.named_parameters():
        if torch.isnan(p.data).any():
            print(f"  [WARNING] NaN detected in {name} at epoch {epoch} during {stage_name}")
            return True
        if p.grad is not None and torch.isnan(p.grad).any():
            print(f"  [WARNING] NaN gradient in {name} at epoch {epoch} during {stage_name}")
            return True
    return False


# =============================================================================
# Stage 1: Warm-up
# =============================================================================
def train_warmup(net, trainloader, testloader, args):
    """
    Warm-up: train W only with CE loss, frozen gates.
    Purpose: adapt weights to work through the quantization/gating pipeline.
    """
    print("\n" + "=" * 60)
    print("STAGE 1: Warm-up (train W only, gates frozen at ~1)")
    print("=" * 60)

    net.freeze_structure()  # Freeze alpha, beta
    net.unfreeze_weights()

    criterion = nn.CrossEntropyLoss()

    # Use SGD — same optimizer as original training, to stay in the same basin
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=args.warmup_lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.warmup_epochs)

    if args.cuda:
        net.cuda()
        criterion.cuda()

    # Check initial accuracy (should be close to pretrained)
    init_loss, init_acc = evaluate(net, testloader, criterion, args.cuda)
    print(f"  Initial test accuracy: {init_acc:.2f}% (should be ~93% if pretrained)")

    for epoch in range(args.warmup_epochs):
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if args.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()

        test_loss, test_acc = evaluate(net, testloader, criterion, args.cuda)
        print(f"  Epoch {epoch+1}/{args.warmup_epochs} | "
              f"Train Loss: {train_loss/total:.4f} | Train Acc: {100.*correct/total:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    # Save checkpoint
    save_path = os.path.join(args.save_dir, 'stage1_warmup.pth')
    torch.save(net.state_dict(), save_path)
    print(f"  Saved: {save_path}")

    return net


# =============================================================================
# Stage 2: Joint Search
# =============================================================================
def train_joint_search(net, trainloader, testloader, args):
    """Joint search: train W + alpha + beta with Lagrangian."""
    print("\n" + "=" * 60)
    print("STAGE 2: Joint Search (Lagrangian optimization)")
    print("=" * 60)

    net.unfreeze_structure()
    net.unfreeze_weights()

    criterion = nn.CrossEntropyLoss()

    # Separate optimizer groups
    # Key insight: use SGD for weights (stays near pretrained basin),
    # Adam for structure params (alpha, beta) since they have different gradient scales
    w_params = []
    for name, p in net.named_parameters():
        if 'pruning_alphas' not in name and 'bitwidth_betas' not in name and p.requires_grad:
            w_params.append(p)

    a_params = list(net.pruning_alphas.parameters())
    b_params = list(net.bitwidth_betas.parameters())

    optimizer_w = optim.SGD(w_params, lr=args.joint_lr_w, momentum=0.9, weight_decay=5e-4)
    optimizer_struct = optim.Adam(
        [{'params': a_params, 'lr': args.joint_lr_alpha},
         {'params': b_params, 'lr': args.joint_lr_beta}],
        weight_decay=0
    )

    scheduler_w = optim.lr_scheduler.CosineAnnealingLR(optimizer_w, T_max=args.joint_epochs)

    if args.cuda:
        net.cuda()
        criterion.cuda()

    # Compute S_max from ratio
    with torch.no_grad():
        baseline_size = 0
        for ln in net.conv_layer_names:
            info = net.conv_layer_info[ln]
            baseline_size += 32 * info['out_channels'] * info['params_per_filter']
    S_max = args.S_max_ratio * baseline_size
    print(f"  Baseline size: {baseline_size} bits ({baseline_size/8/1024:.1f} KB)")
    print(f"  Target S_max:  {S_max:.0f} bits ({S_max/8/1024:.1f} KB) = {args.S_max_ratio:.0%} of baseline")

    # Check initial accuracy before joint search
    init_loss, init_acc = evaluate(net, testloader, criterion, args.cuda)
    print(f"  Initial accuracy: {init_acc:.2f}%")

    # Dual variable (Lagrange multiplier) — start small
    lambda_s = args.lambda_s

    best_acc = 0

    for epoch in range(args.joint_epochs):
        net.train()

        # Anneal temperature: cosine from tau_start to tau_end
        tau = cosine_annealing(args.tau_start, args.tau_end, epoch, args.joint_epochs)
        net.set_temperature(tau)

        train_loss = 0
        total = 0
        correct = 0
        epoch_lagrangian = 0
        epoch_size_pen = 0
        epoch_sparse_reg = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if args.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            optimizer_w.zero_grad()
            optimizer_struct.zero_grad()

            outputs = net(inputs)
            task_loss = criterion(outputs, targets)

            # Compute Lagrangian penalties
            model_size = net.compute_model_size()
            size_penalty = lambda_s * F.relu(model_size - S_max)
            sparsity_reg = args.gamma * net.compute_sparsity_reg()

            lagrangian = task_loss + size_penalty + sparsity_reg
            lagrangian.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(w_params, max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(a_params + b_params, max_norm=1.0)

            optimizer_w.step()
            optimizer_struct.step()

            train_loss += task_loss.item() * inputs.size(0)
            epoch_lagrangian += lagrangian.item() * inputs.size(0)
            epoch_size_pen += size_penalty.item() * inputs.size(0)
            epoch_sparse_reg += sparsity_reg.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler_w.step()

        # Dual variable update every 5 epochs (slow, conservative)
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                current_size = net.compute_model_size().item()
                constraint_violation = (current_size - S_max) / baseline_size  # normalized
                lambda_s = max(0, lambda_s + args.dual_lr * constraint_violation)

        test_loss, test_acc = evaluate(net, testloader, criterion, args.cuda)
        sparsity = net.get_sparsity_ratio()

        if test_acc > best_acc:
            best_acc = test_acc

        # Check for NaN
        if check_for_nan(net, epoch, "Stage 2"):
            print("  [ERROR] NaN detected, stopping joint search early")
            break

        print(f"  Epoch {epoch+1}/{args.joint_epochs} | "
              f"CE: {train_loss/total:.4f} | Size: {epoch_size_pen/total:.4f} | "
              f"Sparse: {epoch_sparse_reg/total:.4f} | "
              f"Acc: {test_acc:.2f}% | Sparsity: {sparsity:.2%} | "
              f"tau: {tau:.3f} | lam: {lambda_s:.4f}")

    print(f"\n  Best accuracy during joint search: {best_acc:.2f}%")

    # Save checkpoint
    save_path = os.path.join(args.save_dir, 'stage2_joint.pth')
    torch.save({
        'state_dict': net.state_dict(),
        'lambda_s': lambda_s,
        'S_max': S_max,
    }, save_path)
    print(f"  Saved: {save_path}")

    return net


# =============================================================================
# Stage 3: Discretize
# =============================================================================
def discretize(net, args):
    """Discretize: snap gates and bit-widths to discrete values."""
    print("\n" + "=" * 60)
    print("STAGE 3: Discretization")
    print("=" * 60)

    net.discretize(threshold=args.gate_threshold)
    net.get_compression_summary()

    save_path = os.path.join(args.save_dir, 'stage3_discrete.pth')
    torch.save(net.state_dict(), save_path)
    print(f"  Saved: {save_path}")

    return net


# =============================================================================
# Stage 4: Fine-tune
# =============================================================================
def train_finetune(net, trainloader, testloader, args):
    """Fine-tune: retrain W with frozen discrete structure."""
    print("\n" + "=" * 60)
    print("STAGE 4: Fine-tuning (QAT with fixed structure)")
    print("=" * 60)

    net.freeze_structure()
    net.unfreeze_weights()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=args.finetune_lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)

    if args.cuda:
        net.cuda()
        criterion.cuda()

    # Check accuracy after discretization
    init_loss, init_acc = evaluate(net, testloader, criterion, args.cuda)
    print(f"  Post-discretization accuracy: {init_acc:.2f}%")

    best_acc = 0

    for epoch in range(args.finetune_epochs):
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if args.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()

        test_loss, test_acc = evaluate(net, testloader, criterion, args.cuda)
        print(f"  Epoch {epoch+1}/{args.finetune_epochs} | "
              f"Train Loss: {train_loss/total:.4f} | Train Acc: {100.*correct/total:.2f}% | "
              f"Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(args.save_dir, 'stage4_best.pth')
            torch.save(net.state_dict(), save_path)

    # Save final
    save_path = os.path.join(args.save_dir, 'final.pth')
    torch.save(net.state_dict(), save_path)
    print(f"  Best test accuracy: {best_acc:.2f}%")
    print(f"  Saved: {save_path}")

    net.get_compression_summary()

    return net


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Joint Pruning + Quantization Training')

    # Model
    parser.add_argument('--model', default='resnet56', help='model name')
    parser.add_argument('--pretrained', default='', help='path to pretrained baseline model')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--save_dir', default='cifar10/trained_nets/resnet56_joint',
                        help='directory to save checkpoints')

    # Data
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=2, type=int)

    # Bit-widths
    parser.add_argument('--bit_widths', default='2,4,8,16', type=str,
                        help='comma-separated candidate bit-widths')

    # Stage 1: Warm-up
    parser.add_argument('--warmup_epochs', default=10, type=int)
    parser.add_argument('--warmup_lr', default=0.005, type=float,
                        help='small LR to adapt to quantization without losing accuracy')

    # Stage 2: Joint search
    parser.add_argument('--joint_epochs', default=100, type=int)
    parser.add_argument('--joint_lr_w', default=0.01, type=float, help='SGD LR for weights')
    parser.add_argument('--joint_lr_alpha', default=0.01, type=float, help='Adam LR for pruning gates')
    parser.add_argument('--joint_lr_beta', default=0.01, type=float, help='Adam LR for bit-width logits')
    parser.add_argument('--lambda_s', default=0.01, type=float,
                        help='initial Lagrange multiplier (start small, dual update grows it)')
    parser.add_argument('--gamma', default=0.001, type=float,
                        help='sparsity regularization weight (small: reg is already normalized)')
    parser.add_argument('--S_max_ratio', default=0.5, type=float,
                        help='target size as fraction of baseline (e.g. 0.5 = 50%%)')
    parser.add_argument('--tau_start', default=5.0, type=float, help='initial Gumbel temperature')
    parser.add_argument('--tau_end', default=0.5, type=float,
                        help='final Gumbel temperature (higher floor for stability)')
    parser.add_argument('--dual_lr', default=0.01, type=float, help='dual variable learning rate')
    parser.add_argument('--gate_threshold', default=0.5, type=float,
                        help='sigmoid threshold for discretization (0.5 is natural midpoint)')

    # Stage 4: Fine-tune
    parser.add_argument('--finetune_epochs', default=100, type=int)
    parser.add_argument('--finetune_lr', default=0.01, type=float)

    # Stage selection
    parser.add_argument('--start_stage', default=1, type=int, choices=[1, 2, 3, 4],
                        help='start from this stage (resume)')
    parser.add_argument('--load_checkpoint', default='', help='checkpoint to resume from')

    args = parser.parse_args()

    # Parse bit-widths
    bit_widths = [int(b) for b in args.bit_widths.split(',')]

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load base model
    print(f"Loading base model: {args.model}")
    if args.pretrained:
        base_net = cifar10_loader.load(args.model, args.pretrained)
        print(f"  Loaded pretrained weights from {args.pretrained}")
    else:
        models = cifar10_loader.models
        base_net = models[args.model]()
        print(f"  Initialized from scratch (WARNING: pretrained recommended)")

    # Wrap with pruning + quantization
    net = PruningQuantizationWrapper(base_net, bit_widths=bit_widths)

    # Resume from checkpoint if specified
    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            net.load_state_dict(checkpoint['state_dict'])
        else:
            net.load_state_dict(checkpoint)
        print(f"  Resumed from {args.load_checkpoint}")

    # Load data
    trainloader, testloader = get_cifar10_loaders(args.batch_size, args.num_workers)

    # Quick sanity check: evaluate pretrained accuracy
    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        net.cuda()
        criterion.cuda()
    init_loss, init_acc = evaluate(net, testloader, criterion, args.cuda)
    print(f"\n  Pretrained accuracy through wrapper: {init_acc:.2f}%")
    if init_acc < 50.0 and args.pretrained:
        print("  [WARNING] Accuracy is very low for a pretrained model!")
        print("  Check that you are using the fully trained checkpoint (e.g., model_300.t7)")

    # Run stages
    if args.start_stage <= 1:
        net = train_warmup(net, trainloader, testloader, args)

    if args.start_stage <= 2:
        net = train_joint_search(net, trainloader, testloader, args)

    if args.start_stage <= 3:
        net = discretize(net, args)

    if args.start_stage <= 4:
        net = train_finetune(net, trainloader, testloader, args)

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
