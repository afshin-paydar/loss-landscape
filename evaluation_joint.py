"""
Extended evaluation for joint pruning + mixed-precision quantization models.

Computes task loss, Lagrangian loss, and compression metrics at each point
in the loss landscape.
"""

import torch
import torch.nn as nn
from torch.autograd.variable import Variable


def eval_joint_loss(net, criterion, loader, use_cuda=False,
                    lambda_s=1.0, gamma=0.01, S_max=None):
    """
    Evaluate the joint Lagrangian loss for a PruningQuantizationWrapper model.
    
    L = L_task + lambda_s * max(0, C_size - S_max) + gamma * R(alpha)
    
    Args:
        net: PruningQuantizationWrapper model
        criterion: loss function (e.g. CrossEntropyLoss)
        loader: data loader
        use_cuda: use GPU
        lambda_s: Lagrange multiplier for size constraint
        gamma: sparsity regularization weight
        S_max: maximum model size budget in bits (None = no constraint)
    
    Returns:
        (task_loss, accuracy, lagrangian_loss, model_size_bits, sparsity_ratio)
    """
    correct = 0
    total_loss = 0
    total = 0
    
    if use_cuda:
        net.cuda()
    net.eval()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            batch_size = inputs.size(0)
            total += batch_size
            inputs = Variable(inputs)
            targets = Variable(targets)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets).sum().item()
    
    task_loss = total_loss / total
    accuracy = 100.0 * correct / total
    
    # Compute compression metrics
    from joint_model import PruningQuantizationWrapper
    if isinstance(net, PruningQuantizationWrapper):
        model_size = net.compute_model_size().item()
        sparsity = net.get_sparsity_ratio()
        sparsity_reg = net.compute_sparsity_reg().item()
    elif hasattr(net, 'module') and isinstance(net.module, PruningQuantizationWrapper):
        model_size = net.module.compute_model_size().item()
        sparsity = net.module.get_sparsity_ratio()
        sparsity_reg = net.module.compute_sparsity_reg().item()
    else:
        # Fallback for non-joint models
        model_size = 0.0
        sparsity = 0.0
        sparsity_reg = 0.0
    
    # Compute Lagrangian
    lagrangian = task_loss
    if S_max is not None and S_max > 0:
        constraint_violation = max(0.0, model_size - S_max)
        lagrangian += lambda_s * constraint_violation
    lagrangian += gamma * sparsity_reg
    
    return task_loss, accuracy, lagrangian, model_size, sparsity


def eval_constraint_metrics(net):
    """
    Compute constraint metrics for a PruningQuantizationWrapper model.
    
    Returns:
        dict with keys: model_size_bits, sparsity_ratio, effective_bitwidths
    """
    from joint_model import PruningQuantizationWrapper
    
    wrapper = net
    if hasattr(net, 'module'):
        wrapper = net.module
    
    if not isinstance(wrapper, PruningQuantizationWrapper):
        return {'model_size_bits': 0, 'sparsity_ratio': 0, 'effective_bitwidths': {}}
    
    return {
        'model_size_bits': wrapper.compute_model_size().item(),
        'sparsity_ratio': wrapper.get_sparsity_ratio(),
        'effective_bitwidths': wrapper.get_effective_bitwidths(),
    }
