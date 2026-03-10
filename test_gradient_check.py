"""
Quick verification that gradient flow works correctly through JointConv2d.

Tests:
1. Gradients flow to conv weights (not zero/NaN)
2. Gradients flow to pruning alphas
3. Gradients flow to bitwidth betas
4. Forward pass produces valid outputs
5. STE quantization correctness
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from joint_model import PruningQuantizationWrapper, ste_quantize
from cifar10.models.resnet import ResNet56


def test_ste_quantize():
    """Test STE: forward should quantize, backward should pass gradient through."""
    print("Test 1: STE quantization")
    w = torch.randn(4, 3, 3, 3, requires_grad=True)
    w_q = ste_quantize(w, 8)

    # Forward: w_q should differ from w (quantized)
    assert not torch.allclose(w_q, w, atol=1e-6), "w_q should differ from w"

    # Backward: gradient should flow
    loss = w_q.sum()
    loss.backward()
    assert w.grad is not None, "Gradient should not be None"
    assert not torch.isnan(w.grad).any(), "Gradient should not contain NaN"
    # STE: gradient of sum(w_q) w.r.t. w should be all 1s
    assert torch.allclose(w.grad, torch.ones_like(w.grad)), \
        f"STE gradient should be all 1s, got min={w.grad.min():.4f}, max={w.grad.max():.4f}"
    print("  PASSED: STE forward quantizes, backward is identity")


def test_gradient_flow():
    """Test that gradients flow to all parameter groups."""
    print("\nTest 2: Gradient flow through PruningQuantizationWrapper")

    base_model = ResNet56()
    net = PruningQuantizationWrapper(base_model, bit_widths=[2, 4, 8, 16])

    # Random input
    x = torch.randn(2, 3, 32, 32)
    target = torch.tensor([3, 7])

    criterion = nn.CrossEntropyLoss()
    out = net(x)
    loss = criterion(out, target)
    loss.backward()

    # Check weight gradients
    w_grads_ok = 0
    w_grads_total = 0
    for name, p in net.base_model.named_parameters():
        if p.grad is not None:
            w_grads_total += 1
            if p.grad.abs().sum() > 0:
                w_grads_ok += 1
            assert not torch.isnan(p.grad).any(), f"NaN in grad of {name}"
    print(f"  Weight gradients: {w_grads_ok}/{w_grads_total} non-zero")
    assert w_grads_ok > 0, "At least some weight gradients should be non-zero"
    print("  PASSED: Weight gradients flow correctly")

    # Check alpha gradients
    alpha_grads_ok = 0
    for name, p in net.pruning_alphas.named_parameters():
        assert p.grad is not None, f"No gradient for alpha {name}"
        assert not torch.isnan(p.grad).any(), f"NaN in grad of alpha {name}"
        if p.grad.abs().sum() > 0:
            alpha_grads_ok += 1
    print(f"  Alpha gradients: {alpha_grads_ok}/{len(list(net.pruning_alphas.parameters()))} non-zero")
    assert alpha_grads_ok > 0, "At least some alpha gradients should be non-zero"
    print("  PASSED: Pruning alpha gradients flow correctly")

    # Check beta gradients
    beta_grads_ok = 0
    for name, p in net.bitwidth_betas.named_parameters():
        assert p.grad is not None, f"No gradient for beta {name}"
        assert not torch.isnan(p.grad).any(), f"NaN in grad of beta {name}"
        if p.grad.abs().sum() > 0:
            beta_grads_ok += 1
    print(f"  Beta gradients: {beta_grads_ok}/{len(list(net.bitwidth_betas.parameters()))} non-zero")
    assert beta_grads_ok > 0, "At least some beta gradients should be non-zero"
    print("  PASSED: Bit-width beta gradients flow correctly")


def test_forward_correctness():
    """Test that forward pass produces valid class logits."""
    print("\nTest 3: Forward pass correctness")

    base_model = ResNet56()
    net = PruningQuantizationWrapper(base_model, bit_widths=[2, 4, 8, 16])
    net.eval()

    x = torch.randn(4, 3, 32, 32)
    out = net(x)

    assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"
    print(f"  Output shape: {out.shape}")
    print(f"  Output range: [{out.min():.3f}, {out.max():.3f}]")
    print("  PASSED: Forward produces valid logits")


def test_optimizer_step():
    """Test that an optimizer step actually changes weights."""
    print("\nTest 4: Optimizer step changes parameters")

    base_model = ResNet56()
    net = PruningQuantizationWrapper(base_model, bit_widths=[2, 4, 8, 16])

    # Snapshot initial weights
    initial_w = {}
    for name, p in net.named_parameters():
        initial_w[name] = p.data.clone()

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(2, 3, 32, 32)
    target = torch.tensor([3, 7])

    optimizer.zero_grad()
    out = net(x)
    loss = criterion(out, target)
    loss.backward()
    optimizer.step()

    # Check that weights changed
    changed = 0
    total = 0
    for name, p in net.named_parameters():
        total += 1
        if not torch.allclose(p.data, initial_w[name], atol=1e-10):
            changed += 1
    print(f"  Parameters changed: {changed}/{total}")
    assert changed > 0, "At least some parameters should change after optimizer step"
    print("  PASSED: Optimizer step updates parameters")


def test_lagrangian_components():
    """Test that Lagrangian components are differentiable and reasonable."""
    print("\nTest 5: Lagrangian components")

    base_model = ResNet56()
    net = PruningQuantizationWrapper(base_model, bit_widths=[2, 4, 8, 16])
    net.set_temperature(1.0)

    # Model size
    size = net.compute_model_size()
    assert size.requires_grad, "Model size should be differentiable"
    assert size.item() > 0, "Model size should be positive"
    print(f"  Model size: {size.item():.0f} bits ({size.item()/8/1024:.1f} KB)")

    # Sparsity reg (normalized)
    reg = net.compute_sparsity_reg()
    assert reg.requires_grad, "Sparsity reg should be differentiable"
    assert 0 <= reg.item() <= 1.0, f"Normalized reg should be in [0,1], got {reg.item():.4f}"
    print(f"  Sparsity reg (normalized): {reg.item():.4f}")

    # Sparsity ratio
    sparsity = net.get_sparsity_ratio()
    print(f"  Sparsity ratio: {sparsity:.2%}")
    assert sparsity < 0.01, "Initially all gates should be open (near 0% pruned)"
    print("  PASSED: Lagrangian components are valid and differentiable")


if __name__ == '__main__':
    print("=" * 60)
    print("Joint Model Gradient Flow Verification")
    print("=" * 60)

    test_ste_quantize()
    test_gradient_flow()
    test_forward_correctness()
    test_optimizer_step()
    test_lagrangian_components()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
