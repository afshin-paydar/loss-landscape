"""
Joint Pruning & Mixed-Precision Quantization Wrapper for Loss Landscape Visualization.

This module wraps an existing neural network (e.g. ResNet_cifar) with:
  - Per-channel pruning gates via sigmoid(alpha)
  - Per-layer mixed-precision quantization via Gumbel-Softmax over bit-width candidates
  - Straight-Through Estimator (STE) for quantization gradients

The key design: each Conv2d is REPLACED with a JointConv2d module that uses
F.conv2d with STE-quantized weights. This keeps the entire computation in the
autograd graph — no hooks, no weight.data.copy_().

Reference: Mathematical formulation from "Joint Pruning & Mixed-Precision Quantization
Objective Function in DNNs" document, adapted from Li et al. (NeurIPS 2018).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


# =============================================================================
# STE Quantization (proper gradient flow)
# =============================================================================
def ste_quantize(w, b):
    """
    Uniform symmetric quantization with Straight-Through Estimator.

    Forward:  returns Q_b(w) = clamp(round(w/s), -qmax-1, qmax) * s
    Backward: gradient passes through as identity (STE)

    The trick: w + (Q(w) - w).detach()
      - Evaluates to Q(w) in forward (since detach doesn't affect value)
      - Has gradient dQ/dw = 1 in backward (since (Q(w)-w).detach() has zero gradient)

    Args:
        w: weight tensor (any shape)
        b: number of bits (int or float)
    Returns:
        quantized weight tensor with STE gradient
    """
    if b >= 32:
        return w  # Full precision

    # Quantization parameters
    qmax = 2 ** (b - 1) - 1  # e.g., b=8 -> qmax=127
    w_max = w.detach().abs().max().clamp(min=1e-8)
    s = w_max / qmax

    # Quantize (no gradient through round/clamp)
    w_q = torch.clamp(torch.round(w.detach() / s), -qmax - 1, qmax) * s

    # STE: forward = w_q, backward gradient flows through w
    return w + (w_q - w).detach()


# =============================================================================
# Gumbel-Softmax for differentiable bit-width selection
# =============================================================================
def gumbel_softmax(logits, tau=1.0, hard=False):
    """
    Gumbel-Softmax sampling for categorical distribution over bit-widths.

    Args:
        logits: unnormalized log-probabilities [K]
        tau: temperature (lower = more discrete)
        hard: if True, return one-hot in forward but soft in backward
    Returns:
        soft (or hard) categorical sample
    """
    # Sample Gumbel noise
    gumbels = -torch.empty_like(logits).exponential_().clamp(min=1e-10).log()
    gumbels = (logits + gumbels) / max(tau, 0.01)  # Floor tau for safety
    y_soft = F.softmax(gumbels, dim=-1)

    if hard:
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        return y_hard - y_soft.detach() + y_soft

    return y_soft


# =============================================================================
# JointConv2d: Conv2d replacement with pruning + quantization
# =============================================================================
class SharedConfig:
    """
    Non-Module container for shared config between wrapper and JointConv2d.
    Using a plain Python object avoids circular nn.Module references that
    cause infinite recursion during .eval() / .train().
    """
    def __init__(self, bit_widths, temperature=1.0):
        self.bit_widths = bit_widths
        self.K = len(bit_widths)
        self.temperature = temperature


class JointConv2d(nn.Module):
    """
    Drop-in replacement for nn.Conv2d that applies pruning gates and
    mixed-precision quantization in the forward pass.

    The original Conv2d's weight is transferred here as a proper nn.Parameter,
    so gradients flow correctly through the quantization STE.
    """

    def __init__(self, original_conv, alpha, beta, shared_config):
        super().__init__()

        # Transfer the weight as our own parameter (keeps gradient connection)
        self.weight = original_conv.weight  # nn.Parameter
        self.bias = original_conv.bias      # None for ResNet (bias=False)

        # Conv metadata
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups
        self.out_channels = original_conv.out_channels
        self.in_channels = original_conv.in_channels
        self.kernel_size = original_conv.kernel_size

        # References to shared learnable parameters (owned by wrapper)
        self.alpha = alpha    # pruning gate logits [C_out]
        self.beta = beta      # bit-width logits [K]
        # Store config as plain Python object (not nn.Module) to avoid circular refs
        self._config = shared_config

    def forward(self, x):
        cfg = self._config

        # --- Pruning gate ---
        gate = torch.sigmoid(self.alpha)  # [C_out], values in (0, 1)

        # Apply gate: scale each output filter
        gated_w = self.weight * gate.view(-1, 1, 1, 1)

        # --- Mixed-precision quantization ---
        if self.training:
            pi = gumbel_softmax(self.beta, tau=cfg.temperature, hard=False)
        else:
            # At eval time, use hard argmax
            pi = F.one_hot(self.beta.argmax(), num_classes=cfg.K).float()
            if self.beta.is_cuda:
                pi = pi.cuda()

        # Weighted sum of quantized weights: w_q = sum_k pi_k * Q_{b_k}(gated_w)
        # Each Q_{b_k} uses STE so gradients flow back to self.weight and self.alpha
        w_q = torch.zeros_like(gated_w)
        for k in range(cfg.K):
            bw = cfg.bit_widths[k]
            q_w = ste_quantize(gated_w, bw)
            w_q = w_q + pi[k] * q_w

        # --- Convolution with quantized weights ---
        return F.conv2d(x, w_q, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


# =============================================================================
# Main Wrapper: PruningQuantizationWrapper
# =============================================================================
class PruningQuantizationWrapper(nn.Module):
    """
    Wraps a base model with per-layer pruning gates and mixed-precision quantization.

    For each Conv2d layer in the base model:
      - Adds pruning gate alpha_l of shape [C_out], gate = sigmoid(alpha_l)
      - Adds bit-width logits beta_l of shape [K], pi = Gumbel-Softmax(beta_l)

    Implementation: Conv2d modules are REPLACED with JointConv2d modules that
    apply gating + quantization via F.conv2d. This keeps the full autograd graph
    intact — no hooks, no weight.data.copy_().

    Args:
        base_model: nn.Module (e.g. ResNet_cifar)
        bit_widths: list of candidate bit-widths (default: [2, 4, 8, 16])
        init_alpha: initial value for pruning logits (default: 3.0 -> sigmoid ~ 0.95)
        temperature: initial Gumbel-Softmax temperature
    """

    def __init__(self, base_model, bit_widths=None, init_alpha=3.0, temperature=1.0):
        super().__init__()

        if bit_widths is None:
            bit_widths = [2, 4, 8, 16]

        self.base_model = base_model
        self.bit_widths = bit_widths
        self.K = len(bit_widths)
        self.init_alpha = init_alpha

        # Shared config object (plain Python, not nn.Module) to pass to JointConv2d
        self._shared_config = SharedConfig(bit_widths, temperature)

        # Register bit-widths as a buffer (non-learnable constant)
        self.register_buffer('bit_width_tensor', torch.tensor(bit_widths, dtype=torch.float32))

        # Discover all Conv2d layers and create pruning/quantization parameters
        self.conv_layer_names = []
        self.conv_layer_info = OrderedDict()

        self.pruning_alphas = nn.ParameterDict()
        self.bitwidth_betas = nn.ParameterDict()

        # First pass: collect info and create parameters
        conv_modules = []
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Conv2d):
                safe_name = name.replace('.', '_')
                self.conv_layer_names.append(safe_name)
                conv_modules.append((name, safe_name, module))

                out_channels = module.out_channels
                in_channels = module.in_channels
                kernel_size = module.kernel_size
                params_per_filter = in_channels * kernel_size[0] * kernel_size[1]

                self.conv_layer_info[safe_name] = {
                    'out_channels': out_channels,
                    'in_channels': in_channels,
                    'kernel_size': kernel_size,
                    'params_per_filter': params_per_filter,
                    'module_name': name,
                }

                # Pruning gate logits: sigmoid(3.0) ~ 0.95, so gates start mostly open
                alpha = nn.Parameter(torch.full((out_channels,), init_alpha))
                self.pruning_alphas[safe_name] = alpha

                # Bit-width logits: bias toward higher precision initially
                # [2, 4, 8, 16] -> init logits [0, 0, 0.5, 1.0]
                beta_init = torch.linspace(0, 1.0, len(bit_widths))
                beta = nn.Parameter(beta_init.clone())
                self.bitwidth_betas[safe_name] = beta

        self.num_conv_layers = len(self.conv_layer_names)

        # Second pass: replace Conv2d with JointConv2d
        for name, safe_name, module in conv_modules:
            alpha = self.pruning_alphas[safe_name]
            beta = self.bitwidth_betas[safe_name]
            joint_conv = JointConv2d(module, alpha, beta, self._shared_config)
            self._set_module(self.base_model, name, joint_conv)

        # Total number of channels (for normalizing sparsity regularizer)
        self.total_channels = sum(
            info['out_channels'] for info in self.conv_layer_info.values()
        )

        print(f"[JointModel] Replaced {self.num_conv_layers} Conv2d -> JointConv2d")
        print(f"[JointModel] Bit-width candidates: {bit_widths}")
        print(f"[JointModel] Total channels: {self.total_channels}")

    @staticmethod
    def _set_module(model, name, new_module):
        """Replace a submodule in model by dotted name path."""
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    def forward(self, x):
        """Forward pass through the pruned + quantized model."""
        return self.base_model(x)

    @property
    def temperature(self):
        return self._shared_config.temperature

    def set_temperature(self, tau):
        """Set Gumbel-Softmax temperature for bit-width selection."""
        self._shared_config.temperature = max(tau, 0.05)  # Floor to avoid numerical issues

    def compute_model_size(self):
        """
        Compute C_size(alpha, beta) = sum_l b_eff^l * ||gate^l||_0_soft * K_l

        Uses soft approximations:
          - ||gate||_0 ~ sum(sigmoid(alpha))
          - b_eff = sum_k pi_k * b_k

        Returns:
            total model size in bits (differentiable)
        """
        total_bits = torch.tensor(0.0, device=self._get_device())

        for layer_name in self.conv_layer_names:
            alpha = self.pruning_alphas[layer_name]
            beta = self.bitwidth_betas[layer_name]
            info = self.conv_layer_info[layer_name]

            gate = torch.sigmoid(alpha)
            soft_channels = gate.sum()

            pi = F.softmax(beta / max(self.temperature, 0.05), dim=-1)
            b_eff = (pi * self.bit_width_tensor).sum()

            K_l = info['params_per_filter']
            total_bits = total_bits + b_eff * soft_channels * K_l

        return total_bits

    def compute_bops(self):
        """
        Compute C_ops(alpha, beta) ~ sum_l b_eff^2 * channels * H * W * K_l

        Returns:
            total BOPs (differentiable)
        """
        total_bops = torch.tensor(0.0, device=self._get_device())
        spatial = 32  # CIFAR-10 input size

        for layer_name in self.conv_layer_names:
            alpha = self.pruning_alphas[layer_name]
            beta = self.bitwidth_betas[layer_name]
            info = self.conv_layer_info[layer_name]

            gate = torch.sigmoid(alpha)
            soft_channels = gate.sum()

            pi = F.softmax(beta / max(self.temperature, 0.05), dim=-1)
            b_eff = (pi * self.bit_width_tensor).sum()

            K_l = info['params_per_filter']
            layer_bops = b_eff * b_eff * soft_channels * spatial * spatial * K_l
            total_bops = total_bops + layer_bops

        return total_bops

    def compute_sparsity_reg(self):
        """
        Compute NORMALIZED sparsity regularizer:
            R(alpha) = (1 / N_total) * sum_{l,j} sigmoid(alpha_j^l)

        Normalized by total channel count so gamma doesn't need to scale with model size.

        Returns:
            regularization term in [0, 1] (differentiable)
        """
        reg = torch.tensor(0.0, device=self._get_device())
        for layer_name in self.conv_layer_names:
            alpha = self.pruning_alphas[layer_name]
            reg = reg + torch.sigmoid(alpha).sum()
        return reg / self.total_channels

    def get_sparsity_ratio(self):
        """Get the fraction of pruned channels (non-differentiable metric)."""
        total = 0
        pruned = 0
        threshold = 0.01

        for layer_name in self.conv_layer_names:
            alpha = self.pruning_alphas[layer_name]
            gate = torch.sigmoid(alpha)
            total += gate.numel()
            pruned += (gate < threshold).sum().item()

        return pruned / max(total, 1)

    def get_effective_bitwidths(self):
        """Get the selected bit-width per layer (argmax)."""
        result = OrderedDict()
        for layer_name in self.conv_layer_names:
            beta = self.bitwidth_betas[layer_name]
            idx = beta.argmax().item()
            result[layer_name] = self.bit_widths[idx]
        return result

    def discretize(self, threshold=0.5):
        """
        Stage 3: Snap pruning gates to {0,1} and bit-widths to argmax.

        Args:
            threshold: sigmoid gate values below this are pruned
        """
        with torch.no_grad():
            for layer_name in self.conv_layer_names:
                # Discretize pruning gates
                alpha = self.pruning_alphas[layer_name]
                gate = torch.sigmoid(alpha)
                # Set alpha to extreme values to represent 0/1 after sigmoid
                alpha.data = torch.where(
                    gate >= threshold,
                    torch.tensor(10.0, device=alpha.device),   # sigmoid(10) ~ 1
                    torch.tensor(-10.0, device=alpha.device),  # sigmoid(-10) ~ 0
                )

                # Discretize bit-widths: set argmax logit high, others low
                beta = self.bitwidth_betas[layer_name]
                idx = beta.argmax()
                new_beta = torch.full_like(beta, -10.0)
                new_beta[idx] = 10.0
                beta.data = new_beta

        bw = self.get_effective_bitwidths()
        sparsity = self.get_sparsity_ratio()
        print(f"[Discretize] Sparsity: {sparsity:.2%}")
        bw_list = list(bw.values())
        unique_bw = set(bw_list)
        bw_counts = {b: bw_list.count(b) for b in sorted(unique_bw)}
        print(f"[Discretize] Bit-width distribution: {bw_counts}")

    def freeze_structure(self):
        """Freeze pruning and quantization parameters (for fine-tuning stage)."""
        for param in self.pruning_alphas.parameters():
            param.requires_grad = False
        for param in self.bitwidth_betas.parameters():
            param.requires_grad = False

    def unfreeze_structure(self):
        """Unfreeze pruning and quantization parameters."""
        for param in self.pruning_alphas.parameters():
            param.requires_grad = True
        for param in self.bitwidth_betas.parameters():
            param.requires_grad = True

    def freeze_weights(self):
        """Freeze base model weights (for structure-search-only stages)."""
        for name, param in self.base_model.named_parameters():
            # Only freeze the actual conv weights, not alpha/beta
            if 'alpha' not in name and 'beta' not in name:
                param.requires_grad = False

    def unfreeze_weights(self):
        """Unfreeze base model weights."""
        for name, param in self.base_model.named_parameters():
            if 'alpha' not in name and 'beta' not in name:
                param.requires_grad = True

    def _get_device(self):
        """Get device of the model parameters."""
        return next(self.base_model.parameters()).device

    def get_compression_summary(self):
        """Print a summary of compression achieved."""
        baseline_bits = 0
        compressed_bits = 0

        for layer_name in self.conv_layer_names:
            info = self.conv_layer_info[layer_name]
            alpha = self.pruning_alphas[layer_name]
            beta = self.bitwidth_betas[layer_name]

            C_out = info['out_channels']
            K_l = info['params_per_filter']

            baseline_bits += 32 * C_out * K_l

            gate = torch.sigmoid(alpha)
            active = (gate >= 0.5).sum().item()
            bw = self.bit_widths[beta.argmax().item()]
            compressed_bits += bw * active * K_l

        ratio = compressed_bits / max(baseline_bits, 1)
        print(f"\n{'='*60}")
        print(f"Compression Summary")
        print(f"{'='*60}")
        print(f"  Baseline size:    {baseline_bits / 8 / 1024:.1f} KB ({baseline_bits} bits)")
        print(f"  Compressed size:  {compressed_bits / 8 / 1024:.1f} KB ({compressed_bits} bits)")
        print(f"  Compression ratio: {ratio:.2%}")
        print(f"  Sparsity:         {self.get_sparsity_ratio():.2%}")
        print(f"{'='*60}\n")

        return baseline_bits, compressed_bits, ratio
