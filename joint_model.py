"""
Joint Pruning & Mixed-Precision Quantization Wrapper for Loss Landscape Visualization.

This module wraps an existing neural network (e.g. ResNet_cifar) with:
  - Per-channel pruning gates via sigmoid(alpha) 
  - Per-layer mixed-precision quantization via Gumbel-Softmax over bit-width candidates
  - Straight-Through Estimator (STE) for quantization gradients

Reference: Mathematical formulation from "Joint Pruning & Mixed-Precision Quantization
Objective Function in DNNs" document, adapted from Li et al. (NeurIPS 2018).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


# =============================================================================
# Straight-Through Estimator for Quantization
# =============================================================================
class STE_Round(torch.autograd.Function):
    """Round with straight-through gradient estimator."""
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # Identity gradient


def uniform_quantize(w, b, symmetric=True):
    """
    Uniform symmetric quantization of tensor w to b-bit precision.
    
    Q_b(w) = s * clamp(round(w/s), -2^{b-1}, 2^{b-1} - 1)
    where s = max(|w|) / (2^{b-1} - 1)
    
    Args:
        w: weight tensor
        b: number of bits (float, for soft bit-width selection)
        symmetric: use symmetric quantization
    Returns:
        quantized weight tensor (same shape as w)
    """
    if b >= 32:
        return w  # Full precision, no quantization
    
    # Compute scale factor
    n = 2 ** (b - 1) - 1  # e.g. b=8 -> n=127
    # Avoid division by zero
    w_max = w.abs().max().clamp(min=1e-8)
    s = w_max / n
    
    # Quantize: round(w/s) then clamp
    w_scaled = w / s
    w_rounded = STE_Round.apply(w_scaled)
    w_clamped = torch.clamp(w_rounded, -n - 1, n)
    
    return w_clamped * s


# =============================================================================
# Gumbel-Softmax for differentiable bit-width selection
# =============================================================================
def gumbel_softmax(logits, tau=1.0, hard=False):
    """
    Gumbel-Softmax sampling for categorical distribution over bit-widths.
    
    Args:
        logits: unnormalized log-probabilities [K] or [L, K]
        tau: temperature (lower = more discrete)
        hard: if True, return one-hot in forward but soft in backward
    Returns:
        soft (or hard) categorical sample
    """
    gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau
    y_soft = F.softmax(gumbels, dim=-1)
    
    if hard:
        # Straight-through: one-hot in forward, soft gradient in backward
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    
    return y_soft


# =============================================================================
# Main Wrapper: PruningQuantizationWrapper
# =============================================================================
class PruningQuantizationWrapper(nn.Module):
    """
    Wraps a base model with per-layer pruning gates and mixed-precision quantization.
    
    For each Conv2d layer in the base model:
      - Adds pruning gate alpha_l of shape [C_out], gate = sigmoid(alpha_l)
      - Adds bit-width logits beta_l of shape [K], pi = softmax(beta_l / tau)
      
    The effective quantized weight for layer l is:
      W_q^l = sum_k pi_k^l * Q_{b_k}(gate^l * W^l)
    
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
        self.temperature = temperature
        self.init_alpha = init_alpha
        
        # Register bit-widths as a buffer (non-learnable constant)
        self.register_buffer('bit_width_tensor', torch.tensor(bit_widths, dtype=torch.float32))
        
        # Discover all Conv2d layers and create pruning/quantization parameters
        self.conv_layer_names = []
        self.conv_layer_info = OrderedDict()  # name -> {out_channels, kernel_params}
        
        self.pruning_alphas = nn.ParameterDict()
        self.bitwidth_betas = nn.ParameterDict()
        
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Conv2d):
                safe_name = name.replace('.', '_')
                self.conv_layer_names.append(safe_name)
                
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
                
                # Pruning gate logits: sigmoid(alpha) ~ 0.95 initially
                alpha = nn.Parameter(torch.full((out_channels,), init_alpha))
                self.pruning_alphas[safe_name] = alpha
                
                # Bit-width logits: uniform initialization (equal probability)
                beta = nn.Parameter(torch.zeros(self.K))
                self.bitwidth_betas[safe_name] = beta
        
        self.num_conv_layers = len(self.conv_layer_names)
        print(f"[JointModel] Wrapped {self.num_conv_layers} Conv2d layers")
        print(f"[JointModel] Bit-width candidates: {bit_widths}")
        
        # Store original weights for quantization during forward
        self._original_weights = {}
        self._hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward pre-hooks to apply pruning + quantization."""
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Conv2d):
                safe_name = name.replace('.', '_')
                hook = module.register_forward_pre_hook(
                    self._make_hook(safe_name, module)
                )
                self._hooks.append(hook)
    
    def _make_hook(self, layer_name, conv_module):
        """Create a forward pre-hook that applies pruning gate + quantization."""
        def hook_fn(module, input):
            # Get pruning gate
            alpha = self.pruning_alphas[layer_name]
            gate = torch.sigmoid(alpha)  # [C_out]
            
            # Get bit-width probabilities
            beta = self.bitwidth_betas[layer_name]
            if self.training:
                pi = gumbel_softmax(beta, tau=self.temperature, hard=False)
            else:
                # At eval, use argmax (hard selection)
                pi = F.one_hot(beta.argmax(), num_classes=self.K).float()
            
            # Save original weight
            orig_weight = module.weight.data.clone()
            
            # Compute quantized weight as weighted sum over bit-widths
            # W_q = sum_k pi_k * Q_{b_k}(gate * W)
            gated_weight = module.weight * gate.view(-1, 1, 1, 1)
            
            quantized_weight = torch.zeros_like(gated_weight)
            for k in range(self.K):
                bw = self.bit_widths[k]
                q_w = uniform_quantize(gated_weight, bw)
                quantized_weight = quantized_weight + pi[k] * q_w
            
            # Replace weight for this forward pass
            module.weight.data.copy_(quantized_weight)
            
            # Store original to restore after forward
            self._original_weights[layer_name] = orig_weight
        
        return hook_fn
    
    def _restore_weights(self):
        """Restore original (latent) weights after forward pass."""
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Conv2d):
                safe_name = name.replace('.', '_')
                if safe_name in self._original_weights:
                    module.weight.data.copy_(self._original_weights[safe_name])
        self._original_weights.clear()
    
    def forward(self, x):
        """Forward pass through the pruned + quantized model."""
        out = self.base_model(x)
        # Restore original latent weights so gradients flow correctly
        self._restore_weights()
        return out
    
    def set_temperature(self, tau):
        """Set Gumbel-Softmax temperature for bit-width selection."""
        self.temperature = max(tau, 0.01)  # Floor to avoid numerical issues
    
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
            
            # Soft channel count
            gate = torch.sigmoid(alpha)
            soft_channels = gate.sum()
            
            # Effective bit-width
            pi = F.softmax(beta / self.temperature, dim=-1)
            b_eff = (pi * self.bit_width_tensor).sum()
            
            # Parameters per filter
            K_l = info['params_per_filter']
            
            total_bits = total_bits + b_eff * soft_channels * K_l
        
        return total_bits
    
    def compute_bops(self, input_spatial=None):
        """
        Compute C_ops(alpha, beta) = sum_l b_W^l * b_A^l * channels * H * W * K_l
        
        Simplified: uses b_eff for both weight and activation bit-width.
        
        Args:
            input_spatial: list of (H, W) per layer, or None for estimate
        Returns:
            total BOPs (differentiable)
        """
        total_bops = torch.tensor(0.0, device=self._get_device())
        
        # Simple estimate: assume spatial dims reduce by 2x at each stride-2 layer
        spatial = 32  # CIFAR-10 input size
        
        for layer_name in self.conv_layer_names:
            alpha = self.pruning_alphas[layer_name]
            beta = self.bitwidth_betas[layer_name]
            info = self.conv_layer_info[layer_name]
            
            gate = torch.sigmoid(alpha)
            soft_channels = gate.sum()
            
            pi = F.softmax(beta / self.temperature, dim=-1)
            b_eff = (pi * self.bit_width_tensor).sum()
            
            K_l = info['params_per_filter']
            
            # BOPs for this layer (b_W * b_A approx b_eff^2)
            layer_bops = b_eff * b_eff * soft_channels * spatial * spatial * K_l
            total_bops = total_bops + layer_bops
        
        return total_bops
    
    def compute_sparsity_reg(self):
        """
        Compute sparsity regularizer R(alpha) = sum_{l,j} sigmoid(alpha_j^l)
        
        Encourages gates to close (smaller = more pruning).
        
        Returns:
            regularization term (differentiable)
        """
        reg = torch.tensor(0.0, device=self._get_device())
        for layer_name in self.conv_layer_names:
            alpha = self.pruning_alphas[layer_name]
            reg = reg + torch.sigmoid(alpha).sum()
        return reg
    
    def get_sparsity_ratio(self):
        """Get the fraction of pruned channels (non-differentiable metric)."""
        total_channels = 0
        pruned_channels = 0
        threshold = 0.01
        
        for layer_name in self.conv_layer_names:
            alpha = self.pruning_alphas[layer_name]
            gate = torch.sigmoid(alpha)
            total_channels += gate.numel()
            pruned_channels += (gate < threshold).sum().item()
        
        return pruned_channels / max(total_channels, 1)
    
    def get_effective_bitwidths(self):
        """Get the selected bit-width per layer (argmax)."""
        result = OrderedDict()
        for layer_name in self.conv_layer_names:
            beta = self.bitwidth_betas[layer_name]
            idx = beta.argmax().item()
            result[layer_name] = self.bit_widths[idx]
        return result
    
    def discretize(self, threshold=0.01):
        """
        Stage 3: Snap pruning gates to {0,1} and bit-widths to argmax.
        
        Args:
            threshold: gates below this are set to 0 (pruned)
        """
        with torch.no_grad():
            for layer_name in self.conv_layer_names:
                # Discretize pruning gates
                alpha = self.pruning_alphas[layer_name]
                gate = torch.sigmoid(alpha)
                binary_gate = (gate >= threshold).float()
                # Set alpha to extreme values to represent 0/1 after sigmoid
                alpha.data = torch.where(
                    binary_gate > 0.5,
                    torch.tensor(10.0, device=alpha.device),   # sigmoid(10) ~ 1
                    torch.tensor(-10.0, device=alpha.device),  # sigmoid(-10) ~ 0
                )
                
                # Discretize bit-widths: set argmax logit high, others low
                beta = self.bitwidth_betas[layer_name]
                idx = beta.argmax()
                new_beta = torch.full_like(beta, -10.0)
                new_beta[idx] = 10.0
                beta.data = new_beta
        
        # Report
        bw = self.get_effective_bitwidths()
        sparsity = self.get_sparsity_ratio()
        print(f"[Discretize] Sparsity: {sparsity:.2%}")
        print(f"[Discretize] Bit-widths: {dict(list(bw.items())[:5])}...")
    
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
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def unfreeze_weights(self):
        """Unfreeze base model weights."""
        for param in self.base_model.parameters():
            param.requires_grad = True
    
    def _get_device(self):
        """Get device of the model parameters."""
        return next(self.base_model.parameters()).device
    
    def get_compression_summary(self):
        """Print a summary of compression achieved."""
        # Full precision baseline size
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
            active = (gate >= 0.01).sum().item()
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
        print(f"  Bit-widths:       {self.get_effective_bitwidths()}")
        print(f"{'='*60}\n")
        
        return baseline_bits, compressed_bits, ratio
