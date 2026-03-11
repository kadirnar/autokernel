"""
Reference implementations -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. These are the oracles that the benchmark harness checks against.
"""

import torch
import torch.nn.functional as F

# Matrix Multiplication
def matmul_ref(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Standard matrix multiplication. A @ B."""
    return torch.matmul(A, B)

# Conv1d
def conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """Standard 1D convolution. Used in DAC-VAE encoder and residual units."""
    return F.conv1d(x, weight, bias, stride=stride, padding=padding)


# ConvTranspose1d
def conv_transpose1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
) -> torch.Tensor:
    """Standard 1D transposed convolution. Used in DAC-VAE decoder for upsampling."""
    return F.conv_transpose1d(x, weight, bias, stride=stride, padding=padding,
                              output_padding=output_padding)


# Snake Activation
def snake_activation_ref(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Snake activation: x + (1/alpha) * sin(alpha * x)^2. Used throughout DAC-VAE."""
    return x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
