"""
AutoKernel -- Snake activation kernel.

Current kernel: Element-wise Snake activation (channel-parallel)
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Snake(x, alpha) = x + (1 / (alpha + eps)) * sin(alpha * x)^2

This is a learnable periodic activation used extensively in DAC-VAE.
Each program instance handles one (batch, channel) pair across the length.
"""

KERNEL_TYPE = "snake_activation"

import torch
import triton
import triton.language as tl


@triton.jit
def snake_kernel(
    x_ptr,
    alpha_ptr,
    out_ptr,
    C,
    L,
    stride_x_b,
    stride_x_c,
    stride_out_b,
    stride_out_c,
    EPS: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """Element-wise Snake activation. One program per (batch, channel) pair."""
    pid = tl.program_id(0)
    b_idx = pid // C
    c_idx = pid % C

    # Load alpha for this channel (shape [1, C, 1] -> index by c_idx)
    alpha = tl.load(alpha_ptr + c_idx)

    # Precompute reciprocal
    inv_alpha = 1.0 / (alpha + EPS)

    # Process length in blocks
    for l_start in range(0, L, BLOCK_L):
        l_offs = l_start + tl.arange(0, BLOCK_L)
        l_mask = l_offs < L

        # Load input
        x_off = b_idx * stride_x_b + c_idx * stride_x_c + l_offs
        x_val = tl.load(x_ptr + x_off, mask=l_mask, other=0.0)

        # Snake: x + (1/alpha) * sin(alpha * x)^2
        # Cast to fp32 for tl.sin (requires fp32/fp64)
        x_f32 = x_val.to(tl.float32)
        sin_val = tl.sin(alpha * x_f32)
        result = x_f32 + inv_alpha * sin_val * sin_val
        result = result.to(x_val.dtype)

        # Store output
        out_off = b_idx * stride_out_b + c_idx * stride_out_c + l_offs
        tl.store(out_ptr + out_off, result, mask=l_mask)


def kernel_fn(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.snake_activation_ref signature."""
    assert x.is_cuda and alpha.is_cuda
    assert x.ndim == 3  # [B, C, L]
    assert alpha.ndim == 3  # [1, C, 1]

    B, C, L = x.shape
    output = torch.empty_like(x)

    BLOCK_L = triton.next_power_of_2(min(L, 1024))

    grid = (B * C,)
    snake_kernel[grid](
        x,
        alpha,  # alpha is [1, C, 1], we index as flat[c_idx]
        output,
        C, L,
        x.stride(0), x.stride(1),
        output.stride(0), output.stride(1),
        EPS=1e-9,
        BLOCK_L=BLOCK_L,
    )

    return output
