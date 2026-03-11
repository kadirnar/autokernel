"""
AutoKernel -- ConvTranspose1d kernel.

Current kernel: Direct 1D transposed convolution (stride-aware)
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Handles strided transposed convolutions as used in DAC-VAE decoder (upsampling).
Each program instance handles one (batch, output_channel) pair.
"""

KERNEL_TYPE = "conv_transpose1d"

import torch
import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    # Dimensions
    C_in,
    L_in,
    C_out,
    K,
    stride_conv,
    padding,
    L_out,
    # Strides (in elements)
    stride_x_b,
    stride_x_c,
    stride_w_ci,
    stride_w_co,
    stride_out_b,
    stride_out_c,
    # Flags
    HAS_BIAS: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """Direct 1D transposed convolution.

    For each output position l_out:
      y[b, co, l_out] = sum_{ci, k} x[b, ci, i] * w[ci, co, k]
      where i = (l_out + padding - k) / stride, and i must be integer in [0, L_in).
    """
    pid = tl.program_id(0)
    b_idx = pid // C_out
    co_idx = pid % C_out

    for l_start in range(0, L_out, BLOCK_L):
        l_offs = l_start + tl.arange(0, BLOCK_L)
        l_mask = l_offs < L_out

        acc = tl.zeros((BLOCK_L,), dtype=tl.float32)

        for ci in range(C_in):
            for k in range(K):
                # Compute input index: i = (l_out + padding - k) / stride
                numerator = l_offs + padding - k
                # Check divisibility by stride
                is_div = (numerator % stride_conv) == 0
                i_pos = numerator // stride_conv
                valid = l_mask & is_div & (i_pos >= 0) & (i_pos < L_in)

                # Load input
                x_off = b_idx * stride_x_b + ci * stride_x_c + i_pos
                x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)

                # Load weight: w[ci, co, k]
                w_off = ci * stride_w_ci + co_idx * stride_w_co + k
                w_val = tl.load(w_ptr + w_off)

                acc += x_val * w_val

        if HAS_BIAS:
            bias_val = tl.load(bias_ptr + co_idx)
            acc += bias_val

        out_off = b_idx * stride_out_b + co_idx * stride_out_c + l_offs
        tl.store(out_ptr + out_off, acc, mask=l_mask)


def kernel_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.conv_transpose1d_ref signature."""
    assert x.is_cuda and weight.is_cuda
    assert x.ndim == 3  # [B, C_in, L]
    assert weight.ndim == 3  # [C_in, C_out, K]

    B, C_in, L_in = x.shape
    C_in_w, C_out, K = weight.shape
    assert C_in == C_in_w

    # Compute output length: L_out = (L_in - 1) * stride - 2*padding + K + output_padding
    L_out = (L_in - 1) * stride - 2 * padding + K + output_padding
    output = torch.empty(B, C_out, L_out, device=x.device, dtype=x.dtype)

    BLOCK_L = triton.next_power_of_2(min(L_out, 1024))

    grid = (B * C_out,)
    conv_transpose1d_kernel[grid](
        x, weight,
        bias if bias is not None else x,  # dummy pointer when no bias
        output,
        C_in, L_in, C_out, K, stride, padding, L_out,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        HAS_BIAS=(bias is not None),
        BLOCK_L=BLOCK_L,
    )

    return output
