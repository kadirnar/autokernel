"""
Minimal DAC-VAE (Descript Audio Codec - Variational Autoencoder) implementation
for AutoKernel profiling.

This is a self-contained model that does NOT require the dacvae, audiotools, or
huggingface_hub libraries. It implements the core encoder-decoder-VAE architecture
so AutoKernel can profile and optimize the bottleneck kernels.

Based on: https://github.com/facebookresearch/dacvae

Key operations for GPU optimization:
  - Conv1d (strided downsampling in encoder, processing in residual units)
  - ConvTranspose1d (strided upsampling in decoder)
  - Snake activation (learnable periodic activation, used throughout)

Usage:
    uv run profile.py --model models/dacvae.py --class-name DACVAE --input-shape 1,1,44100 --dtype float32
    uv run profile.py --model models/dacvae.py --class-name DACVAESmall --input-shape 1,1,22050 --dtype float32
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# ---------------------------------------------------------------------------
# Snake activation (from dacvae/nn/layers.py)
# ---------------------------------------------------------------------------

@torch.jit.script
def snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Snake activation: x + (1/alpha) * sin(alpha * x)^2"""
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    """Learnable Snake activation with per-channel alpha parameter."""
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return snake(x, self.alpha)


# ---------------------------------------------------------------------------
# Normalized convolutions (from dacvae/nn/layers.py)
# ---------------------------------------------------------------------------

class WNConv1d(nn.Module):
    """Weight-normalized Conv1d with symmetric padding."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        pad = (kernel_size - stride) * dilation // 2
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=pad, dilation=dilation,
                groups=groups, bias=bias,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class WNConvTranspose1d(nn.Module):
    """Weight-normalized ConvTranspose1d with symmetric padding."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        padding = (stride + 1) // 2
        output_padding = 1 if stride % 2 else 0
        self.conv = weight_norm(
            nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding,
                output_padding=output_padding, bias=bias,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualUnit(nn.Module):
    """Residual unit with two convolutions and Snake activation."""
    def __init__(self, dim: int, kernel: int = 7, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel, dilation=dilation),
            Snake1d(dim),
            WNConv1d(dim, dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class EncoderBlock(nn.Module):
    """Encoder block: 3 residual units + strided downsample."""
    def __init__(self, dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, kernel=7, dilation=1),
            ResidualUnit(dim // 2, kernel=7, dilation=3),
            ResidualUnit(dim // 2, kernel=7, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """Decoder block: strided upsample + 3 residual units."""
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(input_dim, output_dim, kernel_size=2 * stride, stride=stride),
            ResidualUnit(output_dim, kernel=7, dilation=1),
            ResidualUnit(output_dim, kernel=7, dilation=3),
            ResidualUnit(output_dim, kernel=7, dilation=9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Encoder / Decoder / VAE Bottleneck
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Audio encoder: Conv1d -> EncoderBlocks (downsample) -> Conv1d to latent."""
    def __init__(self, d_model: int = 64, strides: list = [2, 4, 8, 8], d_latent: int = 64):
        super().__init__()
        layers = [WNConv1d(1, d_model, kernel_size=7)]
        for s in strides:
            d_model *= 2
            layers.append(EncoderBlock(d_model, stride=s))
        layers.append(Snake1d(d_model))
        layers.append(WNConv1d(d_model, d_latent, kernel_size=3))
        self.block = nn.Sequential(*layers)
        self.enc_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VAEBottleneck(nn.Module):
    """VAE bottleneck: project to mean/scale, sample, project back."""
    def __init__(self, input_dim: int, codebook_dim: int = 64):
        super().__init__()
        self.in_proj = WNConv1d(input_dim, codebook_dim * 2, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mean, scale = self.in_proj(z).chunk(2, dim=1)
        stdev = F.softplus(scale) + 1e-4
        z_q = torch.randn_like(mean) * stdev + mean
        z_q = self.out_proj(z_q)
        return z_q


class Decoder(nn.Module):
    """Audio decoder: Conv1d -> DecoderBlocks (upsample) -> Conv1d to waveform."""
    def __init__(self, input_channel: int, channels: int, rates: list):
        super().__init__()
        layers = [WNConv1d(input_channel, channels, kernel_size=7)]
        for i, stride in enumerate(rates):
            in_dim = channels // 2 ** i
            out_dim = channels // 2 ** (i + 1)
            layers.append(DecoderBlock(in_dim, out_dim, stride))
        layers.append(Snake1d(out_dim))
        layers.append(WNConv1d(out_dim, 1, kernel_size=7))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Full DAC-VAE Model
# ---------------------------------------------------------------------------

class DACVAE(nn.Module):
    """
    DAC-VAE: Variational Autoencoder for audio codec.

    Default config matches the original DAC-VAE architecture:
      - Encoder: 64 -> 128 -> 256 -> 512 -> 1024 channels (downsample 512x)
      - Latent: 64 dimensions
      - Decoder: 1536 -> 768 -> 384 -> 192 -> 96 channels (upsample 512x)

    Total parameters: ~35M

    Input: [B, 1, T] audio waveform (mono, any sample rate)
    Output: [B, 1, T] reconstructed audio waveform

    Usage:
        uv run profile.py --model models/dacvae.py --class-name DACVAE \\
            --input-shape 1,1,44100 --dtype float32
    """

    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: list = [2, 4, 8, 8],
        latent_dim: int = 64,
        decoder_dim: int = 1536,
        decoder_rates: list = [8, 8, 4, 2],
        sample_rate: int = 44100,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = 1
        for r in encoder_rates:
            self.hop_length *= r  # 2*4*8*8 = 512

        # Encoder output dimension
        enc_out_dim = encoder_dim * (2 ** len(encoder_rates))  # 64 * 16 = 1024

        self.encoder = Encoder(encoder_dim, encoder_rates, enc_out_dim)
        self.bottleneck = VAEBottleneck(enc_out_dim, latent_dim)
        self.decoder = Decoder(enc_out_dim, decoder_dim, decoder_rates)

        # Parameter count
        n_params = sum(p.numel() for p in self.parameters())
        print(f"DACVAE: {n_params / 1e6:.1f}M parameters")

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        """Pad input to be divisible by hop_length."""
        length = x.size(-1)
        if length % self.hop_length:
            pad_amount = self.hop_length - (length % self.hop_length)
            x = F.pad(x, (0, pad_amount), mode="reflect")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode -> bottleneck -> decode.

        Parameters
        ----------
        x : Tensor[B, 1, T]
            Input audio waveform (mono).

        Returns
        -------
        Tensor[B, 1, T]
            Reconstructed audio waveform.
        """
        length = x.size(-1)
        x = self._pad(x)
        z = self.encoder(x)
        z = self.bottleneck(z)
        x_hat = self.decoder(z)
        return x_hat[..., :length]


class DACVAESmall(nn.Module):
    """
    Smaller DAC-VAE variant for faster profiling on limited GPU memory.

    Reduced architecture:
      - Encoder: 32 -> 64 -> 128 -> 256 channels (downsample 32x)
      - Latent: 32 dimensions
      - Decoder: 384 -> 192 -> 96 -> 48 channels (upsample 32x)

    Total parameters: ~4M

    Usage:
        uv run profile.py --model models/dacvae.py --class-name DACVAESmall \\
            --input-shape 1,1,22050 --dtype float32
    """

    def __init__(self):
        super().__init__()
        encoder_dim = 32
        encoder_rates = [2, 4, 4]
        latent_dim = 32
        decoder_dim = 384
        decoder_rates = [4, 4, 2]
        self.hop_length = 1
        for r in encoder_rates:
            self.hop_length *= r  # 32

        enc_out_dim = encoder_dim * (2 ** len(encoder_rates))  # 32 * 8 = 256

        self.encoder = Encoder(encoder_dim, encoder_rates, enc_out_dim)
        self.bottleneck = VAEBottleneck(enc_out_dim, latent_dim)
        self.decoder = Decoder(enc_out_dim, decoder_dim, decoder_rates)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"DACVAESmall: {n_params / 1e6:.1f}M parameters")

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(-1)
        if length % self.hop_length:
            pad_amount = self.hop_length - (length % self.hop_length)
            x = F.pad(x, (0, pad_amount), mode="reflect")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(-1)
        x = self._pad(x)
        z = self.encoder(x)
        z = self.bottleneck(z)
        x_hat = self.decoder(z)
        return x_hat[..., :length]
