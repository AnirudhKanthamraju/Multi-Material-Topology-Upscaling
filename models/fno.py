"""
Fourier Neural Operator (FNO2d) for Topology Upscaling.

Architecture:
  - Lifting Layer:    Projects (image + 2D grid coords) -> width channels
  - Fourier Layers:   4x [SpectralConv2d + pointwise Conv bypass + GELU]
  - Projection Layer: width -> 128 -> 1 (scalar topology field)

Mesh invariance:
  The last SpectralConv2d can accept an `output_size` argument. When supplied,
  it zero-pads the truncated Fourier spectrum before the Inverse FFT, evaluating
  the learned function on a denser (or differently-sized) grid. The spatial
  bypass in the last layer is upsampled bilinearly to match.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """2-D Fourier integral operator layer."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        # Two sets of weights: low-freq top-left and bottom-left corners
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # (batch, in_ch, H, W), (in_ch, out_ch, H, W) -> (batch, out_ch, H, W)
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor, output_size=None) -> torch.Tensor:
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x)  # (B, C, H, W//2+1)

        # --- multiply the kept modes ---
        out_ft = torch.zeros(
            B, self.out_channels, x_ft.shape[-2], x_ft.shape[-1],
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        # --- optional upscaling via Fourier zero-padding ---
        if output_size is not None:
            H_out, W_out = output_size
            W_freq = W_out // 2 + 1
            padded = torch.zeros(
                B, self.out_channels, H_out, W_freq,
                dtype=torch.cfloat, device=x.device
            )
            # Place kept modes into the larger spectrum
            padded[:, :, :self.modes1, :self.modes2] = out_ft[:, :, :self.modes1, :self.modes2]
            padded[:, :, -self.modes1:, :self.modes2] = out_ft[:, :, -self.modes1:, :self.modes2]
            return torch.fft.irfft2(padded, s=(H_out, W_out))
        else:
            return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNO2d(nn.Module):
    """
    FNO-2D for mesh-invariant topology upscaling.

    Args:
        modes1 (int): Fourier modes kept along height dimension.
        modes2 (int): Fourier modes kept along width  dimension.
        width  (int): Hidden channel width throughout the network.
        in_channels (int): Input image feature channels (e.g. 1 for grayscale, 3 for RGB).
        out_channels (int): Output predicted channels.
    """

    def __init__(self, modes1: int = 12, modes2: int = 12, width: int = 32,
                 in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        # Lifting: (image features + grid_x + grid_y) -> width
        # spatial coords are always 2 channels, so:
        self.p = nn.Linear(in_channels + 2, width)

        # 4 Fourier + bypass layers
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)

        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        # Projection: width -> 128 -> out_channels
        self.mlp0 = nn.Linear(width, 128)
        self.mlp1 = nn.Linear(128, out_channels)

    def get_grid(self, shape, device):
        """Returns a (B, H, W, 2) grid of normalised [0,1] coordinates."""
        B, H, W = shape
        gx = torch.linspace(0, 1, H, device=device).view(1, H, 1, 1).expand(B, H, W, 1)
        gy = torch.linspace(0, 1, W, device=device).view(1, 1, W, 1).expand(B, H, W, 1)
        return torch.cat([gx, gy], dim=-1)           # (B, H, W, 2)

    def forward(self, x: torch.Tensor, output_size=None) -> torch.Tensor:
        """
        Args:
            x           : (B, H, W) greyscale topology tensor, values in [0,1].
            output_size : Optional tuple (H_out, W_out). When provided the last
                          Fourier layer evaluates on this larger grid (mesh invariance).
        Returns:
            Tensor of shape (B, C, H_out, W_out) if output_size else (B, C, H, W).
        """
        B, C, H, W = x.shape
        grid = self.get_grid((B, H, W), x.device)          # (B, H, W, 2)
        
        # Move image channels to last dimension: (B, C, H, W) -> (B, H, W, C)
        x_perm = x.permute(0, 2, 3, 1)
        x_features = torch.cat([x_perm, grid], dim=-1)     # (B, H, W, C+2)
        x_features = self.p(x_features)                    # (B, H, W, width)
        x_features = x_features.permute(0, 3, 1, 2)        # (B, width, H, W)

        # Layers 0-2: standard resolution
        x_features = F.gelu(self.conv0(x_features) + self.w0(x_features))
        x_features = F.gelu(self.conv1(x_features) + self.w1(x_features))
        x_features = F.gelu(self.conv2(x_features) + self.w2(x_features))

        # Layer 3: optional upscaling
        x_fourier = self.conv3(x_features, output_size=output_size)
        if output_size is not None:
            x_bypass = F.interpolate(self.w3(x_features), size=output_size, mode='bilinear', align_corners=False)
        else:
            x_bypass = self.w3(x_features)
        
        x_features = x_fourier + x_bypass                  # (B, width, H_out, W_out)

        x_features = x_features.permute(0, 2, 3, 1)        # (B, H_out, W_out, width)
        x_features = F.gelu(self.mlp0(x_features))         # (B, H_out, W_out, 128)
        x_out = self.mlp1(x_features)                      # (B, H_out, W_out, out_channels)
        return x_out.permute(0, 3, 1, 2)                   # (B, out_channels, H_out, W_out)
