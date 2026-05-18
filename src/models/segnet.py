"""
src/models/segnet.py
====================
1D U-Net for per-scan-point noise segmentation of Sanger traces.

Architecture: 4-level encoder-decoder with skip connections.
  Input  : (B, N_CHANNELS, TARGET_LEN)
  Output : (B, 1, TARGET_LEN)  — per-point noisy probability in [0, 1]

N_CHANNELS = 20 (4 peaks + 8 derivatives + total + max_ch + pr +
                  bc_mask + bc_qual + roll_snr + roll_base + pos_idx)
TARGET_LEN = 4096 (traces resampled to this fixed length)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CHANNELS = 20
TARGET_LEN = 4096


class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3):
        super().__init__(
            nn.Conv1d(in_ch, out_ch, k, padding=k // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )


class _EncBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            _ConvBnRelu(in_ch, out_ch),
            _ConvBnRelu(out_ch, out_ch),
        )
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        return self.pool(skip), skip


class _DecBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose1d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            _ConvBnRelu(in_ch // 2 + skip_ch, out_ch),
            _ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
        return self.conv(torch.cat([x, skip], dim=1))


class UNet1D(nn.Module):
    """
    4-level 1D U-Net for trace noise segmentation.

    Parameters
    ----------
    n_channels : number of input feature channels (default 20)
    base_ch    : base filter count; doubles each encoder level (default 32)

    Model size at base_ch=32: ~1.5 M parameters
    Model size at base_ch=64: ~6 M parameters
    """

    def __init__(self, n_channels: int = N_CHANNELS, base_ch: int = 32):
        super().__init__()
        b = base_ch
        self.enc1 = _EncBlock(n_channels, b)
        self.enc2 = _EncBlock(b,      b * 2)
        self.enc3 = _EncBlock(b * 2,  b * 4)
        self.enc4 = _EncBlock(b * 4,  b * 8)

        self.bottleneck = nn.Sequential(
            _ConvBnRelu(b * 8,  b * 16),
            _ConvBnRelu(b * 16, b * 16),
        )

        self.dec4 = _DecBlock(b * 16, b * 8, b * 8)
        self.dec3 = _DecBlock(b * 8,  b * 4, b * 4)
        self.dec2 = _DecBlock(b * 4,  b * 2, b * 2)
        self.dec1 = _DecBlock(b * 2,  b,     b)

        self.head = nn.Conv1d(b, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)

        x = self.bottleneck(x4)

        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return torch.sigmoid(self.head(x))   # (B, 1, L)


if __name__ == "__main__":
    model = UNet1D()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet1D  |  params: {n_params:,}")
    x = torch.randn(2, N_CHANNELS, TARGET_LEN)
    y = model(x)
    print(f"Input:  {tuple(x.shape)}  →  Output: {tuple(y.shape)}")
