import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    """Conv1d → BatchNorm → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                      padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN1D(nn.Module):
    """1-D CNN classifier for noisy/clean peak signals.

    Input : (batch, 4, max_len)  — 4 channels = peakA/C/G/T
    Output: (batch, num_classes) — logits (use CrossEntropyLoss)
    """

    def __init__(self, in_channels: int = 4, num_classes: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            # stage 1 — wide kernel to capture broad baseline patterns
            _ConvBlock(in_channels, 32, kernel=7),
            nn.MaxPool1d(2),                    # L → L/2

            # stage 2
            _ConvBlock(32, 64, kernel=5),
            nn.MaxPool1d(2),                    # → L/4

            # stage 3
            _ConvBlock(64, 128, kernel=3),
            nn.MaxPool1d(2),                    # → L/8

            # stage 4
            _ConvBlock(128, 256, kernel=3),
            nn.MaxPool1d(2),                    # → L/16

            # stage 5 — fine-grained features
            _ConvBlock(256, 512, kernel=3),
            nn.AdaptiveAvgPool1d(1),            # → 1 (length-agnostic)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


if __name__ == "__main__":
    model = CNN1D()
    dummy = torch.randn(8, 4, 2000)
    out = model(dummy)
    print("Output shape:", out.shape)   # (8, 2)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")
