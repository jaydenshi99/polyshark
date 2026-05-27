"""
TDL value network.

  forward(spatial, global_vec) -> scalar ∈ [-1, 1]

Architecture
────────────
  spatial [B, C_IN, 11, 11]
    Stem Conv(C_IN→64, 3×3, pad=1)  →  BN → ReLU    [B,  64, 11, 11]
    ResBlock(64)                                       [B,  64, 11, 11]
    ResBlock(64)                                       [B,  64, 11, 11]
    Conv(64→128,  3×3, pad=0)        →  BN → ReLU    [B, 128,  9,  9]
    Conv(128→128, 3×3, pad=0)        →  BN → ReLU    [B, 128,  7,  7]
    Conv(128→128, 3×3, pad=0)        →  BN → ReLU    [B, 128,  5,  5]
    Conv(128→64,  1×1)               →  BN → ReLU    [B,  64,  5,  5]
    flatten                                            [B, 1600]
  cat with global_vec [B, G]                          [B, 1600+G]
    Linear(1600+G → 256) → ReLU
    Linear(256    →   1) → Tanh
"""

import torch
import torch.nn as nn

from encoder import C_IN, G


class _ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.net(x) + x)


class ValueNet(nn.Module):
    def __init__(self, c_in: int = C_IN, g: int = G):
        super().__init__()

        self.spatial = nn.Sequential(
            # Stem: preserve 11×11
            nn.Conv2d(c_in, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Two residual blocks (full-board receptive field)
            _ResBlock(64),
            _ResBlock(64),
            # Gradual unpadded reduction: 11→9→7→5
            nn.Conv2d(64,  128, 3, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            # 1×1 channel compress before flatten
            nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Flatten(),  # → [B, 64*5*5] = [B, 1600]
        )

        self.head = nn.Sequential(
            nn.Linear(1600 + g, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, spatial: torch.Tensor, global_vec: torch.Tensor) -> torch.Tensor:
        """
        spatial    : [B, C_IN, 11, 11]  float32
        global_vec : [B, G]             float32
        returns    : [B, 1]             float32 in [-1, 1]
        """
        feat = self.spatial(spatial)           # [B, 1600]
        x    = torch.cat([feat, global_vec], dim=1)  # [B, 1611]
        return self.head(x)                    # [B, 1]
