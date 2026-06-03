"""
TDL value network.

  forward(spatial, global_vec) -> scalar ∈ [-1, 1]

Architecture
────────────
  spatial [B, C_IN, 11, 11]
    Conv(C_IN→32, 3×3, pad=1)  →  BN → ReLU    [B,  32, 11, 11]
    Conv(32→64,   3×3, pad=0)  →  BN → ReLU    [B,  64,  9,  9]
    Conv(64→64,   3×3, pad=0)  →  BN → ReLU    [B,  64,  7,  7]
    Conv(64→32,   3×3, pad=0)  →  BN → ReLU    [B,  32,  5,  5]
    flatten                                       [B, 800]
  cat with global_vec [B, G]                     [B, 800+G]
    Linear(800+G → 64) → ReLU
    Linear(64    →  1) → Tanh
"""

import torch
import torch.nn as nn

from encoder import C_IN, G


class ValueNet(nn.Module):
    def __init__(self, c_in: int = C_IN, g: int = G):
        super().__init__()

        self.spatial = nn.Sequential(
            # Stem: preserve 11×11
            nn.Conv2d(c_in, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            # Unpadded reduction: 11→9→7→5
            nn.Conv2d(32, 64, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Flatten(),  # → [B, 32*5*5] = [B, 800]
        )

        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(800 + g, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, spatial: torch.Tensor, global_vec: torch.Tensor) -> torch.Tensor:
        """
        spatial    : [B, C_IN, 11, 11]  float32
        global_vec : [B, G]             float32
        returns    : [B, 1]             float32 in [-1, 1]
        """
        feat = self.spatial(spatial)                  # [B, 800]
        x    = torch.cat([feat, global_vec], dim=1)  # [B, 811]
        return self.head(x)                    # [B, 1]
