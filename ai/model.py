import torch
import torch.nn as nn

SPATIAL_CHANNELS = 50
GLOBAL_SIZE = 11
MAP_SIZE = 11
ACTION_SIZE = 7995

_FLAT = 128 * MAP_SIZE * MAP_SIZE  # 15488
_CONCAT = _FLAT + GLOBAL_SIZE       # 15499


class PolysharkNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(SPATIAL_CHANNELS, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.trunk = nn.Sequential(
            nn.Linear(_CONCAT, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

        self.policy_head = nn.Linear(256, ACTION_SIZE)

        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, spatial, global_vec, action_mask=None):
        """
        spatial     : [B, 50, 11, 11]
        global_vec  : [B, 11]
        action_mask : [B, 7995] bool, True = legal  (optional)

        Returns
        -------
        policy : [B, 7995]  normalised probabilities
        value  : [B, 1]     in [-1, 1]
        """
        x = self.conv(spatial)               # [B, 128, 11, 11]
        x = x.flatten(1)                     # [B, 15488]
        x = torch.cat([x, global_vec], dim=1)  # [B, 15499]
        x = self.trunk(x)                    # [B, 256]

        logits = self.policy_head(x)         # [B, 7995]
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))
        policy = torch.softmax(logits, dim=-1)

        value = self.value_head(x)           # [B, 1]
        return policy, value
