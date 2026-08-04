import torch
import torch.nn as nn

from spec import AISpec, DEFAULT_SPEC


class PolysharkNet(nn.Module):
    """
    Conv trunk over the spatial map + global features → policy / value heads.

    All shape constants come from `AISpec`. The same architecture works for
    any (map_size, unit count, tech count); only the channel counts and
    flatten widths change.
    """

    def __init__(self, spec: AISpec = DEFAULT_SPEC):
        super().__init__()
        self.spec = spec

        sc = spec.spatial_channels
        sz = spec.map_size
        gs = spec.global_size
        a  = spec.action_size

        self.conv = nn.Sequential(
            nn.Conv2d(sc, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        flat   = 128 * sz * sz
        concat = flat + gs

        self.trunk = nn.Sequential(
            nn.Linear(concat, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

        self.policy_head = nn.Linear(256, a)

        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, spatial, global_vec, action_mask=None):
        x = self.conv(spatial)
        x = x.flatten(1)
        x = torch.cat([x, global_vec], dim=1)
        x = self.trunk(x)

        logits = self.policy_head(x)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))
        policy = torch.softmax(logits, dim=-1)

        value = self.value_head(x)
        return policy, value
