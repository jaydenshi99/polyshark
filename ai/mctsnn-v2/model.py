"""
Full state -> value network (policy head TBD).

Pipeline (see docs/embedding.md, docs/attention.md, docs/board.md):

  entities --embed+project--> tokens --2 transformer blocks--> context-mixed tokens
       |                                                              |
       |                                              project 128->32, scatter to grid
       |                                                              v
  board [18,H,W] ----------------------------------- concat -> [82,H,W] -> conv tower
                                                                              |
  globals [12] --MLP--> [32] ------------------------ concat -> trunk -> value (tanh)
"""

import torch
import torch.nn as nn

from features import UNIT_FEAT_DIM, CITY_FEAT_DIM, NUM_UNIT_TYPES, BOARD_CHANNELS, GLOBAL_DIM

D_MODEL = 128
TYPE_EMB_DIM = 8
N_HEADS = 4          # 128 / 4 = 32-dim heads
FF_DIM = 512         # 4 * d_model
N_BLOCKS = 2

UNIT_TOKEN_DIM = TYPE_EMB_DIM + UNIT_FEAT_DIM  # 8 + 9 = 17
CITY_TOKEN_DIM = CITY_FEAT_DIM                 # 14

SCATTER_DIM = 16                               # entity token -> grid plane width
CONV_IN = BOARD_CHANNELS + 2 * SCATTER_DIM     # 18 + 16 + 16 = 50
CONV_CH = 64
N_RES_BLOCKS = 3
GLOBAL_EMB = 32
TRUNK_DIM = 256


class EntityProjection(nn.Module):
    """Linear(in -> d) -> GELU -> Linear(d -> d) -> LayerNorm."""

    def __init__(self, in_dim: int, d: int = D_MODEL):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

    def forward(self, x):
        return self.net(x)


class EntityEncoder(nn.Module):
    """Entities -> context-mixed tokens via type embedding, projection, 2 blocks."""

    def __init__(self):
        super().__init__()
        self.type_emb = nn.Embedding(NUM_UNIT_TYPES, TYPE_EMB_DIM)
        self.unit_proj = EntityProjection(UNIT_TOKEN_DIM)
        self.city_proj = EntityProjection(CITY_TOKEN_DIM)

        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=FF_DIM,
            activation="gelu", norm_first=True, batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=N_BLOCKS, enable_nested_tensor=False,
        )

    def forward(self, unit_types, unit_feats, unit_mask, city_feats, city_mask):
        """Returns unit_tokens [B,Lu,128], city_tokens [B,Lc,128]."""
        u_emb = self.type_emb(unit_types)
        u_tok = self.unit_proj(torch.cat([u_emb, unit_feats], dim=-1))  # [B,Lu,128]
        c_tok = self.city_proj(city_feats)                             # [B,Lc,128]

        Lu = u_tok.shape[1]
        tokens = torch.cat([u_tok, c_tok], dim=1)                      # [B,L,128]
        pad_mask = ~torch.cat([unit_mask, city_mask], dim=1)           # True = ignore
        tokens = self.blocks(tokens, src_key_padding_mask=pad_mask)

        return tokens[:, :Lu], tokens[:, Lu:]


def scatter_to_grid(src, tiles, map_tiles):
    """
    Scatter per-entity vectors onto a flat grid, then reshape to [B, C, H, W].
      src   : [B, L, C]  (padded rows must be pre-zeroed)
      tiles : [B, L]     int64 tile index per entity (padded rows -> 0)
    Padded rows contribute 0 at tile 0 (scatter-add), so they don't corrupt a real
    entity there. Each plane holds <=1 real entity per tile, so no real collisions.
    """
    B, _, C = src.shape
    plane = torch.zeros(B, map_tiles, C, device=src.device, dtype=src.dtype)
    idx = tiles.unsqueeze(-1).expand(-1, -1, C)          # [B,L,C]
    plane.scatter_add_(1, idx, src)                      # [B, map_tiles, C]
    hw = int(map_tiles ** 0.5)
    return plane.view(B, hw, hw, C).permute(0, 3, 1, 2).contiguous()  # [B,C,H,W]


class ResBlock(nn.Module):
    """3x3 -> BN -> ReLU -> 3x3 -> BN, + identity, -> ReLU. Zero-padded, size-preserving."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + x)


class PolysharkNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.entity = EntityEncoder()
        self.unit_scatter = nn.Linear(D_MODEL, SCATTER_DIM)
        self.city_scatter = nn.Linear(D_MODEL, SCATTER_DIM)

        # Stem lifts the 50-channel input to the 64-wide residual body; 3 ResBlocks
        # keep 64 channels and 11x11 spatial (3x3, pad=1 -> zero-padded, no shrink).
        self.stem = nn.Sequential(
            nn.Conv2d(CONV_IN, CONV_CH, 3, padding=1, bias=False),
            nn.BatchNorm2d(CONV_CH), nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(CONV_CH) for _ in range(N_RES_BLOCKS)])

        self.global_mlp = nn.Sequential(nn.Linear(GLOBAL_DIM, GLOBAL_EMB), nn.GELU())

        # Shared trunk: fuse pooled board + globals -> 256-d shared representation.
        # Value (and later policy-value) heads branch off this.
        trunk_in = CONV_CH + GLOBAL_EMB  # 64 + 32 = 96
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, TRUNK_DIM), nn.LayerNorm(TRUNK_DIM), nn.GELU(),
            nn.Linear(TRUNK_DIM, TRUNK_DIM), nn.LayerNorm(TRUNK_DIM), nn.GELU(),
        )
        self.value_head = nn.Sequential(nn.Linear(TRUNK_DIM, 1), nn.Tanh())

    def forward(self, unit_types, unit_feats, unit_mask, unit_tiles,
                city_feats, city_mask, city_tiles, board, globals_):
        unit_tok, city_tok = self.entity(
            unit_types, unit_feats, unit_mask, city_feats, city_mask,
        )

        map_tiles = board.shape[-1] * board.shape[-2]
        # Zero padded rows before scatter so they contribute nothing.
        u32 = self.unit_scatter(unit_tok) * unit_mask.unsqueeze(-1)
        c32 = self.city_scatter(city_tok) * city_mask.unsqueeze(-1)
        unit_plane = scatter_to_grid(u32, unit_tiles, map_tiles)   # [B,32,H,W]
        city_plane = scatter_to_grid(c32, city_tiles, map_tiles)   # [B,32,H,W]

        x = torch.cat([board, unit_plane, city_plane], dim=1)      # [B,50,H,W]
        x = self.blocks(self.stem(x))                              # [B,64,H,W]
        x = x.mean(dim=(2, 3))                                     # global avg pool [B,64]

        g = self.global_mlp(globals_)                             # [B,32]
        rep = self.trunk(torch.cat([x, g], dim=1))                # [B,256] shared rep
        value = self.value_head(rep)                              # [B,1]
        return value
