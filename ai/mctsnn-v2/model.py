"""
Entity-set encoder: projects unit/city tokens into a shared d=128 space and mixes
them with 2 pre-norm transformer blocks. See embedding.md and attention.md.

Policy/value heads are downstream and not built yet — this module ends at the
context-mixed entity tokens.
"""

import torch
import torch.nn as nn

from features import UNIT_FEAT_DIM, CITY_FEAT_DIM, NUM_UNIT_TYPES

D_MODEL = 128
TYPE_EMB_DIM = 8
N_HEADS = 4          # 128 / 4 = 32-dim heads
FF_DIM = 512         # 4 * d_model
N_BLOCKS = 2

UNIT_TOKEN_DIM = TYPE_EMB_DIM + UNIT_FEAT_DIM  # 8 + 9 = 17
CITY_TOKEN_DIM = CITY_FEAT_DIM                 # 14


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
    def __init__(self):
        super().__init__()
        self.type_emb = nn.Embedding(NUM_UNIT_TYPES, TYPE_EMB_DIM)
        self.unit_proj = EntityProjection(UNIT_TOKEN_DIM)
        self.city_proj = EntityProjection(CITY_TOKEN_DIM)

        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=FF_DIM,
            activation="gelu",
            norm_first=True,     # pre-norm: x = x + sublayer(LayerNorm(x))
            batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=N_BLOCKS, enable_nested_tensor=False,
        )

    def forward(self, unit_types, unit_feats, unit_mask, city_feats, city_mask):
        """
        unit_types : int64   [B, Lu]
        unit_feats : float32 [B, Lu, 9]
        unit_mask  : bool    [B, Lu]     True = real entity
        city_feats : float32 [B, Lc, 14]
        city_mask  : bool    [B, Lc]     True = real entity

        Returns
        -------
        tokens   : [B, Lu+Lc, 128]   context-mixed entity tokens
        real_mask: bool [B, Lu+Lc]   True = real (for downstream pooling/masking)
        """
        # Unit token: type embedding concatenated with numeric features -> project.
        u_emb = self.type_emb(unit_types)                    # [B, Lu, 8]
        u_tok = torch.cat([u_emb, unit_feats], dim=-1)       # [B, Lu, 17]
        u_tok = self.unit_proj(u_tok)                        # [B, Lu, 128]

        c_tok = self.city_proj(city_feats)                   # [B, Lc, 128]

        tokens = torch.cat([u_tok, c_tok], dim=1)            # [B, L, 128]
        real_mask = torch.cat([unit_mask, city_mask], dim=1)  # [B, L], True = real

        # Transformer wants True = ignore, so invert. Every state has >=1 real
        # entity, so no query row is fully masked (which would NaN the softmax).
        pad_mask = ~real_mask
        tokens = self.blocks(tokens, src_key_padding_mask=pad_mask)
        return tokens, real_mask
