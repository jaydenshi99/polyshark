"""
Autoregressive factored policy head (see docs/policy_head.md).

Reads the TrunkCache (one trunk pass per state) and exposes per-stage logits. Every
stage takes a boolean legal-mask (True = legal) and returns masked logits; caller applies
softmax / log_softmax. The mask itself is produced by the (integration-time) legal-action
helper — not here. Stages:

  type      : core -> N_TYPES logits
  entity    : pointer over unit_tok OR city_tok  (move/attack/recover -> units; train -> cities)
  tile      : pointer over 121 feature_map cells  (move/attack dst, harvest, capture)
  train_unit: categorical over trainable unit types
  research  : categorical over techs
  upgrade   : modal head (UpgradingCity phase), 2 options

Prior conditioning is uniform: prior = concat(type_emb[16], chosen_entity_emb[128]),
with the entity part zeroed for stages that have no entity chosen before them.
"""

import math

import torch
import torch.nn as nn

from model import D_MODEL, TRUNK_DIM, CONV_CH  # 128, 256, 64

# Stage-1 action types (Idle phase). Upgrade is a separate modal (phase-gated).
N_TYPES = 8
T_MOVE, T_ATTACK, T_HARVEST, T_CAPTURE, T_TRAIN, T_RESEARCH, T_RECOVER, T_END = range(N_TYPES)

TYPE_EMB_DIM = 16                       # action-type embedding (≠ model's unit-type embedding)
PRIOR_DIM = TYPE_EMB_DIM + D_MODEL      # type_emb + chosen-entity emb = 144
K_TRAIN_UNITS = 5                       # Warrior..Giant (masked by tech/affordability)
K_TECH = 8                              # Hunting..Strategy
N_UPGRADE_OPTS = 2                      # every level bracket offers exactly 2
BRACKET_DIM = 4                         # level bracket one-hot: ==2, ==3, ==4, >=5

NEG_INF = float("-inf")


def _mlp(in_dim, hidden, out):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out))


class PolicyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.type_emb = nn.Embedding(N_TYPES, TYPE_EMB_DIM)

        self.type_head = _mlp(TRUNK_DIM, 128, N_TYPES)

        # entity pointer: query from (core, prior), keys from entity tokens
        self.entity_q = _mlp(TRUNK_DIM + PRIOR_DIM, D_MODEL, D_MODEL)
        self.entity_k = nn.Linear(D_MODEL, D_MODEL)

        # tile pointer: query projected to conv channels, dotted against feature_map
        self.tile_q = _mlp(TRUNK_DIM + PRIOR_DIM, CONV_CH, CONV_CH)

        # categoricals
        self.train_unit_head = _mlp(TRUNK_DIM + PRIOR_DIM, 128, K_TRAIN_UNITS)
        self.research_head = _mlp(TRUNK_DIM + PRIOR_DIM, 128, K_TECH)

        # modal upgrade head: core + pending-city token + bracket one-hot -> 2
        self.upgrade_head = _mlp(TRUNK_DIM + D_MODEL + BRACKET_DIM, 128, N_UPGRADE_OPTS)

    def _prior(self, type_idx, entity_emb=None):
        """type_idx [B] long -> prior [B, 144]. entity_emb [B,128] or None (zeros)."""
        temb = self.type_emb(type_idx)                       # [B, 16]
        if entity_emb is None:
            entity_emb = temb.new_zeros(temb.shape[0], D_MODEL)
        return torch.cat([temb, entity_emb], dim=-1)         # [B, 144]

    # --- stages: each returns masked logits ---

    def type_logits(self, core, mask):
        return self.type_head(core).masked_fill(~mask, NEG_INF)          # [B, N_TYPES]

    def entity_logits(self, core, type_idx, tokens, mask):
        q = self.entity_q(torch.cat([core, self._prior(type_idx)], dim=-1))   # [B, 128]
        keys = self.entity_k(tokens)                                          # [B, N, 128]
        scores = (keys @ q.unsqueeze(-1)).squeeze(-1) / math.sqrt(D_MODEL)    # [B, N]
        return scores.masked_fill(~mask, NEG_INF)

    def tile_logits(self, core, type_idx, entity_emb, feature_map, mask):
        q = self.tile_q(torch.cat([core, self._prior(type_idx, entity_emb)], dim=-1))  # [B, 64]
        scores = torch.einsum("bc,bchw->bhw", q, feature_map)                # [B, H, W]
        scores = scores.flatten(1) / math.sqrt(CONV_CH)                      # [B, 121]
        return scores.masked_fill(~mask, NEG_INF)

    def train_unit_logits(self, core, type_idx, city_emb, mask):
        prior = self._prior(type_idx, city_emb)
        return self.train_unit_head(torch.cat([core, prior], dim=-1)).masked_fill(~mask, NEG_INF)

    def research_logits(self, core, type_idx, mask):
        prior = self._prior(type_idx)
        return self.research_head(torch.cat([core, prior], dim=-1)).masked_fill(~mask, NEG_INF)

    def upgrade_logits(self, core, city_token, bracket):
        # Both bracket options are always legal -> no mask.
        return self.upgrade_head(torch.cat([core, city_token, bracket], dim=-1))  # [B, 2]
