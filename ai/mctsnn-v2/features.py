"""
GameState -> per-entity feature arrays (numpy). Pure, no torch.

Produces one token's worth of raw features per visible entity, from the current
player's perspective (me/opp, fog-gated). See embedding.md for the field layout
and normalization decisions.

  Unit token = type_id (embedded in model, 8d) + unit_feats (9)   -> 17
  City token = city_feats (14)                                    -> 14

Entities are enumerated by scanning tiles (the engine exposes no unit/city list),
gated on is_visible(me) to respect fog. Only alive, placed units appear on tiles.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../build/bindings"))
import polyshark  # noqa: E402

UNIT_FEAT_DIM = 9
CITY_FEAT_DIM = 14
NUM_UNIT_TYPES = 6  # None, Warrior, Archer, Rider, Defender, Giant (enum incl. index 0)

# Base HP per UnitType index — mirror of UNIT_DEFS in engine/include/unit_def.h.
# Used only to derive is_veteran (max_hp > base_hp). No binding exposes this yet;
# a `unit_base_hp` binding (like unit_base_move) would remove the duplication.
_BASE_HP = [0, 10, 10, 10, 15, 40]

# Base movement per UnitType index, cached from the engine at import.
_BASE_MOVE = [polyshark.unit_base_move(i) for i in range(NUM_UNIT_TYPES)]


def encode_entities(state):
    """
    Returns a dict with, from the current player's perspective:
        unit_types : int64   [Nu]         UnitType enum index, for embedding lookup
        unit_feats : float32 [Nu, 9]
        city_feats : float32 [Nc, 14]
    Nu / Nc are the counts of visible units / cities (>= 0).
    """
    me = state.current_player()
    sz = state.map_size()
    denom = max(sz - 1, 1)  # normalize row/col to [0, 1]

    unit_types, unit_feats, city_feats = [], [], []

    for i in range(sz * sz):
        if not state.is_visible(me, i):
            continue
        tile = state.tile_at(i)
        row, col = divmod(i, sz)

        if tile.has_unit:
            u = state.get_unit(tile.unit_id)
            if u.is_alive:
                t = int(u.type)
                base_move = _BASE_MOVE[t] if t < NUM_UNIT_TYPES else 1
                base_hp = _BASE_HP[t] if t < NUM_UNIT_TYPES else 10
                unit_types.append(t)
                unit_feats.append([
                    1.0 if u.owner == me else 0.0,       # owner: me/opp
                    row / denom,                         # row
                    col / denom,                         # col
                    u.hp / max(u.max_hp, 1),             # hp / max_hp
                    1.0 if u.max_hp > base_hp else 0.0,  # is_veteran
                    u.move_points / max(base_move, 1),   # move / base_move
                    min(u.kills / 3.0, 1.0),             # kills (clamped)
                    float(u.has_attacked),               # has_attacked
                    float(u.promotion_ready),            # promotion_ready
                ])

        if tile.has_city:
            c = state.get_city(tile.city_id)
            lvl = c.level
            unit_cap = c.unit_capacity
            city_feats.append([
                1.0 if c.owner == me else 0.0,           # owner: me/opp
                row / denom,                             # row
                col / denom,                             # col
                np.log1p(lvl),                           # log(1 + level)
                c.population / (lvl + 1),                # pop / (level+1)
                1.0 if c.border_radius == 2 else 0.0,    # border_radius (bool)
                c.units_owned / max(unit_cap, 1),        # units_owned / capacity
                float(c.is_capital),                     # is_capital
                float(c.is_sieged),                      # is_sieged
                float(c.capture_ready),                  # capture_ready
                float(c.has_walls),                      # has_walls
                float(c.pending_upgrade),                # pending_upgrade
                float(c.has_workshop),                   # has_workshop
                float(c.can_spawn),                      # can_spawn
            ])

    return {
        "unit_types": np.asarray(unit_types, dtype=np.int64),
        "unit_feats": np.asarray(unit_feats, dtype=np.float32).reshape(-1, UNIT_FEAT_DIM),
        "city_feats": np.asarray(city_feats, dtype=np.float32).reshape(-1, CITY_FEAT_DIM),
    }


def collate(states):
    """
    Batch a list of states into padded tensors for the model.

    MCTS hands the eval_fn a variable-length list of states, each with a
    different number of entities. Pad to the batch max and return boolean masks
    (True = real entity, False = pad). The model turns these into the
    src_key_padding_mask the transformer needs.

    Returns a dict of numpy arrays:
        unit_types : int64   [B, Lu]
        unit_feats : float32 [B, Lu, 9]
        unit_mask  : bool    [B, Lu]     True = real
        city_feats : float32 [B, Lc, 14]
        city_mask  : bool    [B, Lc]     True = real
    """

    # finding the maximum lengths
    encoded = [encode_entities(s) for s in states]
    B = len(encoded)
    Lu = max((e["unit_types"].shape[0] for e in encoded), default=0)
    Lc = max((e["city_feats"].shape[0] for e in encoded), default=0)
    Lu = max(Lu, 1)  # keep a length-1 sequence rather than a 0-width tensor
    Lc = max(Lc, 1)

    # we need masks because padding creates potentially fake entities and cities.
    unit_types = np.zeros((B, Lu), dtype=np.int64)
    unit_feats = np.zeros((B, Lu, UNIT_FEAT_DIM), dtype=np.float32)
    unit_mask = np.zeros((B, Lu), dtype=bool)
    city_feats = np.zeros((B, Lc, CITY_FEAT_DIM), dtype=np.float32)
    city_mask = np.zeros((B, Lc), dtype=bool)

    for b, e in enumerate(encoded):
        nu = e["unit_types"].shape[0]
        nc = e["city_feats"].shape[0]
        if nu:
            unit_types[b, :nu] = e["unit_types"]
            unit_feats[b, :nu] = e["unit_feats"]
            unit_mask[b, :nu] = True
        if nc:
            city_feats[b, :nc] = e["city_feats"]
            city_mask[b, :nc] = True

    return {
        "unit_types": unit_types,
        "unit_feats": unit_feats,
        "unit_mask": unit_mask,
        "city_feats": city_feats,
        "city_mask": city_mask,
    }
