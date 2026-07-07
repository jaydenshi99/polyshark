"""
GameState -> input arrays (numpy). Pure, no torch.

Three input groups, all from the current player's perspective (me/opp, fog-gated):
  - entities : per unit/city tokens (see docs/embedding.md)
  - board    : [18, sz, sz] non-entity spatial grid (see docs/board.md)
  - globals  : [12] non-spatial scalars

Entities are enumerated by scanning tiles (the engine exposes no unit/city list),
gated on is_visible(me) — which equals "explored" in this engine (permanent reveal).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../build/bindings"))
import polyshark  # noqa: E402

UNIT_FEAT_DIM = 9
CITY_FEAT_DIM = 14
BOARD_CHANNELS = 18
GLOBAL_DIM = 12
NUM_UNIT_TYPES = 6  # None, Warrior, Archer, Rider, Defender, Giant (enum incl. index 0)

# Base HP per UnitType index — mirror of UNIT_DEFS in engine/include/unit_def.h.
# Used only to derive is_veteran (max_hp > base_hp). No binding exposes this yet;
# a `unit_base_hp` binding (like unit_base_move) would remove the duplication.
_BASE_HP = [0, 10, 10, 10, 15, 40]

# Base movement per UnitType index, cached from the engine at import.
_BASE_MOVE = [polyshark.unit_base_move(i) for i in range(NUM_UNIT_TYPES)]

# Board channel offsets (see docs/board.md).
_TERRAIN_OFF  = 0   # 5: Field, Forest, Mountain, Water, Village
_RESOURCE_OFF = 5   # 4: Fruit, Crop, Animal, Metal
_BUILDING_OFF = 9   # 6: my/opp x (Mine, Farm, Road)
_BORDER_OFF   = 15  # 2: my_border, opp_border
_VIS_OFF      = 17  # 1: is_visible


def encode_entities(state):
    """
    Returns, from the current player's perspective:
        unit_types : int64   [Nu]       UnitType enum index (embedding lookup)
        unit_feats : float32 [Nu, 9]
        unit_tiles : int64   [Nu]       tile index each unit sits on (for scatter)
        city_feats : float32 [Nc, 14]
        city_tiles : int64   [Nc]       tile index each city sits on (for scatter)
    """
    me = state.current_player()
    sz = state.map_size()
    denom = max(sz - 1, 1)  # normalize row/col to [0, 1]

    unit_types, unit_feats, unit_tiles = [], [], []
    city_feats, city_tiles = [], []

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
                unit_tiles.append(i)
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
            city_tiles.append(i)
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
        "unit_tiles": np.asarray(unit_tiles, dtype=np.int64),
        "city_feats": np.asarray(city_feats, dtype=np.float32).reshape(-1, CITY_FEAT_DIM),
        "city_tiles": np.asarray(city_tiles, dtype=np.int64),
    }


def encode_board(state):
    """[18, sz, sz] non-entity spatial grid, fog-gated. Unexplored tile = all zeros."""
    me = state.current_player()
    sz = state.map_size()
    board = np.zeros((BOARD_CHANNELS, sz, sz), dtype=np.float32)

    for i in range(sz * sz):
        if not state.is_visible(me, i):
            continue
        row, col = divmod(i, sz)
        board[_VIS_OFF, row, col] = 1.0

        tile = state.tile_at(i)

        # Terrain one-hot (enum: None=0, Field=1 .. Village=5 -> ch 0..4).
        terr = int(tile.terrain)
        if terr > 0:
            board[_TERRAIN_OFF + (terr - 1), row, col] = 1.0

        # Resource binary (None=0, Fruit=1 .. Metal=4 -> ch 0..3).
        res = int(tile.resource)
        if res > 0:
            board[_RESOURCE_OFF + (res - 1), row, col] = 1.0

        # Building my/opp (None=0, Mine=1 .. Road=3). Owner via border_city_id.
        bld = int(tile.building)
        bcid = tile.border_city_id
        if bld > 0 and bcid != -1:
            mine = state.get_city(bcid).owner == me
            board[_BUILDING_OFF + (bld - 1) * 2 + (0 if mine else 1), row, col] = 1.0

        # Border ownership.
        if bcid != -1:
            mine = state.get_city(bcid).owner == me
            board[_BORDER_OFF + (0 if mine else 1), row, col] = 1.0

    return board


def encode_globals(state):
    """[12] non-spatial scalars, current player only."""
    me = state.current_player()
    sz = state.map_size()

    income = 0
    for i in range(sz * sz):
        tile = state.tile_at(i)
        if tile.has_city:
            c = state.get_city(tile.city_id)
            if c.owner == me:
                income += c.stars_per_turn

    g = np.zeros(GLOBAL_DIM, dtype=np.float32)
    g[0] = state.get_turn() / 100.0
    g[1] = state.get_stars(me) / 30.0
    g[2] = income / 20.0
    mask = state.techs_mask(me)
    for t in range(1, 9):  # TechType Hunting(1) .. Strategy(8) -> idx 3..10
        g[2 + t] = float((mask >> t) & 1)
    g[11] = 1.0 if state.phase() == polyshark.GameStateType.UpgradingCity else 0.0
    return g


def collate(states):
    """
    Batch a list of states into padded arrays for the model.

    Entity counts vary per state, so pad to the batch max and return boolean masks
    (True = real entity). Board and globals are fixed-shape, no padding needed.

    Returns a dict of numpy arrays:
        unit_types : int64   [B, Lu]
        unit_feats : float32 [B, Lu, 9]
        unit_mask  : bool    [B, Lu]
        unit_tiles : int64   [B, Lu]
        city_feats : float32 [B, Lc, 14]
        city_mask  : bool    [B, Lc]
        city_tiles : int64   [B, Lc]
        board      : float32 [B, 18, sz, sz]
        globals    : float32 [B, 12]
    """
    encoded = [encode_entities(s) for s in states]
    B = len(states)
    Lu = max(max((e["unit_types"].shape[0] for e in encoded), default=0), 1)
    Lc = max(max((e["city_feats"].shape[0] for e in encoded), default=0), 1)

    unit_types = np.zeros((B, Lu), dtype=np.int64)
    unit_feats = np.zeros((B, Lu, UNIT_FEAT_DIM), dtype=np.float32)
    unit_mask = np.zeros((B, Lu), dtype=bool)
    unit_tiles = np.zeros((B, Lu), dtype=np.int64)
    city_feats = np.zeros((B, Lc, CITY_FEAT_DIM), dtype=np.float32)
    city_mask = np.zeros((B, Lc), dtype=bool)
    city_tiles = np.zeros((B, Lc), dtype=np.int64)

    for b, e in enumerate(encoded):
        nu = e["unit_types"].shape[0]
        nc = e["city_feats"].shape[0]
        if nu:
            unit_types[b, :nu] = e["unit_types"]
            unit_feats[b, :nu] = e["unit_feats"]
            unit_tiles[b, :nu] = e["unit_tiles"]
            unit_mask[b, :nu] = True
        if nc:
            city_feats[b, :nc] = e["city_feats"]
            city_tiles[b, :nc] = e["city_tiles"]
            city_mask[b, :nc] = True

    board = np.stack([encode_board(s) for s in states])
    globals_ = np.stack([encode_globals(s) for s in states])

    return {
        "unit_types": unit_types,
        "unit_feats": unit_feats,
        "unit_mask": unit_mask,
        "unit_tiles": unit_tiles,
        "city_feats": city_feats,
        "city_mask": city_mask,
        "city_tiles": city_tiles,
        "board": board,
        "globals": globals_,
    }
