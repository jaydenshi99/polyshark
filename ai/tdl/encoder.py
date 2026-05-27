"""
Input encoder for the TDL value network.

encode(state) -> (spatial, global_vec)
  spatial    : float32 ndarray [C_IN, MAP_SIZE, MAP_SIZE]  (C_IN = 60)
  global_vec : float32 ndarray [G]                         (G    = 11)

All channels are 0 for unexplored tiles (gated on is_explored).
The encoding is always from the perspective of state.current_player().

Spatial channel layout
──────────────────────
  0    is_explored

  1-5  terrain one-hot  (Field, Forest, Mountain, Water, Village)
  6-9  resource one-hot (Fruit, Crop, Animal, Metal)

 10    my_Mine          building ownership via border_city_id → city.owner
 11    opp_Mine
 12    my_Farm
 13    opp_Farm
 14    my_Road
 15    opp_Road

 16    my_border
 17    opp_border

 18    has_my_unit
 19-23 my_unit type one-hot (Warrior, Archer, Rider, Defender, Giant)
 24    my_unit hp / max_hp
 25    my_unit has_attacked
 26    my_unit move_pts / base_move
 27    my_unit kills / 10
 28    my_unit promotion_ready

 29    has_opp_unit
 30-34 opp_unit type one-hot
 35    opp_unit hp / max_hp
 36    opp_unit has_attacked
 37    opp_unit move_pts / base_move
 38    opp_unit kills / 10
 39    opp_unit promotion_ready

 40    has_my_city
 41    my_city is_capital
 42    my_city level / 10
 43    my_city pop / (level+1)
 44    my_city units_owned / unit_capacity
 45    my_city is_sieged
 46    my_city has_walls
 47    my_city has_workshop
 48    my_city pending_upgrade
 49    my_city capture_ready

 50    has_opp_city
 51    opp_city is_capital
 52    opp_city level / 10
 53    opp_city pop / (level+1)
 54    opp_city units_owned / unit_capacity
 55    opp_city is_sieged
 56    opp_city has_walls
 57    opp_city has_workshop
 58    opp_city pending_upgrade
 59    opp_city capture_ready

Global vector layout
────────────────────
  0    turn / 100
  1    my_stars / 30
  2    my_income / 20
  3-10 tech bits: Hunting, Organisation, Farming, Riding, Climbing, Archery, Mining, Strategy
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/bindings'))
import polyshark

C_IN = 60
G    = 11

# Unit type → offset from has_unit channel (channel 18 for mine, 29 for opp)
_UNIT_TYPE_OFF = {
    polyshark.UnitType.Warrior:  1,
    polyshark.UnitType.Archer:   2,
    polyshark.UnitType.Rider:    3,
    polyshark.UnitType.Defender: 4,
    polyshark.UnitType.Giant:    5,
}

# Building type → (my_channel, opp_channel)
_BUILDING_CH = {
    polyshark.BuildingType.Mine: (10, 11),
    polyshark.BuildingType.Farm: (12, 13),
    polyshark.BuildingType.Road: (14, 15),
}

# TechType int value → global vector index (Hunting=1 → g[3], ..., Strategy=8 → g[10])
_TECH_GIDX = {i + 1: i + 3 for i in range(8)}  # tech ints 1-8 → g[3]-g[10]

# Cached base movement per unit type int
_base_move_cache: dict[int, int] = {}

def _base_move(unit_type_int: int) -> int:
    if unit_type_int not in _base_move_cache:
        _base_move_cache[unit_type_int] = polyshark.unit_base_move(unit_type_int)
    return _base_move_cache[unit_type_int]


def encode(state) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode a GameState into (spatial [C_IN, H, W], global_vec [G]).
    Always encoded from state.current_player()'s perspective.
    """
    p   = state.current_player()
    opp = 1 - p
    sz  = state.map_size()

    spatial = np.zeros((C_IN, sz, sz), dtype=np.float32)

    for idx in range(sz * sz):
        if not state.is_explored(p, idx):
            continue  # all channels stay 0

        row, col = divmod(idx, sz)
        tile     = state.tile_at(idx)

        # ch 0: explored
        spatial[0, row, col] = 1.0

        # ch 1-5: terrain
        ter = tile.terrain
        if   ter == polyshark.TerrainType.Field:    spatial[1, row, col] = 1.0
        elif ter == polyshark.TerrainType.Forest:   spatial[2, row, col] = 1.0
        elif ter == polyshark.TerrainType.Mountain: spatial[3, row, col] = 1.0
        elif ter == polyshark.TerrainType.Water:    spatial[4, row, col] = 1.0
        elif ter == polyshark.TerrainType.Village:  spatial[5, row, col] = 1.0

        # ch 6-9: resource
        res = tile.resource
        if   res == polyshark.ResourceType.Fruit:  spatial[6, row, col] = 1.0
        elif res == polyshark.ResourceType.Crop:   spatial[7, row, col] = 1.0
        elif res == polyshark.ResourceType.Animal: spatial[8, row, col] = 1.0
        elif res == polyshark.ResourceType.Metal:  spatial[9, row, col] = 1.0

        # ch 10-15: buildings — ownership from border_city_id → city.owner
        bld = tile.building
        if bld != polyshark.BuildingType.No_Building:
            chs = _BUILDING_CH.get(bld)
            if chs is not None:
                bcid = tile.border_city_id
                if bcid >= 0:
                    b_owner = state.get_city(bcid).owner
                    spatial[chs[0] if b_owner == p else chs[1], row, col] = 1.0

        # ch 16-17: borders
        bcid = tile.border_city_id
        if bcid >= 0:
            b_owner = state.get_city(bcid).owner
            spatial[16 if b_owner == p else 17, row, col] = 1.0

        # ch 18-28 / 29-39: units
        if tile.has_unit:
            u      = state.get_unit(tile.unit_id)
            base   = 18 if u.owner == p else 29
            ut_int = int(u.type)
            off    = _UNIT_TYPE_OFF.get(u.type)

            spatial[base, row, col] = 1.0
            if off is not None:
                spatial[base + off, row, col] = 1.0
            if u.max_hp > 0:
                spatial[base + 6, row, col] = u.hp / u.max_hp
            spatial[base + 7, row, col] = float(u.has_attacked)
            bm = _base_move(ut_int)
            if bm > 0:
                spatial[base + 8, row, col] = u.move_points / bm
            spatial[base + 9, row, col]  = min(u.kills, 10) / 10.0
            spatial[base + 10, row, col] = float(u.promotion_ready)

        # ch 40-49 / 50-59: cities
        if tile.has_city:
            c    = state.get_city(tile.city_id)
            base = 40 if c.owner == p else 50
            cap  = c.unit_capacity  # level + 1

            spatial[base,      row, col] = 1.0
            spatial[base + 1,  row, col] = float(c.is_capital)
            spatial[base + 2,  row, col] = c.level / 10.0
            spatial[base + 3,  row, col] = c.population / cap
            spatial[base + 4,  row, col] = c.units_owned / cap
            spatial[base + 5,  row, col] = float(c.is_sieged)
            spatial[base + 6,  row, col] = float(c.has_walls)
            spatial[base + 7,  row, col] = float(c.has_workshop)
            spatial[base + 8,  row, col] = float(c.pending_upgrade)
            spatial[base + 9,  row, col] = float(c.capture_ready)

    # Global vector
    gvec = np.zeros(G, dtype=np.float32)
    gvec[0] = state.get_turn() / 100.0
    gvec[1] = state.get_stars(p) / 30.0

    income = sum(
        state.get_city(state.tile_at(i).city_id).stars_per_turn
        for i in range(sz * sz)
        if state.tile_at(i).has_city and state.get_city(state.tile_at(i).city_id).owner == p
    )
    gvec[2] = income / 20.0

    mask = state.techs_mask(p)
    for tech_int, gidx in _TECH_GIDX.items():
        gvec[gidx] = float((mask >> tech_int) & 1)

    return spatial, gvec


def encode_batch(states) -> tuple[np.ndarray, np.ndarray]:
    """Encode a list of states. Returns ([B, C_IN, H, W], [B, G])."""
    spatials, gvecs = zip(*(encode(s) for s in states))
    return np.stack(spatials), np.stack(gvecs)
