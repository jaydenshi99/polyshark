"""
Encodes a GameState into tensors for the value network.

Perspective: always the current player (me vs opponent), so the network is
player-invariant — same representation regardless of which player is active.

Fog rule: if is_visible(me, tile) == False, all feature channels for that tile
are 0. Only the two fog indicator planes carry signal for hidden tiles.

Channel layout — spatial [SPATIAL_CHANNELS, map_size, map_size]
---------------------------------------------------------------
  0- 4  Terrain one-hot   Field, Forest, Mountain, Water, Village
  5- 8  Resource binary    Fruit, Crop, Animal, Metal        (None → all 0)
  9-11  Building binary    Mine, Farm, Road                  (None → all 0)
 12-13  Border             my_border, opp_border
 14-15  Fog                is_visible, is_explored           (always written)
 16-24  My units  (9ch)    has, warrior, archer, rider,
                           hp/max_hp, has_attacked,
                           move_pts/base_move, kills/3, promotion_ready
 25-33  Opp units (9ch)    same — zeroed when tile not visible
 34-41  My cities (8ch)    has, is_capital, level/10,
                           pop/(level+1), is_sieged,
                           has_walls, has_workshop, pending_upgrade
 42-49  Opp cities (8ch)   same — zeroed when tile not visible

Global feature vector [GLOBAL_SIZE]
------------------------------------
  0    turn / 100
  1    my_stars / 30
  2    my_income / 20    (sum of city.stars_per_turn for my cities)
  3-10 my_tech[0..7]     one bit per TechType (Origin=0 .. Mining=7)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../build/bindings"))
import polyshark

SPATIAL_CHANNELS = 50
GLOBAL_SIZE      = 11

# Spatial channel offsets
_TERRAIN_OFF  = 0
_RESOURCE_OFF = 5
_BUILDING_OFF = 9
_BORDER_OFF   = 12
_FOG_OFF      = 14
_MY_UNIT_OFF  = 16
_OPP_UNIT_OFF = 25
_MY_CITY_OFF  = 34
_OPP_CITY_OFF = 42

# Normalisation denominators
_MAX_TURN    = 100
_MAX_STARS   = 30
_MAX_INCOME  = 20
_MAX_KILLS   = 3
_MAX_LEVEL   = 10
_N_TECHS     = 8   # TechType::Count

# Base movement per unit type int value (0=None,1=Warrior,2=Archer,3=Rider).
# Cached at import time to avoid repeated C++ calls.
_BASE_MOVE = [polyshark.unit_base_move(i) for i in range(4)]


def encode(state):
    """
    Parameters
    ----------
    state : polyshark.GameState

    Returns
    -------
    spatial    : float32 ndarray  [SPATIAL_CHANNELS, map_size, map_size]
    global_vec : float32 ndarray  [GLOBAL_SIZE]
    """
    me  = state.current_player()
    sz  = state.map_size()
    n   = sz * sz

    spatial   = np.zeros((SPATIAL_CHANNELS, sz, sz), dtype=np.float32)
    my_income = 0

    for idx in range(n):
        row, col = divmod(idx, sz)

        visible  = state.is_visible(me, idx)
        explored = state.is_explored(me, idx)

        # Fog planes written regardless of visibility.
        spatial[_FOG_OFF + 0, row, col] = float(visible)
        spatial[_FOG_OFF + 1, row, col] = float(explored)

        if not visible:
            continue

        tile = state.tile_at(idx)

        # Terrain one-hot (enum int matches channel offset 0-4)
        spatial[_TERRAIN_OFF + int(tile.terrain), row, col] = 1.0

        # Resource binary (ResourceType::None == 0, skip; Fruit=1 → ch 0, etc.)
        r = int(tile.resource)
        if r > 0:
            spatial[_RESOURCE_OFF + (r - 1), row, col] = 1.0

        # Building binary (BuildingType::None == 0, skip; Mine=1 → ch 0, etc.)
        b = int(tile.building)
        if b > 0:
            spatial[_BUILDING_OFF + (b - 1), row, col] = 1.0

        # Border ownership
        bcid = tile.border_city_id
        if bcid != -1:
            border_owner = state.get_city(bcid).owner
            spatial[_BORDER_OFF + (0 if border_owner == me else 1), row, col] = 1.0

        # Unit
        if tile.has_unit:
            unit = state.get_unit(tile.unit_id)
            if unit.is_alive:
                off      = _MY_UNIT_OFF if unit.owner == me else _OPP_UNIT_OFF
                utype    = int(unit.type)   # 1=Warrior 2=Archer 3=Rider
                base_mov = _BASE_MOVE[utype] if utype < len(_BASE_MOVE) else 1
                spatial[off + 0, row, col] = 1.0
                if 1 <= utype <= 3:
                    spatial[off + utype, row, col] = 1.0
                spatial[off + 4, row, col] = unit.hp / max(unit.max_hp, 1)
                spatial[off + 5, row, col] = float(unit.has_attacked)
                spatial[off + 6, row, col] = unit.move_points / max(base_mov, 1)
                spatial[off + 7, row, col] = min(unit.kills / _MAX_KILLS, 1.0)
                spatial[off + 8, row, col] = float(unit.promotion_ready)

        # City
        if tile.has_city:
            city = state.get_city(tile.city_id)
            off  = _MY_CITY_OFF if city.owner == me else _OPP_CITY_OFF
            lvl  = city.level
            spatial[off + 0, row, col] = 1.0
            spatial[off + 1, row, col] = float(city.is_capital)
            spatial[off + 2, row, col] = min(lvl / _MAX_LEVEL, 1.0)
            spatial[off + 3, row, col] = city.population / (lvl + 1)
            spatial[off + 4, row, col] = float(city.is_sieged)
            spatial[off + 5, row, col] = float(city.has_walls)
            spatial[off + 6, row, col] = float(city.has_workshop)
            spatial[off + 7, row, col] = float(city.pending_upgrade)

            if city.owner == me:
                my_income += city.stars_per_turn

    # Global vector
    techs      = state.get_techs(me)
    tech_bits  = [(techs >> i) & 1 for i in range(_N_TECHS)]
    global_vec = np.array([
        state.get_turn() / _MAX_TURN,
        state.get_stars(me) / _MAX_STARS,
        my_income / _MAX_INCOME,
        *tech_bits,
    ], dtype=np.float32)

    return spatial, global_vec
