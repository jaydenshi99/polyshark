"""
Encodes a GameState into tensors for the value network.

Perspective: always the current player (me vs opponent), so the network is
player-invariant.

Fog rule: if is_visible(me, tile) is False, all feature channels for that
tile are 0; only the two fog indicator planes carry signal for hidden tiles.

Channel layout is determined entirely by `AISpec` so the encoder adapts to
map size / unit count / tech count automatically.

Spatial channel ordering per spec:
   terrain (5) | resource (n_res) | building (n_bld) | border (2) | fog (2)
   my_units (FIXED + n_units) | opp_units (FIXED + n_units)
   my_cities (8)              | opp_cities (8)

Per-side unit block ordering:
   [0]                 has
   [1..n_units]        type one-hot
   [+1]                hp_frac
   [+2]                has_attacked
   [+3]                mp_frac
   [+4]                kills_frac
   [+5]                promotion_ready

Per-side city block (8 channels, fixed):
   has, is_capital, level_norm, pop_frac,
   is_sieged, has_walls, has_workshop, pending_upgrade

Global feature vector:
   [0]  turn / MAX_TURN
   [1]  my_stars / MAX_STARS
   [2]  my_income / MAX_INCOME
   [3..3+n_tech] one bit per TechType
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../build/bindings"))
import polyshark

from spec import AISpec, DEFAULT_SPEC

# Normalisation denominators (kept as plain constants since they're hyperparams,
# not engine-derived).
_MAX_TURN   = 100
_MAX_STARS  = 30
_MAX_INCOME = 20
_MAX_KILLS  = 3
_MAX_LEVEL  = 10


def encode(state, spec: AISpec = DEFAULT_SPEC):
    """
    Returns
    -------
    spatial    : float32 ndarray  [spec.spatial_channels, sz, sz]
    global_vec : float32 ndarray  [spec.global_size]
    """
    me  = state.current_player()
    sz  = spec.map_size
    n   = spec.n_tiles

    spatial = np.zeros((spec.spatial_channels, sz, sz), dtype=np.float32)
    my_income = 0

    for idx in range(n):
        row, col = divmod(idx, sz)

        visible  = state.is_visible(me, idx)
        explored = state.is_explored(me, idx)

        spatial[spec.fog_off + 0, row, col] = float(visible)
        spatial[spec.fog_off + 1, row, col] = float(explored)

        if not visible:
            continue

        tile = state.tile_at(idx)

        # Terrain one-hot (enum int matches channel index 0..4).
        spatial[spec.terrain_off + int(tile.terrain), row, col] = 1.0

        # Resource (None == 0; skip).
        r = int(tile.resource)
        if r > 0:
            spatial[spec.resource_off + (r - 1), row, col] = 1.0

        # Building (None == 0; skip).
        b = int(tile.building)
        if b > 0:
            spatial[spec.building_off + (b - 1), row, col] = 1.0

        # Border ownership.
        bcid = tile.border_city_id
        if bcid != -1:
            border_owner = state.get_city(bcid).owner
            spatial[spec.border_off + (0 if border_owner == me else 1), row, col] = 1.0

        # Unit.
        if tile.has_unit:
            unit = state.get_unit(tile.unit_id)
            if unit.is_alive:
                base = spec.my_unit_off if unit.owner == me else spec.opp_unit_off
                utype = int(unit.type)
                spatial[base + 0, row, col] = 1.0
                if 1 <= utype <= spec.n_unit_types:
                    spatial[base + spec.unit_type_one_hot_off + (utype - 1), row, col] = 1.0
                spatial[base + spec.unit_hp_frac_off, row, col] = unit.hp / max(unit.max_hp, 1)
                spatial[base + spec.unit_attacked_off, row, col] = float(unit.has_attacked)
                bmv = spec.base_move(utype)
                spatial[base + spec.unit_mp_off, row, col] = unit.move_points / max(bmv, 1)
                spatial[base + spec.unit_kills_off, row, col] = min(unit.kills / _MAX_KILLS, 1.0)
                spatial[base + spec.unit_promo_off, row, col] = float(unit.promotion_ready)

        # City.
        if tile.has_city:
            city = state.get_city(tile.city_id)
            base = spec.my_city_off if city.owner == me else spec.opp_city_off
            lvl  = city.level
            spatial[base + 0, row, col] = 1.0
            spatial[base + 1, row, col] = float(city.is_capital)
            spatial[base + 2, row, col] = min(lvl / _MAX_LEVEL, 1.0)
            spatial[base + 3, row, col] = city.population / (lvl + 1)
            spatial[base + 4, row, col] = float(city.is_sieged)
            spatial[base + 5, row, col] = float(city.has_walls)
            spatial[base + 6, row, col] = float(city.has_workshop)
            spatial[base + 7, row, col] = float(city.pending_upgrade)
            if city.owner == me:
                my_income += city.stars_per_turn

    # Global vector.
    techs     = state.get_techs(me)
    owned_idx = {int(t) for t in techs}
    tech_bits = [1 if i in owned_idx else 0 for i in range(spec.n_tech_types)]
    global_vec = np.array(
        [
            state.get_turn() / _MAX_TURN,
            state.get_stars(me) / _MAX_STARS,
            my_income / _MAX_INCOME,
            *tech_bits,
        ],
        dtype=np.float32,
    )

    return spatial, global_vec
