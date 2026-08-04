"""
AISpec — single source of derived shape constants for the NN, encoder and
action codec. Built from an `EngineSettings` (the C++-side runtime config).

Everything that used to be a hardcoded module-level constant (MAP_SIZE,
ACTION_SIZE, channel offsets, …) now lives here and is derived from the
engine's own enums + the active settings. Pass an `AISpec` (or its
`settings`) wherever shape information is needed.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../build/bindings"))
import polyshark


# --- Per-side unit channels in the encoder ---
# These are *encoder-internal* and need to stay in lockstep with encoder.py.
# 4 fixed channels + one one-hot per unit type (excluding None).
#   has, hp_frac, has_attacked, mp_frac, kills_frac, promotion_ready  (6 fixed)
#   + unit-type one-hots
# Total per side = 6 + n_unit_types.
_FIXED_UNIT_CHANNELS = 6

# Per-side city channels: has, is_capital, level_norm, pop_frac,
# is_sieged, has_walls, has_workshop, pending_upgrade  (8 fixed).
_CITY_CHANNELS = 8


class AISpec:
    """Derived shape constants for the NN, computed from an EngineSettings."""

    def __init__(self, settings: "polyshark.EngineSettings | None" = None):
        self.settings = settings if settings is not None else polyshark.EngineSettings()

        # ---- Board dims ----
        self.map_size = self.settings.map_size
        self.n_tiles  = self.map_size * self.map_size

        # ---- Engine-enum counts (excluding None sentinel where applicable) ----
        self.n_unit_types     = int(polyshark.UnitType.Count) - 1        # exclude None
        self.n_tech_types     = int(polyshark.TechType.Count)
        self.n_resource_types = int(polyshark.ResourceType.Count) - 1    # exclude None
        self.n_building_types = int(polyshark.BuildingType.Count) - 1    # exclude None
        self.n_upgrade_types  = int(polyshark.CityUpgradeType.Count)
        # Move/Attack offset square: 5x5 around source (dr/dc in {-2..2}).
        self.move_offsets     = 25

        # ---- Encoder channel layout ----
        # Spatial: terrain(5) + resource(N_R) + building(N_B) + border(2) + fog(2)
        #          + 2 * (FIXED_UNIT + n_unit_types) + 2 * CITY_CHANNELS
        self.n_terrain_types = 5  # Field, Forest, Mountain, Water, Village
        self.unit_channels_per_side = _FIXED_UNIT_CHANNELS + self.n_unit_types
        self.spatial_channels = (
            self.n_terrain_types
            + self.n_resource_types
            + self.n_building_types
            + 2  # border
            + 2  # fog
            + 2 * self.unit_channels_per_side
            + 2 * _CITY_CHANNELS
        )

        # Channel offsets (used by encoder.py)
        self.terrain_off  = 0
        self.resource_off = self.terrain_off  + self.n_terrain_types
        self.building_off = self.resource_off + self.n_resource_types
        self.border_off   = self.building_off + self.n_building_types
        self.fog_off      = self.border_off   + 2
        self.my_unit_off  = self.fog_off      + 2
        self.opp_unit_off = self.my_unit_off  + self.unit_channels_per_side
        self.my_city_off  = self.opp_unit_off + self.unit_channels_per_side
        self.opp_city_off = self.my_city_off  + _CITY_CHANNELS

        # Within a unit block: layout is
        #   [0]            has
        #   [1..n]         type one-hot (n = n_unit_types)
        #   [n+1]          hp_frac
        #   [n+2]          has_attacked
        #   [n+3]          mp_frac
        #   [n+4]          kills_frac
        #   [n+5]          promotion_ready
        self.unit_type_one_hot_off = 1
        self.unit_hp_frac_off      = 1 + self.n_unit_types
        self.unit_attacked_off     = 2 + self.n_unit_types
        self.unit_mp_off           = 3 + self.n_unit_types
        self.unit_kills_off        = 4 + self.n_unit_types
        self.unit_promo_off        = 5 + self.n_unit_types

        # ---- Global feature vector ----
        # turn_norm, my_stars_norm, my_income_norm, tech_bits[n_tech_types]
        self.global_size = 3 + self.n_tech_types

        # ---- Action space layout ----
        n = self.n_tiles
        self.move_off    = 0
        self.attack_off  = self.move_off    + n * self.move_offsets
        self.train_off   = self.attack_off  + n * self.move_offsets
        self.harvest_off = self.train_off   + n * self.n_unit_types
        self.upgrade_off = self.harvest_off + n * self.n_resource_types
        self.research_off = self.upgrade_off + n * self.n_upgrade_types
        self.capture_off  = self.research_off + self.n_tech_types
        self.end_turn_idx = self.capture_off  + n
        self.action_size  = self.end_turn_idx + 1

        # ---- Cached base movement per unit type (index = UnitType int) ----
        # Includes index 0 (None) → 0 movement for safety.
        n_types_with_none = self.n_unit_types + 1
        self._base_move = [
            polyshark.unit_base_move(i) if i > 0 else 0
            for i in range(n_types_with_none)
        ]

    # ---- Helpers ----

    def base_move(self, unit_type_int: int) -> int:
        if 0 <= unit_type_int < len(self._base_move):
            return self._base_move[unit_type_int]
        return 1  # defensive fallback

    def __repr__(self):
        return (f"AISpec(map_size={self.map_size}, "
                f"action_size={self.action_size}, "
                f"spatial_channels={self.spatial_channels}, "
                f"global_size={self.global_size})")


# A module-level default for code paths that haven't been threaded through yet.
DEFAULT_SPEC = AISpec()
