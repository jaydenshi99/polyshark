#pragma once
#include <cstdint>

namespace cfg {

// ----- Map / generation -----------------------------------------------------
namespace map {
    constexpr int      DEFAULT_SIZE          = 8;     // side length of the square map
    constexpr int      VILLAGE_MIN_SPACING   = 3;      // Chebyshev distance between villages
    constexpr uint8_t  VILLAGE_EDGE_EXCLUDE  = 0b00001011; // bits = forbidden edge distances
    constexpr uint64_t DEFAULT_SEED          = 0;      // 0 = nondeterministic
}

// ----- Terrain spawn rates (Drylands biome) ---------------------------------
namespace rates_drylands {
    // Outer ring (tiles not adjacent to any village)
    constexpr float OUTER_MOUNTAIN = 0.08f;
    constexpr float OUTER_FOREST   = 0.15f;
    constexpr float OUTER_FRUIT    = 0.20f; // | Field
    constexpr float OUTER_CROP     = 0.20f; // | Field, not Fruit
    constexpr float OUTER_METAL    = 0.40f; // | Mountain
    constexpr float OUTER_ANIMAL   = 0.50f; // | Forest

    // Inner ring (tiles Chebyshev-1 of a village) — usually richer
    constexpr float INNER_MOUNTAIN = 0.08f;
    constexpr float INNER_FOREST   = 0.15f;
    constexpr float INNER_FRUIT    = 0.20f;
    constexpr float INNER_CROP     = 0.20f;
    constexpr float INNER_METAL    = 0.40f;
    constexpr float INNER_ANIMAL   = 0.50f;
}

// ----- Cities ---------------------------------------------------------------
namespace city {
    constexpr int  BORDER_RADIUS_BASE        = 1;  // 3x3 footprint
    constexpr int  BORDER_RADIUS_GROWN       = 2;  // 5x5 after L3 Border Growth
    constexpr int  CAPITAL_STAR_BONUS        = 1;  // extra star/turn for capital
    constexpr int  L2_RESOURCES_STAR_REWARD  = 5;  // stars granted on pick
    constexpr int  L3_POP_GROWTH_REWARD      = 3;  // population on pick
    constexpr bool LEVEL_CAP_ENABLED         = false;
}

// ----- Units / combat -------------------------------------------------------
namespace combat {
    constexpr int HEAL_FRIENDLY_TERRITORY = 4; // HP/turn in own borders
    constexpr int HEAL_NEUTRAL            = 2; // HP/turn elsewhere
    constexpr int KILLS_FOR_PROMOTION     = 3;
    constexpr int PROMOTION_HP_BONUS      = 5; // also full-heals
}

// ----- Vision / fog ---------------------------------------------------------
namespace vision {
    constexpr int UNIT_RADIUS     = 1; // 3x3
    constexpr int MOUNTAIN_RADIUS = 2; // extended vision (TBD — placeholder)
}

// ----- Explorer (L1 city upgrade auto-path) ---------------------------------
namespace explorer {
    constexpr int STEPS              = 15;  // total moves the explorer walks
    constexpr int SCAN_RADIUS        = 4;   // BFS depth used to find reachable fog
    constexpr int FOG_REVEAL_CAP     = 4;   // cap on "tiles cleared" credit (real max is 5)
    constexpr int CLEAR_WEIGHT       = 10;  // score per fog tile uncovered on arrival
    constexpr int LIGHTHOUSE_BONUS   = 33;  // a touch more than clearing 3 tiles (3*10)
    constexpr int BACKTRACK_PENALTY  = 3;   // applied when first-step matches recent history
    constexpr int HISTORY_LEN        = 4;   // sliding window of recently visited tiles
}

// ----- Win condition --------------------------------------------------------
namespace win {
    constexpr bool DOMINATION_ONLY = true; // capture enemy capital
    // Future: SCORE_LIMIT, TURN_LIMIT, etc.
}

// ----- Future feature flags (wire up when implemented) ----------------------
namespace features {
    constexpr bool NAVAL_UNITS        = false;
    constexpr bool DIPLOMACY          = false;
    constexpr bool MULTI_TRIBE_BONUS  = false;
    constexpr bool FULL_TECH_TREE     = false;
}

} // namespace cfg
