#pragma once

#include <cstdint>

enum class TerrainType {
    None,
    Field,
    Forest,
    Mountain,
    Water,
    Village,  // uncaptured only; terrain changes to Field when captured
};

enum class ResourceType {
    None,
    Fruit,   // field food
    Crop,    // field food (requires Farming)
    Animal,  // forest food
    Metal,   // mountain ore — requires Mining tech
    Count,
};

enum class UnitType {
    None = 0,
    Warrior,
    Archer,
    Rider,
    Defender,
    Count,
};

// Unit ability bitmask flags
constexpr uint32_t ABILITY_FORTIFY  = 1 << 0;  // ×1.5 defence on forest/mountain/city
constexpr uint32_t ABILITY_DASH     = 1 << 1;  // can move after non-attack action
constexpr uint32_t ABILITY_ESCAPE   = 1 << 2;  // can move again after attacking
constexpr uint32_t ABILITY_RANGED   = 1 << 3;  // attacks at 2-tile range, doesn't advance


// City upgrade choices presented to the player on each level-up.
// Two options per level bracket; L4_PLUS_* are reused for level 5+.
enum class CityUpgradeType {
    // Level 1 -> 2
    L1_WORKSHOP   = 0, L1_EXPLORER,

    // Level 2 -> 3
    L2_RESOURCES, L2_WALLS,

    // Level 3 -> 4
    L3_BORDER_GROWTH, L3_POPULATION_GROWTH,

    // Level 4 -> 5  (also used for every level-up beyond 5)
    L4_PARK,  L4_SUPERUNIT,

    Count,
};

enum class TechType {
    Origin = 0,   // root node — always owned; unlocks tier-1 techs
    Hunting,      // unlocks Animal harvesting; unlocks Archery
    Organisation,
    Farming,      // unlocks Crop harvesting
    Riding,
    Climbing,     // allows units to enter mountain tiles; unlocks Mining
    Archery,      // requires Hunting
    Mining,       // unlocks Metal harvesting; requires Climbing
    Strategy,
    Count,
};
