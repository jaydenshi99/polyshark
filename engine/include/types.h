#pragma once

#include <cstdint>

enum class TerrainType {
    Field,
    Forest,
    Mountain,
    Water,
    Village,  // uncaptured only; terrain changes to Field when captured
};

enum class ResourceType {
    None,
    Fruit,   // field food
    Game,    // forest food
    Metal,   // mountain, requires Mining
};

enum class UnitType {
    None = 0,
    Warrior,
    Archer,
    Rider,
    Count,
};

// Unit ability bitmask flags
constexpr uint32_t ABILITY_FORTIFY  = 1 << 0;  // ×1.5 defence on forest/mountain/city
constexpr uint32_t ABILITY_DASH     = 1 << 1;  // can move after non-attack action
constexpr uint32_t ABILITY_ESCAPE   = 1 << 2;  // can move again after attacking
constexpr uint32_t ABILITY_RANGED   = 1 << 3;  // attacks at 2-tile range, doesn't advance


enum class TechType {
    Mining = 0,
    Archery,
    Riding,
    Count,
};
