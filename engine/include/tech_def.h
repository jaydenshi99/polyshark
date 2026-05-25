#pragma once

#include "types.h"
#include <cstdint>
#include <vector>
constexpr uint32_t tech_bit(TechType t) { return 1u << static_cast<int>(t); }

constexpr int MAX_TECH_RESOURCES = 4;
constexpr int MAX_TECH_TERRAINS  = 4;

struct TechDef {
    const char*  name;
    int          tier;          // 0 = Origin (free); 1, 2, 3 = tech tree depth from root
    uint32_t     unlocks;                                  // bitmask of techs unlocked when this is researched
    std::vector<ResourceType> unlocks_resources;    // ResourceType::None pads unused slots
    std::vector<TerrainType>  defense_terrains;      // TerrainType::None pads unused slots
};

// Indexed by TechType — must stay in sync with the enum order.
//
// Tech tree:
//   Origin -> Hunting, Organisation, Riding, Climbing
//   Hunting  -> Archery
//   Climbing -> Mining
//
static const TechDef TECH_DEFS[] = {
    // name            tier   unlocks                                  unlocks_resources           defense_terrains
    { "Origin",        0,     tech_bit(TechType::Hunting)      |
                               tech_bit(TechType::Organisation) |
                               tech_bit(TechType::Riding)       |
                               tech_bit(TechType::Climbing),           { ResourceType::None },     { TerrainType::None     } },
    { "Hunting",       1,     tech_bit(TechType::Archery),             { ResourceType::Animal },   { TerrainType::None     } },
    { "Organisation",  1,     tech_bit(TechType::Farming)       |
                              tech_bit(TechType::Strategy),             { ResourceType::Fruit },    { TerrainType::None     } },
    { "Farming",       2,     0,                                       { ResourceType::Crop },     { TerrainType::None     } },
    { "Riding",        1,     0,                                       { ResourceType::None },     { TerrainType::None     } },
    { "Climbing",      1,     tech_bit(TechType::Mining),              { ResourceType::None },     { TerrainType::Mountain } },
    { "Archery",       2,     0,                                       { ResourceType::None },     { TerrainType::Forest   } },
    { "Mining",        2,     0,                                       { ResourceType::Metal },    { TerrainType::None     } },
    { "Strategy",      2,     0,                                       { ResourceType::None },     { TerrainType::None     } },
    { "(none)",        0,     0,                                       { ResourceType::None },     { TerrainType::None     } },
};

static_assert(
    static_cast<int>(TechType::Count) ==
    static_cast<int>(sizeof(TECH_DEFS) / sizeof(TECH_DEFS[0])),
    "TECH_DEFS out of sync with TechType enum"
);

inline const TechDef& tech_def(TechType t) {
    return TECH_DEFS[static_cast<int>(t)];
}

// Tech price scales with empire size. For a tech of tier T:
//   cost = (4 + T) + T * c        // = 4 + T*(1 + c)
// where c = owned cities beyond the capital (max(owned - 1, 0)).
//   tier 1 → 5 + 1*c    (Hunting, Organisation, Farming, Riding, Climbing)
//   tier 2 → 6 + 2*c    (Archery, Mining)
//   tier 3 → 7 + 3*c    (future)
// Origin (tier 0) is free.
inline int tech_cost(TechType t, int owned_cities) {
    int tier = tech_def(t).tier;
    if (tier == 0) return 0;
    int c = owned_cities > 1 ? owned_cities - 1 : 0;
    return (4 + tier) + tier * c;
}

// available_techs: union of all unlocks reachable from owned, minus already owned.
// Origin is always pre-owned so its unlocks (tier-1) are always in the pool.
inline uint32_t available_techs(uint32_t owned) {
    uint32_t owned_with_origin = owned | tech_bit(TechType::Origin);
    uint32_t unlocked = 0;
    uint32_t tmp = owned_with_origin;
    while (tmp) {
        int i = __builtin_ctz(tmp);
        unlocked |= TECH_DEFS[i].unlocks;
        tmp &= tmp - 1;
    }
    return unlocked & ~owned_with_origin;
}
