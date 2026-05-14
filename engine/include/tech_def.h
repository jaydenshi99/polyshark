#pragma once

#include "types.h"
#include <cstdint>

constexpr uint32_t tech_bit(TechType t) { return 1u << static_cast<int>(t); }

struct TechDef {
    const char* name;
    int         cost;
    uint32_t    unlocks;  // bitmask of techs unlocked when this is researched
};

// Indexed by TechType — must stay in sync with the enum order.
//
// Tech tree:
//   Origin -> Hunting, Organisation, Riding, Climbing
//   Hunting  -> Archery
//   Climbing -> Mining
//
static const TechDef TECH_DEFS[] = {
    // name            cost   unlocks
    { "Origin",        0,     tech_bit(TechType::Hunting)      |
                               tech_bit(TechType::Organisation) |
                               tech_bit(TechType::Riding)       |
                               tech_bit(TechType::Climbing)     },
    { "Hunting",       5,     tech_bit(TechType::Archery)      },
    { "Organisation",  5,     tech_bit(TechType::Farming)      },
    { "Farming",       5,     0                                },
    { "Riding",        5,     0                                },
    { "Climbing",      5,     tech_bit(TechType::Mining)       },
    { "Archery",       6,     0                                },
    { "Mining",        6,     0                                },
};

static_assert(
    static_cast<int>(TechType::Count) ==
    static_cast<int>(sizeof(TECH_DEFS) / sizeof(TECH_DEFS[0])),
    "TECH_DEFS out of sync with TechType enum"
);

inline const TechDef& tech_def(TechType t) {
    return TECH_DEFS[static_cast<int>(t)];
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
