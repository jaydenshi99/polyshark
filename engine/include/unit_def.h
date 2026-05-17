#pragma once

#include <cstdint>
#include "types.h"

struct UnitDef {
    int      hp;
    int      attack;
    int      defense;
    int      movement;
    int      range;
    int      cost;
    uint32_t abilities;
    TechType required_tech;  // TechType::Count = no tech required
};

// Indexed by UnitType — must stay in sync with UnitType enum order
static const UnitDef UNIT_DEFS[] = {
    // None
    {},
    // Warrior:   HP  ATK DEF MOV RNG COST  ABILITIES                                         TECH
    {             10,  2,  2,  1,  1,  2,   ABILITY_FORTIFY | ABILITY_DASH,    TechType::Count   },
    // Archer
    {             10,  2,  1,  1,  2,  3,   ABILITY_RANGED  | ABILITY_FORTIFY, TechType::Archery },
    // Rider
    {             10,  2,  1,  2,  1,  3,   ABILITY_ESCAPE  | ABILITY_FORTIFY | ABILITY_DASH,
                                                                                TechType::Riding  },
    {             15,  1,  3,  1,  1,  3,   ABILITY_FORTIFY,
                                                                                TechType::Strategy  },
};

inline const UnitDef& unit_def(UnitType t) {
    return UNIT_DEFS[static_cast<int>(t)];
}
