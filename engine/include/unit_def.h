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
    TechType required_tech;  // TechType::Count = trainable by anyone;
                             // TechType::None  = no tech can unlock it (super-unit only).
};

// Indexed by UnitType — must stay in sync with UnitType enum order
static const UnitDef UNIT_DEFS[] = {
    // None
    {},
    // Warrior:   HP  ATK DEF MOV RNG COST  ABILITIES                                         TECH
    {             10,  2,  2,  1,  1,  2,   ABILITY_FORTIFY | ABILITY_DASH,    TechType::Count   },
    // Archer
    {             10,  2,  1,  1,  2,  3,   ABILITY_RANGED  | ABILITY_FORTIFY | ABILITY_DASH, TechType::Archery },
    // Rider
    {             10,  2,  1,  2,  1,  3,   ABILITY_ESCAPE  | ABILITY_FORTIFY | ABILITY_DASH,
                                                                                TechType::Riding  },
    // Defender
    {             15,  1,  3,  1,  1,  3,   ABILITY_FORTIFY,
                                                                                TechType::Strategy },
    // Giant 
    {             40,  5,  4,  1,  1,  0,   ABILITY_STATIC,                    TechType::None    },
};

inline const UnitDef& unit_def(UnitType t) {
    return UNIT_DEFS[static_cast<int>(t)];
}
