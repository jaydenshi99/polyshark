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
};

// Indexed by UnitType — must stay in sync with UnitType enum order
static const UnitDef UNIT_DEFS[] = {
    // None
    {},
    // Warrior:   HP  ATK DEF MOV RNG COST  ABILITIES
    {             10,  2,  2,  1,  1,  2,   ABILITY_FORTIFY | ABILITY_DASH    },
    // Archer
    {             10,  2,  1,  1,  2,  3,   ABILITY_RANGED  | ABILITY_FORTIFY },
    // Rider
    {             10,  2,  1,  2,  1,  3,   ABILITY_ESCAPE  | ABILITY_FORTIFY | ABILITY_DASH },
};

inline const UnitDef& unit_def(UnitType t) {
    return UNIT_DEFS[static_cast<int>(t)];
}
