#pragma once

#include "unit.h"
#include "tile.h"

// Stateless combat math. All methods are static — instantiating is meaningless.
class DamageCalculator {
public:
    struct Result {
        int  attacker_damage;
        int  defender_damage;
        bool attacker_dies;
        bool defender_dies;
    };

    // Retaliation is decided internally:
    //   retaliates = (defender.range >= distance) && opponent_can_see_attacker
    static Result resolve(const Unit& attacker, const Unit& defender,
                          const Tile& def_tile,  const float& defense_bonus,
                          int distance, bool opponent_can_see_attacker);
};
