#include "damage_calculator.h"
#include "unit_def.h"
#include "types.h"

DamageCalculator::Result DamageCalculator::resolve(
    const Unit& attacker, const Unit& defender,
    const Tile& def_tile, const float&  defense_bonus,
    int distance, bool opponent_can_see_attacker)
{
    const UnitDef& adef = unit_def(attacker.type());
    const UnitDef& ddef = unit_def(defender.type());

    bool retaliates = (ddef.range >= distance) && opponent_can_see_attacker;

    float attack_force  = adef.attack  * ((float)attacker.hp() / attacker.max_hp());
    float defence_force = ddef.defense * ((float)defender.hp() / defender.max_hp()) * defense_bonus;
    float total         = attack_force + defence_force;

    int def_dmg = (int)((attack_force  / total) * adef.attack  * 4.5f + 0.5f);
    def_dmg = def_dmg < 1 ? 1 : def_dmg;

    bool defender_dies = (defender.hp() - def_dmg <= 0);

    // A dead defender can't strike back.
    int atk_dmg = 0;
    if (retaliates && !defender_dies) {
        atk_dmg = (int)((defence_force / total) * ddef.defense * 4.5f + 0.5f);
        if (atk_dmg < 1) atk_dmg = 1;
    }

    Result r;
    r.defender_damage = def_dmg;
    r.attacker_damage = atk_dmg;
    r.defender_dies   = defender_dies;
    r.attacker_dies   = (attacker.hp() - atk_dmg <= 0);
    return r;
}
