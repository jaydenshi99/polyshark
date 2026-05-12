#include "unit.h"

void Unit::take_damage(int amount) {
    _hp -= amount;
    if (_hp < 0) _hp = 0;
}

void Unit::spend_movement(int amount) {
    _move_points -= amount;
    if (_move_points < 0) _move_points = 0;
}

void Unit::mark_attacked() {
    _has_attacked = true;
}

void Unit::reset_turn(int base_movement) {
    _move_points  = base_movement;
    _has_attacked = false;
}
