#include "city.h"

int City::stars_per_turn() const {
    if (_is_sieged) return 0;
    return _level + (_is_capital ? 1 : 0);
}

void City::add_population(int n) {
    _population += n;
    // level up while enough population, capped at MAX_CITY_LEVEL
    while (_population >= _level + 1) {
        _population -= _level + 1;
        _level++;
    }
}

void City::capture(int new_owner) {
    _owner    = new_owner;
    _is_sieged = false;
}
