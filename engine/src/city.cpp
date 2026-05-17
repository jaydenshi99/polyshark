#include "city.h"
#include "config.h"

int City::stars_per_turn() const {
    if (_is_sieged) return 0;
    return _level + (_is_capital ? cfg::city::CAPITAL_STAR_BONUS : 0) + (_has_workshop ? 1 : 0)  + _parks;
}

void City::set_workshop(bool value) {
    _has_workshop = value;
}

void City::try_levelup() {
    if (_population >= _level + 1) {
        _population -= _level + 1;
        _level++;
        _pending_upgrade = true;
    }
}

void City::add_population(int n) {
    _population += n;
    try_levelup();
}

void City::add_unit() {
    _units_owned++;
}

void City::remove_unit() {
    if (_units_owned > 0) _units_owned--;
}

void City::capture(int new_owner) {
    _owner          = new_owner;
    _is_sieged      = false;
    _capture_ready  = false;
}

void City::update_border(int new_size) {
    if (new_size <= _border_radius) return;
    _border_radius = new_size;
}

void City::set_walls(bool value) {
    if (_has_walls && !value) return;
    _has_walls = value;
}
