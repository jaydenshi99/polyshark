#include "city.h"
#include "logger.h"

int City::stars_per_turn() const {
    if (_is_sieged) return 0;
    return _level + (_is_capital ? 1 : 0) + (_has_workshop ? 1 : 0);
}

void City::set_workshop(bool value) {
    Logger::print("City workshop: %s -> %s", _has_workshop ? "true" : "false", value ? "true" : "false");
    _has_workshop = value;
}

void City::try_levelup() {
    if (_population >= _level + 1) {
        _population -= _level + 1;
        _level++;
        _pending_upgrade = true;
        Logger::print("City levelled up to %d (pop remainder: %d)", _level, _population);
    }
}

void City::add_population(int n) {
    _population += n;
    Logger::print("City pop %d -> %d (needs %d to level)", _population - n, _population, _level + 1);
    try_levelup();
}

void City::capture(int new_owner) {
    Logger::print("City captured: owner %d -> %d", _owner, new_owner);
    _owner     = new_owner;
    _is_sieged = false;
}

void City::update_border(int new_size) {
    if (new_size <= _border_radius) {
        Logger::print("City border update ignored: %d -> %d (no shrink)", _border_radius, new_size);
        return;
    }
    Logger::print("City border expanded: %d -> %d", _border_radius, new_size);
    _border_radius = new_size;
}

void City::set_walls(bool value) {
    if (_has_walls && !value) {
        Logger::print("City walls cannot be removed once built");
        return;
    }
    Logger::print("City walls: %s -> %s", _has_walls ? "true" : "false", value ? "true" : "false");
    _has_walls = value;
}
