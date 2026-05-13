#pragma once

#include <cstdint>

constexpr int MAX_CITY_LEVEL = 3;

class City {
public:
    int  owner()      const { return _owner; }
    int  level()      const { return _level; }
    int  population() const { return _population; }
    int  tile_index() const { return _tile_index; }
    bool is_capital() const { return _is_capital; }
    bool is_sieged()  const { return _is_sieged; }

    // Income: 1★ per level + 1★ if capital, 0 if sieged
    int stars_per_turn() const;

    // Unit capacity: level + 1
    int unit_capacity() const { return _level + 1; }

    void add_population(int n);
    // Caller must also set tile terrain to Field if it was Village
    void capture(int new_owner);

    void set_owner(int owner)      { _owner = owner; }
    void set_tile(int tile_index)  { _tile_index = tile_index; }
    void set_capital(bool capital) { _is_capital = capital; }
    void set_sieged(bool sieged)   { _is_sieged = sieged; }

private:
    int  _owner      = -1;
    int  _level      = 1;
    int  _population = 0;
    int  _tile_index = -1;
    bool _is_capital = false;
    bool _is_sieged  = false;
};
