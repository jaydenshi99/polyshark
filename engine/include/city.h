#pragma once

#include <cstdint>

class City {
public:
    int  owner()      const { return _owner; }
    int  level()      const { return _level; }
    int  population() const { return _population; }
    int  tile_index() const { return _tile_index; }
    bool is_capital() const { return _is_capital; }
    bool is_sieged()       const { return _is_sieged; }
    // True when the attacking player started their turn already on this city — capture is now legal.
    bool capture_ready()   const { return _capture_ready; }
    bool has_pending_upgrade() const { return _pending_upgrade; }
    bool has_walls()     const { return _has_walls; }
    bool has_workshop()  const { return _has_workshop; }
    bool can_spawn() const {return _units_owned <= _level;}

    // Income: 1★ per level + 1★ if capital, 0 if sieged
    int stars_per_turn() const;

    // Unit capacity: level + 1
    int  unit_capacity() const { return _level + 1; }

    // Tracks how many alive units this city has trained.
    int  units_owned() const { return _units_owned; }
    void add_unit();      // +1 — call when this city trains a unit
    void remove_unit();   // -1 — call when one of its units dies

    // Border radius: 1 (3x3) for levels 1-3, 2 (5x5) for level 4+
    int border_radius() const { return _border_radius; }
    void update_border(int new_size);
    void add_population(int n);
    void try_levelup();   // apply one level-up if population is sufficient; sets pending_upgrade
    // Caller must also set tile terrain to Field if it was Village
    void capture(int new_owner);
    void set_walls(bool hasWalls);
    void set_workshop(bool value);
    void set_owner(int owner)      { _owner = owner; }
    void set_tile(int tile_index)  { _tile_index = tile_index; }
    void set_capital(bool capital) { _is_capital = capital; }
    void set_sieged(bool sieged)             { _is_sieged = sieged; }
    void set_capture_ready(bool ready)       { _capture_ready = ready; }
    void clear_pending_upgrade()             { _pending_upgrade = false; }
    void add_park()                          { _parks++;}
private:
    int  _owner      = -1;
    int  _level      = 1;
    int  _population = 0;
    int  _tile_index = -1;
    int  _border_radius = 1;
    int  _units_owned = 0;
    int  _parks = 0;
    bool _is_capital        = false;
    bool _is_sieged         = false;
    bool _capture_ready     = false;
    bool _has_walls         = false;
    bool _pending_upgrade   = false;
    bool _has_workshop      = false;
};
