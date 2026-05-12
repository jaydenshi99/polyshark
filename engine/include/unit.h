#pragma once

#include "types.h"

class Unit {
public:
    UnitType type()         const { return _type; }
    int      owner()        const { return _owner; }
    int      tile_index()   const { return _tile_index; }
    int      hp()           const { return _hp; }
    int      move_points()  const { return _move_points; }
    bool     has_attacked() const { return _has_attacked; }

    bool is_alive() const { return _hp > 0; }

    void take_damage(int amount);
    void spend_movement(int amount);
    void mark_attacked();

    // Resets turn state — caller provides base movement from UnitDef
    void reset_turn(int base_movement);

    // Only updates unit's record of its position.
    // Caller must also update old and new Tile::unit_id to keep them in sync.
    void set_tile(int tile_index) { _tile_index = tile_index; }

    void set_owner(int owner) { _owner = owner; }
    void set_type(UnitType type) { _type = type; }
    void set_hp(int hp) { _hp = hp; }

private:
    UnitType _type         = UnitType::None;
    int      _owner        = -1;
    int      _tile_index   = -1;
    int      _hp           = 0;
    int      _move_points  = 0;
    bool     _has_attacked = false;
};
