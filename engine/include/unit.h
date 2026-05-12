#pragma once

#include "types.h"

class Unit {
public:
    UnitType type()            const { return _type; }
    int      owner()           const { return _owner; }
    int      tile_index()      const { return _tile_index; }
    int      hp()              const { return _hp; }
    int      max_hp()          const { return _max_hp; }
    int      move_points()     const { return _move_points; }
    bool     has_attacked()    const { return _has_attacked; }
    int      kills()           const { return _kills; }
    bool     promotion_ready() const { return _promotion_ready; }

    bool is_alive() const { return _hp > 0; }

    void take_damage(int amount);
    void spend_movement(int amount);
    void mark_attacked();
    void add_kill();          // increments kill count, sets promotion_ready at 3
    void accept_promotion();  // +5 max hp, full heal, clears promotion_ready

    // Resets turn state — caller provides base movement from UnitDef
    void reset_turn(int base_movement);

    // Only updates unit's record of its position.
    // Caller must also update old and new Tile::unit_id to keep in sync.
    void set_tile(int tile_index) { _tile_index = tile_index; }
    void set_owner(int owner)     { _owner = owner; }
    void set_type(UnitType type)  { _type = type; }
    void set_hp(int hp)           { _hp = hp; }
    void set_max_hp(int max_hp)   { _max_hp = max_hp; }

private:
    UnitType _type            = UnitType::None;
    int      _owner           = -1;
    int      _tile_index      = -1;
    int      _hp              = 0;
    int      _max_hp          = 0;
    int      _move_points     = 0;
    int      _kills           = 0;
    bool     _has_attacked    = false;
    bool     _promotion_ready = false;
};
