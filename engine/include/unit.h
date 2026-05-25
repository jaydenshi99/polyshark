#pragma once

#include "types.h"
#include "unit_def.h"

class Unit {
public:
    UnitType type()            const { return _type; }
    bool     has_ability(uint32_t ability) const {
        return unit_def(_type).abilities & ability;
    }
    int      owner()           const { return _owner; }
    int      tile_index()      const { return _tile_index; }
    int      hp()              const { return _hp; }
    int      max_hp()          const { return _max_hp; }
    int      move_points()     const { return _move_points; }
    bool     has_attacked()    const { return _has_attacked; }
    int      kills()           const { return _kills; }
    bool     promotion_ready() const { return _promotion_ready; }
    int      city_id()         const { return _city_id; }
    void     set_city_id(int city_id) { _city_id = city_id; }
    // Most recent direction (0..7 in MOVE_DX/MOVE_DY) the unit moved or, for
    // ranged units, attacked. -1 = never acted. Used by forced-spawn pushes.
    int      last_dir()        const { return _last_dir; }
    void     set_last_dir(int8_t d)   { _last_dir = d; }

    bool is_alive() const { return _hp > 0; }
    void take_damage(int amount);
    void heal(int amount);         // recovers HP, capped at max_hp
    void spend_movement(int amount);
    void mark_attacked();
    void add_kill();          // increments kill count, sets promotion_ready at 3
    void accept_promotion();  // +5 max hp, full heal, clears promotion_ready

    // Helper for actions that fully consume the unit's turn (e.g. CaptureCity).
    void end_turn_actions() { _move_points = 0; _has_attacked = true; }
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
    int      _city_id         = -1;  // city that trained this unit; 
    bool     _has_attacked    = false;
    bool     _promotion_ready = false;
    int8_t   _last_dir        = -1;
};
