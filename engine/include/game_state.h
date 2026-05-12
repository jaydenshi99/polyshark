#pragma once

#include "grid.h"
#include "types.h"

constexpr int MAX_UNITS  = 64;
constexpr int MAX_CITIES = 16;

struct Tile {
    TerrainType  terrain  = TerrainType::Flat;
    ResourceType resource = ResourceType::None;
    int unit_id           = -1;
    int city_id           = -1;
};

struct Unit {
    UnitType type;
    int  owner        = -1;
    int  hp           = 0;
    int  move_points  = 0;
    bool has_attacked = false;
};

struct City {
    int owner      = -1;
    int population = 0;
    int tile_index = -1;
};

struct Action;

class GameState {
public:
    bool is_terminal()     const;
    int  winner()          const;
    int  current_player()  const;

    void print_map() const;

    // Rule engine functions — need private access
    friend void      legal_actions(const GameState& s, Action out[], int& out_count);
    friend GameState apply_action(GameState s, Action a);

private:
    Tile map[MAP_TILES];
    Unit units[MAX_UNITS];
    City cities[MAX_CITIES];

    int unit_count   = 0;
    int city_count   = 0;
    int stars        = 0;
    int turn         = 0;
    int cur_player   = 0;
};
