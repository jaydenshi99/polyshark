#pragma once

#include "grid.h"
#include "tile.h"
#include "unit.h"
#include "city.h"

constexpr int MAX_UNITS   = 121;
constexpr int MAX_CITIES  = 121;
constexpr int MAX_PLAYERS = 2;

struct Action;

class GameState {
public:
    bool is_terminal()    const;
    int  winner()         const;
    int  current_player() const;

    void print_map() const;

    // Rule engine functions — need private access
    friend void      legal_actions(const GameState& s, Action out[], int& out_count);
    friend GameState apply_action(GameState s, Action a);

private:
    Tile map[MAP_TILES];
    Unit units[MAX_UNITS];
    City cities[MAX_CITIES];

    int stars[MAX_PLAYERS] = {};
    int unit_count         = 0;
    int city_count         = 0;
    int num_players        = 2;
    int turn               = 0;
    int cur_player         = 0;
};
