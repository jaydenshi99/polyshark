#pragma once

#include "grid.h"
#include "tile.h"
#include "unit.h"
#include "city.h"
#include "types.h"
#include "action.h"

constexpr int MAX_UNITS   = 121;
constexpr int MAX_CITIES  = 121;
constexpr int MAX_PLAYERS = 2;
constexpr int FOG_WORDS   = (MAP_TILES + 15) / 16;

class GameState {
public:
    bool is_terminal()    const;
    int  winner()         const;
    int  current_player() const;

    // Fog of war
    bool is_visible(int player, int tile)  const { return (visible[player][tile / 16]  >> (tile % 16)) & 1; }
    bool is_explored(int player, int tile) const { return (explored[player][tile / 16] >> (tile % 16)) & 1; }
    void set_visible(int player, int tile)        { visible[player][tile / 16]  |= (1 << (tile % 16)); }
    void set_explored(int player, int tile)       { explored[player][tile / 16] |= (1 << (tile % 16)); set_visible(player, tile); }
    void clear_visible(int player)                { for (int i = 0; i < FOG_WORDS; i++) visible[player][i] = 0; }

    // Tech unlocks
    bool has_tech(int player, TechType t) const { return (techs[player] >> static_cast<int>(t)) & 1; }
    void research_tech(int player, TechType t)  { techs[player] |= (1 << static_cast<int>(t)); }

    void print_map() const;

    friend void      legal_actions(const GameState& s, Action out[], int& out_count);
    friend GameState apply_action(GameState s, Action a);

private:
    Tile map[MAP_TILES];
    Unit units[MAX_UNITS];
    City cities[MAX_CITIES];

    uint16_t explored[MAX_PLAYERS][FOG_WORDS] = {};
    uint16_t visible[MAX_PLAYERS][FOG_WORDS]  = {};
    uint32_t techs[MAX_PLAYERS]               = {};
    int      stars[MAX_PLAYERS]               = {};

    int unit_count  = 0;
    int city_count  = 0;
    int turn        = 0;
    int cur_player  = 0;
};

void      serialize(const GameState& s, uint8_t* out_buf, int& out_size);
GameState deserialize(const uint8_t* buf, int size);
