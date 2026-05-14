#pragma once

#include "grid.h"
#include "tile.h"
#include "unit.h"
#include "city.h"
#include "types.h"
#include "player.h"

constexpr int MAX_PLAYERS = 2;

enum class ActionType {
    Move,
    Attack,
    TrainUnit,
    ConstructBuilding,
    ResearchTech,
    CaptureCity,
    HarvestResource,  // from=city tile, to=resource tile, param=ResourceType
    UpgradeCity,      // from=city_id, param=CityUpgradeType — must be taken before EndTurn when pending
    DebugAddPop,      // from=city_id — adds 1 population; always available when not blocked
    EndTurn,
};

struct Action {
    ActionType type;
    int from;        // tile index (where relevant)
    int to;          // tile index (where relevant)
    int param;       // unit type, tech id, improvement type, etc.
    bool affordable; // false = shown but unplayable (insufficient stars)
};

enum class GameStateType {
    UpgradingCity = 0,
    Idle,
};

class GameState {
public:
    bool is_terminal()    const;
    int  winner()         const;
    int  current_player() const;
    int  get_turn()       const { return turn; }
    int  map_size()       const { return _map_size; }
    int  map_tiles()      const { return _map_size * _map_size; }
    void set_map_size(int sz)   { _map_size = sz; }

    // Derives the current turn phase from city state — never stored redundantly.
    GameStateType phase() const;
    bool          can_harvest() const { return phase() == GameStateType::Idle; }

    // Fog of war
    bool is_visible(int player, int tile)  const { return players[player].is_visible(tile); }
    bool is_explored(int player, int tile) const { return players[player].is_explored(tile); }
    void set_visible(int player, int tile)        { players[player].set_visible(tile); }
    void set_explored(int player, int tile)       { players[player].set_explored(tile); }
    void clear_visible(int player)                { players[player].clear_visible(); }

    // Tech unlocks
    bool     has_tech(int player, TechType t)  const { return players[player].has_tech(t); }
    void     research_tech(int player, TechType t)   { players[player].research_tech(t); }
    uint32_t get_techs(int player)             const { return players[player].get_techs(); }

    // Stars
    int  get_stars(int player) const       { return players[player].stars; }
    void set_stars(int player, int amount) { players[player].stars = amount; }

    const Player& get_player(int player) const { return players[player]; }

    void print_map() const;

    // Mutable tile access for map setup and apply_action
    Tile& tile_at(int index) { return map[index]; }
    const Tile& tile_at(int index) const { return map[index]; }

    // Allocates a unit slot, initialises hp/movement from UnitDef, places on tile.
    // Returns slot index, or -1 if full.
    int spawn_unit(UnitType type, int owner, int tile_index);

    // Allocates a city slot, places on tile.
    // Returns slot index, or -1 if full.
    int spawn_city(int tile_index, int owner, bool is_capital);

    const Unit& get_unit(int id) const { return units[id]; }
    const City& get_city(int id) const { return cities[id]; }

    void      legal_actions(Action out[], int& out_count) const;
    GameState apply_action(Action a) const;

    // Claims all unclaimed tiles within city_id's border radius.
    // Already-claimed tiles are never stolen — first-claimer always wins.
    void claim_border_for_city(int city_id);

private:
    Tile map[MAX_MAP_TILES];
    Unit units[MAX_MAP_TILES];
    City cities[MAX_MAP_TILES];

    Player players[MAX_PLAYERS];

    int unit_count  = 0;
    int city_count  = 0;
    int turn        = 0;
    int cur_player  = 0;
    int _map_size   = MAP_SIZE;
};



void      serialize(const GameState& s, uint8_t* out_buf, int& out_size);
GameState deserialize(const uint8_t* buf, int size);
