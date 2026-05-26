#pragma once

#include "grid.h"
#include "tile.h"
#include "unit.h"
#include "city.h"
#include "types.h"
#include "player.h"
#include <vector>
#include <nlohmann/json.hpp>
#include "logger.h"

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
    Recover,          // from=unit tile — heals the unit and marks its turn spent
    EndTurn,
};

constexpr int MAX_MOVE_PATH_STEPS = 6;

struct Action {
    ActionType type;
    int from;        // tile index (where relevant)
    int to;          // tile index (where relevant)
    int param;       // unit type, tech id, improvement type, etc.
    bool affordable; // false = shown but unplayable (insufficient stars)
    // Move actions only: cheapest BFS path packed as 3-bit direction steps from
    // `from` to `to`. path_bits low-to-high — bits[3*i..3*i+2] = i-th step's
    // direction index into the 8-neighbor MOVE_DX/MOVE_DY tables. path_steps is
    // the number of hops (== tile count - 1). Other action types leave path_steps = 0.
    uint32_t path_bits;
    uint8_t  path_steps;
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
    void set_visible(int player, int tile);

    // Tech unlocks
    bool     has_tech(int player, TechType t)  const { return players[player].has_tech(t); }
    void     research_tech(int player, TechType t)   { players[player].research_tech(t); }
    std::vector<TechType> get_techs(int player) const { return players[player].get_techs(); }
    uint32_t              techs_mask(int player) const { return players[player].techs_mask(); }

    // Stars
    int  get_stars(int player) const       { return players[player].stars; }
    void set_stars(int player, int amount) { players[player].stars = amount; }

    const Player& get_player(int player) const { return players[player]; }

    // Number of cities currently owned by `player` (includes capital).
    int owned_cities(int player) const {
        int n = 0;
        for (int i = 0; i < city_count; i++)
            if (cities[i].owner() == player) n++;
        return n;
    }

    void print_map() const;

    // Mutable tile access for map setup and apply_action
    Tile& tile_at(int index) { return map[index]; }
    const Tile& tile_at(int index) const { return map[index]; }

    // Allocates a unit slot, initialises hp/movement from UnitDef, places on tile.
    // Returns slot index, or -1 if full. If `tile_index` already has a unit,
    // that occupant is force-pushed via push_unit_from (super-unit / Ruin /
    // Polytaur-style spawns; see Polytopia rules).
    int spawn_unit(UnitType type, int owner, int tile_index, int city_id = -1);

    // Forced-spawn push: the existing unit on `spawn_tile` is shoved one tile
    // along a rule-derived direction (see forced_spawn.cpp). If every fallback
    // tile is blocked, the unit is removed from the game entirely — no kill
    // credit, no attack credit.
    void push_unit_from(int unit_id, int spawn_tile, int spawning_player);

    // Relocate `unit_id` to `dest_tile`: removes from its current tile, places
    // on dest, updates tile_index, and reveals around dest (radius 2 on
    // mountain landings, 1 otherwise). Intended for single-tile hops — call
    // repeatedly to step a multi-tile path. Does NOT touch last_dir or
    // movement points; callers handle those.
    void move_unit(int unit_id, int dest_tile);

    // Allocates a city slot, places on tile.
    // Returns slot index, or -1 if full.
    int spawn_city(int tile_index, int owner, bool is_capital);
    float get_defense_bonus(Player& player, Tile& t);
    const Unit& get_unit(int id) const { return units[id]; }
    const City& get_city(int id) const { return cities[id]; }

    void      legal_actions(Action out[], int& out_count) const;
    GameState apply_action(Action a) const;

    // Per-action handlers — mutate *this in place. Called by apply_action on a copy.
    void apply_end_turn();
    void apply_research_tech(Action a);
    void apply_train_unit(Action a);
    void apply_harvest_resource(Action a);
    void apply_debug_add_pop(Action a);
    void apply_upgrade_city(Action a);
    void apply_capture_city(Action a);
    void apply_move(Action a);
    void apply_attack(Action a);
    void apply_recover(Action a);

    void explore(int tile, int player,
                 int* out_path = nullptr, int* out_count = nullptr);
    void claim_border_for_city(int city_id);

    // Sanity-checks every cross-reference in the game state. Returns true if
    // all invariants hold. On failure, prints the first violation to stderr.
    // Cheap enough to run after every applied action in tests / debug builds.
    bool check_invariants() const;

    // Debug helpers — dump GameState to JSON / log it. `static` because they're
    // utility operations, not core state transitions; call as
    // GameState::serialise(s) / GameState::print(s).
    static GameState deserialise(const nlohmann::json& j);
    static nlohmann::json serialise(const GameState& s);
    static void           print(const GameState& s);

private:
    Tile map[MAX_MAP_TILES];
    Unit units[MAX_MAP_TILES];
    City cities[MAX_MAP_TILES];

    Player players[MAX_PLAYERS];

    int unit_count       = 0;
    int city_count       = 0;
    int turn             = 0;
    int cur_player       = 0;
    int _map_size        = MAP_SIZE;
    int capital_city[MAX_PLAYERS] = {-1, -1};
};

// Movement-path helpers (defined in game_state.cpp). Exposed so the visualizer can
// reconstruct the cheapest path for animation purposes and so replays without
// embedded paths (older replay files) can still populate them at load time.
void reachable_tiles(const GameState& s, int unit_id, int8_t out_mp[]);
void reachable_tiles(const GameState& s, int unit_id, int8_t out_mp[], int out_parent[]);
bool encode_path_bits(const int parent[], int src, int dst, int msz,
                      uint32_t* out_bits, uint8_t* out_steps);
void decode_path_bits(int start, uint32_t bits, uint8_t steps, int msz,
                      int out_tiles[MAX_MOVE_PATH_STEPS + 1], int* out_count);
