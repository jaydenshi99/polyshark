// Shared Match Lab runtime: game setup, fog-of-war views, action codec and
// outcome rules used identically by the referee, the tool and self-play.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "game_state.h"
#include "mapgen.h"
#include "polyshark_bot.h"

#ifndef POLYSHARK_ENGINE_HASH
#define POLYSHARK_ENGINE_HASH "dev"
#endif

// Direct member access for view construction (declared friend in game_state.h).
struct LabAccess {
    static Tile*   tiles(GameState& s)   { return s.map; }
    static Unit*   units(GameState& s)   { return s.units; }
    static City*   cities(GameState& s)  { return s.cities; }
    static Player* players(GameState& s) { return s.players; }
    static const Tile*   tiles(const GameState& s)   { return s.map; }
    static const Unit*   units(const GameState& s)   { return s.units; }
    static const City*   cities(const GameState& s)  { return s.cities; }
    static const Player* players(const GameState& s) { return s.players; }
    static int& unit_count(GameState& s)  { return s.unit_count; }
    static int& city_count(GameState& s)  { return s.city_count; }
    static int  unit_count(const GameState& s) { return s.unit_count; }
    static int  city_count(const GameState& s) { return s.city_count; }
    static int* capitals(GameState& s)    { return s.capital_city; }
    static const int* capitals(const GameState& s) { return s.capital_city; }
    static void copy_header(const GameState& src, GameState& dst) {
        dst.turn = src.turn;
        dst.cur_player = src.cur_player;
        dst._map_size = src._map_size;
    }
};

namespace lab {

using json = nlohmann::json;

// Generous upper bound on the engine's unchecked legal_actions() output.
constexpr int LEGAL_CAP = 1 << 16;

uint64_t mix64(uint64_t x);
// Stable per-purpose seed: same inputs → same seed on every machine.
uint64_t derive_seed(uint64_t base, uint64_t a, uint64_t b = 0);

// {"kind":"generated","generator":"drylands","size":N,"seed":S} or {"kind":"state","state":{...}}.
// Throws std::runtime_error when the spec is invalid.
void make_start(const json& start, GameState& out);

// Playable = affordable and not a debug action. These are the only actions bots may pick.
inline bool playable(const Action& a) {
    return a.affordable && a.type != ActionType::DebugAddPop;
}
// Engine legal actions filtered to playable ones, in engine order.
void playable_actions(const GameState& s, std::vector<Action>& out);

// Mirror of GameState::apply_action without the full-state copy.
void apply_inplace(GameState& s, const Action& a);

// Fog of war: the observer's view. Unexplored tiles become terrain None, units and
// cities on them are dropped, cities are renumbered (positional ids) and the
// opponent's explored set is replaced by the observer's own (it is not observable).
struct ViewMap {
    int city_to_view[MAX_MAP_TILES];
    int city_count = 0;
};
void   build_view(const GameState& s, int player, GameState& out, ViewMap& vm);
Action action_to_view(const Action& a, const ViewMap& vm);

// Protocol observation: serialised view plus "player", "observation", per-tile
// "territory" (border owner) and "fog" markers. `scratch` holds the view state.
json observation_json(const GameState& s, int player, bool fog, GameState& scratch, ViewMap& vm);
json actions_json(const std::vector<Action>& actions, int map_size);

json        rules_json();
std::string state_hash(const GameState& s);
std::string describe_action(const GameState& s, const Action& a);
std::string tile_label(int tile, int map_size);
int         income(const GameState& s, int player);
json        player_summary(const GameState& s, int player);

struct Outcome {
    bool        over   = false;
    int         winner = -1;   // -1 = draw / not over
    std::string reason;        // "capital" | "turn_limit"
};
// Win on capital capture; draw once the turn counter has advanced turn_limit times.
Outcome outcome(const GameState& s, int start_turn, int turn_limit);

}  // namespace lab
