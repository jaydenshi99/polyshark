// Polyshark bot SDK (protocol v1) — see lab/PROTOCOL.md.
//
// A bot is a class deriving psb::Bot. POLYSHARK_BOT(MyBot) turns it into:
//   * a protocol executable (newline-delimited JSON on stdin/stdout), used by
//     Match Lab games and the Position Analyser;
//   * an in-process plugin (polyshark_bot_create), used by the native self-play
//     runner with no serialisation at all.
// The same act() code runs in both, against the same fog-of-war view.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <nlohmann/json.hpp>

#include "game_state.h"

namespace psb {

using json = nlohmann::json;

constexpr int PROTOCOL_VERSION = 1;
// Bumped whenever Bot/Observation/Decision change layout (in-process plugins).
constexpr int ABI_VERSION = 1;

// ------------------------------------------------------------- action codec --

inline const char* action_name(ActionType t) {
    switch (t) {
        case ActionType::Move:              return "Move";
        case ActionType::Attack:            return "Attack";
        case ActionType::TrainUnit:         return "TrainUnit";
        case ActionType::ConstructBuilding: return "ConstructBuilding";
        case ActionType::ResearchTech:      return "ResearchTech";
        case ActionType::CaptureCity:       return "CaptureCity";
        case ActionType::HarvestResource:   return "HarvestResource";
        case ActionType::UpgradeCity:       return "UpgradeCity";
        case ActionType::DebugAddPop:       return "DebugAddPop";
        case ActionType::Recover:           return "Recover";
        case ActionType::EndTurn:           return "EndTurn";
    }
    return "?";
}

inline bool action_from_name(const std::string& s, ActionType& out) {
    static const ActionType all[] = {
        ActionType::Move, ActionType::Attack, ActionType::TrainUnit,
        ActionType::ConstructBuilding, ActionType::ResearchTech, ActionType::CaptureCity,
        ActionType::HarvestResource, ActionType::UpgradeCity, ActionType::DebugAddPop,
        ActionType::Recover, ActionType::EndTurn,
    };
    for (ActionType t : all)
        if (s == action_name(t)) { out = t; return true; }
    return false;
}

// Same order as the engine's MOVE_DX/MOVE_DY: N, NE, E, SE, S, SW, W, NW.
constexpr int DIR_DX[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };
constexpr int DIR_DY[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };

inline std::vector<int> path_tiles(const Action& a, int map_size) {
    std::vector<int> tiles{ a.from };
    int cur = a.from;
    for (int i = 0; i < a.path_steps; i++) {
        int d = (int)((a.path_bits >> (3 * i)) & 7u);
        int x = cur % map_size + DIR_DX[d], y = cur / map_size + DIR_DY[d];
        cur = y * map_size + x;
        tiles.push_back(cur);
    }
    return tiles;
}

inline json action_to_json(const Action& a, int map_size) {
    json j = { { "type", action_name(a.type) }, { "from", a.from }, { "to", a.to }, { "param", a.param } };
    if (a.type == ActionType::Move) j["path"] = path_tiles(a, map_size);
    return j;
}

// Throws std::runtime_error on malformed input.
inline Action action_from_json(const json& j, int map_size) {
    Action a{};
    if (!j.is_object()) throw std::runtime_error("action must be an object");
    std::string name = j.at("type").get<std::string>();
    if (!action_from_name(name, a.type)) throw std::runtime_error("unknown action type " + name);
    a.from = j.value("from", -1);
    a.to = j.value("to", -1);
    a.param = j.value("param", 0);
    a.affordable = true;
    a.path_bits = 0;
    a.path_steps = 0;
    if (j.contains("path")) {
        const auto& p = j["path"];
        for (size_t i = 1; i < p.size() && i <= MAX_MOVE_PATH_STEPS; i++) {
            int prev = p[i - 1].get<int>(), cur = p[i].get<int>();
            int dx = cur % map_size - prev % map_size, dy = cur / map_size - prev / map_size;
            int dir = -1;
            for (int d = 0; d < 8; d++)
                if (DIR_DX[d] == dx && DIR_DY[d] == dy) dir = d;
            if (dir < 0) throw std::runtime_error("path steps must be adjacent tiles");
            a.path_bits |= (uint32_t)dir << (3 * (i - 1));
            a.path_steps = (uint8_t)i;
        }
    }
    return a;
}

// Moves are identified by (from, to); everything else also by param.
inline bool same_action(const Action& a, const Action& b) {
    if (a.type != b.type || a.from != b.from || a.to != b.to) return false;
    return a.type == ActionType::Move || a.param == b.param;
}

// ------------------------------------------------------------------ bot API --

struct Hello {
    int         player = 0;
    uint64_t    seed   = 0;         // the only randomness source a bot should use
    std::string mode   = "match";   // "match" | "selfplay" | "analysis"
    json        match;              // start, turn_limit, observation, timeouts, budget
    json        args;               // free-form bot arguments chosen in the lab
    json        rules;              // engine rule tables (PROTOCOL.md §Rules)
};

struct Observation {
    const GameState* state = nullptr;  // the player's view: fog applied unless observation is "full"
    const Action*    legal = nullptr;  // playable actions — authoritative, choose one by index
    int              legal_count = 0;
    int              player = 0;
    int              turn   = 0;       // engine turn counter (increments when player 0 moves next)
    int              step   = 0;       // actions applied so far in this game
    const json*      raw    = nullptr; // the original protocol message (protocol mode only)

    const GameState& view() const { return *state; }
    int find(ActionType t) const {
        for (int i = 0; i < legal_count; i++)
            if (legal[i].type == t) return i;
        return -1;
    }
};

struct Budget {
    int64_t nodes = 0;  // 0 = unspecified
    int64_t ms    = 0;
};

struct Decision {
    int   index = 0;
    float eval  = NAN;  // optional: bot's own evaluation for itself, in [-1, 1]
    json  info;         // optional small diagnostics recorded in the replay
};

struct MoveScore {
    int         index  = 0;
    float       score  = NAN;
    int64_t     visits = -1;
    std::string note;
};

struct Analysis {
    float                  eval = NAN;
    std::vector<MoveScore> moves;  // any subset of the legal list, any order
    json                   pv;     // optional array of action objects (first is from the legal list)
    json                   info;
};

class Bot {
public:
    virtual ~Bot() = default;
    virtual void     init(const Hello&) {}
    virtual Decision act(const Observation& obs) = 0;
    // Optional. Return false when unsupported; the lab then shows act()'s choice.
    virtual bool     analyse(const Observation&, const Budget&, Analysis&) { return false; }
    virtual void     end(const json& /*result*/) {}
};

// --------------------------------------------------------- protocol driver --

namespace detail {

inline void send(const json& j) {
    std::string line = j.dump();
    line.push_back('\n');
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fflush(stdout);
}

struct ParsedObs {
    GameState           state;
    std::vector<Action> legal;
    Observation         obs;
};

inline void parse_obs(const json& msg, int player, ParsedObs& out) {
    const json& o = msg.at("obs");
    out.state = GameState::deserialise(o);
    int msz = out.state.map_size();
    out.legal.clear();
    for (const auto& a : msg.at("legal")) out.legal.push_back(action_from_json(a, msz));
    out.obs = Observation{};
    out.obs.state = &out.state;
    out.obs.legal = out.legal.data();
    out.obs.legal_count = (int)out.legal.size();
    out.obs.player = player;
    out.obs.turn = msg.value("turn", out.state.get_turn());
    out.obs.step = msg.value("step", 0);
    out.obs.raw = &msg;
}

inline json budget_json(const json& msg, Budget& b) {
    json bj = msg.value("budget", json::object());
    b.nodes = bj.value("nodes", (int64_t)0);
    b.ms = bj.value("ms", (int64_t)0);
    return bj;
}

}  // namespace detail

// Runs the stdin/stdout protocol until "end"/"quit" or EOF. Returns the exit code.
template <class B>
int run(B& bot, const char* name, const char* version = "0") {
    static_assert(std::is_base_of<Bot, B>::value, "bots derive psb::Bot");
    // Advertise analyse only when the bot overrides it.
    constexpr bool can_analyse = !std::is_same<decltype(&B::analyse), decltype(&Bot::analyse)>::value;
    std::ios::sync_with_stdio(false);
    // ParsedObs holds a full GameState (~100 KB); keep it off the stack.
    auto parsed = std::make_unique<detail::ParsedObs>();
    int player = 0;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        json msg;
        try {
            msg = json::parse(line);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[psb] bad json from lab: %s\n", e.what());
            continue;
        }
        std::string type = msg.value("type", "");
        try {
            if (type == "hello") {
                Hello h;
                h.player = msg.value("player", 0);
                h.seed = msg.value("seed", (uint64_t)0);
                h.mode = msg.value("mode", "match");
                h.match = msg.value("match", json::object());
                h.args = msg.value("args", json::object());
                h.rules = msg.value("rules", json::object());
                player = h.player;
                bot.init(h);
                json caps = json::array();
                if (can_analyse) caps.push_back("analyse");
                detail::send({ { "type", "ready" }, { "name", name }, { "version", version },
                               { "protocol", PROTOCOL_VERSION }, { "capabilities", caps } });
            } else if (type == "act") {
                detail::parse_obs(msg, msg.value("player", player), *parsed);
                Decision d = bot.act(parsed->obs);
                json reply = { { "type", "action" }, { "id", msg.value("id", 0) }, { "index", d.index } };
                if (!std::isnan(d.eval)) reply["eval"] = d.eval;
                if (!d.info.is_null()) reply["info"] = d.info;
                detail::send(reply);
            } else if (type == "analyse") {
                detail::parse_obs(msg, msg.value("player", player), *parsed);
                Budget b;
                detail::budget_json(msg, b);
                Analysis an;
                json reply = { { "type", "analysis" }, { "id", msg.value("id", 0) } };
                if (bot.analyse(parsed->obs, b, an)) {
                    if (!std::isnan(an.eval)) reply["eval"] = an.eval;
                    json moves = json::array();
                    for (const auto& m : an.moves) {
                        json mj = { { "index", m.index } };
                        if (!std::isnan(m.score)) mj["score"] = m.score;
                        if (m.visits >= 0) mj["visits"] = m.visits;
                        if (!m.note.empty()) mj["note"] = m.note;
                        moves.push_back(mj);
                    }
                    reply["moves"] = moves;
                    if (!an.pv.is_null()) reply["pv"] = an.pv;
                    if (!an.info.is_null()) reply["info"] = an.info;
                } else {
                    reply["unsupported"] = true;
                }
                detail::send(reply);
            } else if (type == "end") {
                bot.end(msg);
                return 0;
            } else if (type == "quit") {
                return 0;
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[psb] %s failed: %s\n", type.c_str(), e.what());
            detail::send({ { "type", "error" }, { "id", msg.value("id", 0) }, { "message", e.what() } });
        }
    }
    return 0;
}

}  // namespace psb

#ifndef POLYSHARK_ENGINE_HASH
#define POLYSHARK_ENGINE_HASH "dev"
#endif

#define POLYSHARK_EXPORT extern "C" __attribute__((visibility("default")))

#ifdef POLYSHARK_NO_MAIN
#define POLYSHARK_BOT_MAIN(BotClass, Name, Version)
#else
#define POLYSHARK_BOT_MAIN(BotClass, Name, Version) \
    int main() { BotClass bot; return psb::run(bot, Name, Version); }
#endif

// Exposes BotClass as both a protocol executable and an in-process plugin.
#define POLYSHARK_BOT(BotClass, Name, Version)                                            \
    POLYSHARK_EXPORT psb::Bot*   polyshark_bot_create() { return new BotClass(); }        \
    POLYSHARK_EXPORT int         polyshark_bot_abi() { return psb::ABI_VERSION; }         \
    POLYSHARK_EXPORT const char* polyshark_bot_engine() { return POLYSHARK_ENGINE_HASH; } \
    POLYSHARK_EXPORT const char* polyshark_bot_name() { return Name; }                    \
    POLYSHARK_BOT_MAIN(BotClass, Name, Version)
