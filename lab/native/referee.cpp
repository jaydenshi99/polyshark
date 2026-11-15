// polyshark-referee: plays one game between two protocol bots.
//
//   polyshark-referee game.json        (spec on the command line or "-" for stdin)
//
// Writes the result JSON to spec.result (and stdout) and the replay to spec.replay.
// See lab/PROTOCOL.md for the messages and lab/README.md for the spec fields.
#include <signal.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#include "lab_core.h"
#include "process.h"

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {

struct Seat {
    lab::BotProcess proc;
    std::string     name;
    json            spec;
    int             decisions = 0;
    double          total_ms = 0, max_ms = 0;
    std::string     version;
};

struct Forfeit {
    int         player = -1;
    std::string kind;     // timeout | crash | illegal | protocol | init | action_limit
    std::string message;
};

double elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

json read_spec(const char* path) {
    if (std::string(path) == "-") return json::parse(std::cin);
    std::ifstream f(path);
    if (!f) throw std::runtime_error(std::string("cannot open ") + path);
    return json::parse(f);
}

void write_json(const std::string& path, const json& j, int indent = -1) {
    if (path.empty()) return;
    std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp);
        f << j.dump(indent) << "\n";
    }
    std::rename(tmp.c_str(), path.c_str());
}

std::string read_kind(lab::BotProcess::Read r) {
    return r == lab::BotProcess::Read::Timeout ? "timeout" : "crash";
}

}  // namespace

void on_terminate(int) { lab::g_terminate = 1; }

int main(int argc, char** argv) {
    signal(SIGPIPE, SIG_IGN);
    signal(SIGTERM, on_terminate);
    signal(SIGINT, on_terminate);
    if (argc < 2) {
        std::cerr << "usage: polyshark-referee <game.json | ->\n";
        return 2;
    }
    json spec;
    try {
        spec = read_spec(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "bad spec: " << e.what() << "\n";
        return 2;
    }

    const auto wall0 = Clock::now();
    const json start_spec = spec.at("start");
    const int turn_limit = spec.value("turn_limit", 0);
    const bool fog = spec.value("observation", "fog") != "full";
    const json timeouts = spec.value("timeouts", json::object());
    const int init_ms = timeouts.value("init_ms", 10000);
    const int move_ms = timeouts.value("move_ms", 10000);
    const int max_per_turn = spec.value("max_actions_per_turn", 2000);
    const size_t log_cap = spec.value("log_cap_bytes", (size_t)1 << 20);
    const size_t info_cap = spec.value("info_cap_bytes", (size_t)4096);
    const json logs = spec.value("logs", json::array());

    json result = { { "status", "error" }, { "engine", POLYSHARK_ENGINE_HASH }, { "game_id", spec.value("game_id", "") } };
    auto finish = [&](int code) {
        result["wall_ms"] = elapsed_ms(wall0);
        write_json(spec.value("result", ""), result, 2);
        std::cout << result.dump() << std::endl;
        return code;
    };

    auto state = std::make_unique<GameState>();
    auto scratch = std::make_unique<GameState>();
    auto vm = std::make_unique<lab::ViewMap>();
    try {
        lab::make_start(start_spec, *state);
    } catch (const std::exception& e) {
        result["error"] = std::string("start: ") + e.what();
        return finish(1);
    }
    const int start_turn = state->get_turn();
    const std::string start_hash = lab::state_hash(*state);
    const int msz = state->map_size();

    // Bots learn the map size, never the generator seed or the hidden state.
    json public_start = { { "kind", start_spec.value("kind", "generated") }, { "size", msz } };
    json match_info = { { "start", public_start }, { "turn_limit", turn_limit },
                        { "observation", fog ? "fog" : "full" },
                        { "timeouts", { { "init_ms", init_ms }, { "move_ms", move_ms } } },
                        { "budget", spec.value("budget", json::object()) },
                        { "start_turn", start_turn }, { "first_player", state->current_player() } };
    const json rules = lab::rules_json();

    Seat seats[2];
    Forfeit forfeit;
    const json& players = spec.at("players");
    for (int p = 0; p < 2; p++) {
        Seat& s = seats[p];
        s.spec = players.at(p);
        s.name = s.spec.value("name", "bot" + std::to_string(p));
        std::vector<std::string> cmd = s.spec.at("cmd").get<std::vector<std::string>>();
        std::map<std::string, std::string> env = s.spec.value("env", std::map<std::string, std::string>());
        std::string err;
        std::string log_path = p < (int)logs.size() ? logs[p].get<std::string>() : "";
        if (!s.proc.start(cmd, s.spec.value("cwd", ""), env, log_path, log_cap, err) && forfeit.player < 0)
            forfeit = { p, "init", "could not start: " + err };
    }

    // Handshake, seat 0 first.
    for (int p = 0; p < 2 && forfeit.player < 0; p++) {
        json hello = { { "type", "hello" }, { "protocol", psb::PROTOCOL_VERSION }, { "player", p },
                       { "seed", seats[p].spec.value("seed", (uint64_t)1) }, { "mode", "match" },
                       { "match", match_info }, { "args", seats[p].spec.value("args", json::object()) },
                       { "rules", rules } };
        std::string reply;
        auto r = seats[p].proc.exchange(hello.dump(), init_ms, reply);
        if (r != lab::BotProcess::Read::Line) {
            forfeit = { p, "init", r == lab::BotProcess::Read::Timeout ? "no ready reply in time" : "exited during init" };
            break;
        }
        try {
            json rj = json::parse(reply);
            if (rj.value("type", "") != "ready") throw std::runtime_error("expected ready, got " + rj.value("type", "?"));
            seats[p].version = rj.value("version", "");
            if (rj.contains("name") && rj["name"].is_string()) result["bot_names"][p] = rj["name"];
        } catch (const std::exception& e) {
            forfeit = { p, "init", e.what() };
        }
    }

    json actions = json::array();
    json decisions = json::array();
    std::vector<Action> legal, legal_view;
    lab::Outcome out;
    int msg_id = 0, actions_this_turn = 0, last_turn_player = -1, last_turn = -1;

    while (forfeit.player < 0) {
        out = lab::outcome(*state, start_turn, turn_limit);
        if (out.over) break;
        const int p = state->current_player();
        if (p != last_turn_player || state->get_turn() != last_turn) {
            last_turn_player = p;
            last_turn = state->get_turn();
            actions_this_turn = 0;
        }
        lab::playable_actions(*state, legal);
        if (legal.empty()) {
            result["error"] = "engine produced no playable action";
            return finish(1);
        }
        legal_view.clear();
        json obs = lab::observation_json(*state, p, fog, *scratch, *vm);
        if (fog)
            for (const Action& a : legal) legal_view.push_back(lab::action_to_view(a, *vm));
        else
            legal_view = legal;

        json msg = { { "type", "act" }, { "id", ++msg_id }, { "player", p }, { "turn", state->get_turn() },
                     { "step", (int)actions.size() }, { "obs", std::move(obs) },
                     { "legal", lab::actions_json(legal_view, msz) } };
        Seat& seat = seats[p];
        auto t0 = Clock::now();
        std::string reply;
        auto r = seat.proc.exchange(msg.dump(), move_ms, reply);
        double ms = elapsed_ms(t0);
        int index = -1;
        json rj;
        // Stale replies (wrong id) are skipped; the clock keeps running.
        while (r == lab::BotProcess::Read::Line) {
            try {
                rj = json::parse(reply);
            } catch (const std::exception& e) {
                forfeit = { p, "protocol", std::string("unparseable reply: ") + e.what() };
                break;
            }
            if (rj.value("id", -1) == msg_id) break;
            int left = move_ms - (int)elapsed_ms(t0);
            r = left > 0 ? seat.proc.receive(left, reply) : lab::BotProcess::Read::Timeout;
        }
        ms = elapsed_ms(t0);
        if (forfeit.player >= 0) break;
        if (r != lab::BotProcess::Read::Line) {
            forfeit = { p, read_kind(r), r == lab::BotProcess::Read::Timeout
                                             ? "no action within " + std::to_string(move_ms) + " ms"
                                             : "bot exited" };
            break;
        }
        std::string type = rj.value("type", "");
        if (type == "error") {
            forfeit = { p, "protocol", "bot error: " + rj.value("message", "") };
            break;
        }
        if (type != "action") {
            forfeit = { p, "protocol", "expected action, got " + type };
            break;
        }
        try {
            if (rj.contains("index")) {
                index = rj["index"].get<int>();
                if (index < 0 || index >= (int)legal.size()) throw std::runtime_error("index out of range");
            } else if (rj.contains("action")) {
                Action want = psb::action_from_json(rj["action"], msz);
                for (int i = 0; i < (int)legal_view.size() && index < 0; i++)
                    if (psb::same_action(legal_view[i], want)) index = i;
                if (index < 0) throw std::runtime_error("action is not legal: " + rj["action"].dump());
            } else {
                throw std::runtime_error("reply has neither index nor action");
            }
        } catch (const std::exception& e) {
            forfeit = { p, "illegal", e.what() };
            break;
        }

        const Action& a = legal[index];
        actions.push_back({ (int)a.type, a.from, a.to, a.param, a.path_bits, a.path_steps });
        json d = { { "p", p }, { "ms", std::round(ms * 1000) / 1000 } };
        if (rj.contains("eval") && rj["eval"].is_number()) d["eval"] = rj["eval"];
        if (rj.contains("info")) {
            std::string dump = rj["info"].dump();
            d["info"] = dump.size() <= info_cap ? rj["info"] : json({ { "truncated", true }, { "bytes", dump.size() } });
        }
        decisions.push_back(std::move(d));
        seat.decisions++;
        seat.total_ms += ms;
        seat.max_ms = std::max(seat.max_ms, ms);

        lab::apply_inplace(*state, a);
        if (++actions_this_turn > max_per_turn) {
            forfeit = { p, "action_limit", "more than " + std::to_string(max_per_turn) + " actions in one turn" };
            break;
        }
    }

    if (lab::g_terminate) {
        for (int p = 0; p < 2; p++) seats[p].proc.shutdown(std::string(), 0);
        result["status"] = "cancelled";
        return finish(3);
    }

    // Result.
    result["status"] = "completed";
    if (forfeit.player >= 0) {
        result["winner"] = 1 - forfeit.player;
        result["reason"] = "forfeit";
        result["forfeit"] = { { "player", forfeit.player }, { "kind", forfeit.kind }, { "message", forfeit.message } };
    } else {
        result["winner"] = out.winner >= 0 ? json(out.winner) : json();
        result["reason"] = out.reason;
    }
    result["turns"] = state->get_turn() - start_turn;
    result["actions"] = actions.size();
    result["start_hash"] = start_hash;
    result["final_hash"] = lab::state_hash(*state);
    result["summary"] = { lab::player_summary(*state, 0), lab::player_summary(*state, 1) };
    json timing = json::array();
    for (int p = 0; p < 2; p++)
        timing.push_back({ { "decisions", seats[p].decisions }, { "total_ms", seats[p].total_ms },
                           { "mean_ms", seats[p].decisions ? seats[p].total_ms / seats[p].decisions : 0.0 },
                           { "max_ms", seats[p].max_ms }, { "version", seats[p].version } });
    result["timing"] = timing;

    json end_msg = { { "type", "end" }, { "winner", result["winner"] }, { "reason", result["reason"] },
                     { "turns", result["turns"] } };
    for (int p = 0; p < 2; p++) {
        json m = end_msg;
        m["player"] = p;
        seats[p].proc.shutdown(m.dump(), 500);
    }

    if (spec.contains("replay") && spec["replay"].is_string()) {
        json players_j = json::array();
        for (int p = 0; p < 2; p++)
            players_j.push_back({ { "name", seats[p].name }, { "bot", seats[p].spec.value("bot_id", "") },
                                  { "seed", seats[p].spec.value("seed", (uint64_t)1) },
                                  { "version", seats[p].version } });
        json replay = { { "format", "polyshark-replay" }, { "version", 1 }, { "engine", POLYSHARK_ENGINE_HASH },
                        { "game_id", spec.value("game_id", "") }, { "start", start_spec },
                        { "start_hash", start_hash }, { "turn_limit", turn_limit },
                        { "observation", fog ? "fog" : "full" }, { "players", players_j },
                        { "actions", actions }, { "decisions", decisions }, { "result", result } };
        write_json(spec["replay"].get<std::string>(), replay);
    }
    return finish(0);
}
