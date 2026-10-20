// polyshark-selfplay: high-throughput in-process self-play.
//
//   polyshark-selfplay run.json
//
// Bots are loaded as plugins (dlopen of a POLYSHARK_BOT shared library) or are
// builtins ("random", "endturn"), so a decision costs one virtual call — no
// processes, pipes or JSON. Games run on N threads with in-place state updates
// and thread-local buffers. Progress and the final summary are JSON lines on
// stdout; per-game results go to a JSONL file and the first K games are saved
// as replays in the same format as Match Lab games.
#include <dlfcn.h>
#include <pthread.h>
#include <signal.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "lab_core.h"

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {

std::atomic<bool> g_stop{ false };
void on_signal(int) { g_stop = true; }

// ------------------------------------------------------------------ builtins --

uint64_t xorshift(uint64_t& s) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return s;
}

class RandomBot : public psb::Bot {
    uint64_t rng_ = 1;
public:
    void init(const psb::Hello& h) override { rng_ = h.seed ? h.seed : 1; }
    psb::Decision act(const psb::Observation& obs) override {
        psb::Decision d;
        d.index = (int)(xorshift(rng_) % (uint64_t)obs.legal_count);
        return d;
    }
};

class EndTurnBot : public psb::Bot {
public:
    psb::Decision act(const psb::Observation& obs) override {
        psb::Decision d;
        int e = obs.find(ActionType::EndTurn);
        d.index = e >= 0 ? e : 0;
        return d;
    }
};

struct Factory {
    std::string name;
    std::string builtin;
    json        args;
    bool        wants_view = true;
    psb::Bot* (*create)() = nullptr;

    psb::Bot* make() const {
        if (builtin == "random") return new RandomBot();
        if (builtin == "endturn") return new EndTurnBot();
        return create();
    }
};

bool load_factory(const json& j, Factory& f, std::string& err) {
    f.name = j.value("name", "bot");
    f.args = j.value("args", json::object());
    if (j.contains("builtin")) {
        f.builtin = j["builtin"].get<std::string>();
        f.wants_view = false;
        if (f.builtin != "random" && f.builtin != "endturn") { err = "unknown builtin " + f.builtin; return false; }
        return true;
    }
    std::string lib = j.value("lib", "");
    void* h = dlopen(lib.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) { err = std::string("dlopen failed: ") + dlerror(); return false; }
    auto abi = (int (*)())dlsym(h, "polyshark_bot_abi");
    auto engine = (const char* (*)())dlsym(h, "polyshark_bot_engine");
    f.create = (psb::Bot * (*)()) dlsym(h, "polyshark_bot_create");
    if (!abi || !f.create) { err = lib + " is not a POLYSHARK_BOT plugin"; return false; }
    if (abi() != psb::ABI_VERSION) { err = lib + " was built for another bot ABI"; return false; }
    std::string theirs = engine ? engine() : "?";
    if (theirs != POLYSHARK_ENGINE_HASH) { err = lib + " was built against engine " + theirs; return false; }
    return true;
}

// ------------------------------------------------------------------- stats --

struct Stats {
    uint64_t games = 0, decisions = 0, turns = 0, draws = 0, forfeits = 0;
    uint64_t wins[2] = {}, seat_wins[2] = {}, bot_decisions[2] = {};
    double   bot_ms[2] = {};
    std::map<std::string, uint64_t> reasons;
    std::vector<uint64_t> turn_hist;

    void merge(const Stats& o) {
        games += o.games; decisions += o.decisions; turns += o.turns; draws += o.draws; forfeits += o.forfeits;
        for (int i = 0; i < 2; i++) {
            wins[i] += o.wins[i]; seat_wins[i] += o.seat_wins[i];
            bot_decisions[i] += o.bot_decisions[i]; bot_ms[i] += o.bot_ms[i];
        }
        for (const auto& kv : o.reasons) reasons[kv.first] += kv.second;
        if (turn_hist.size() < o.turn_hist.size()) turn_hist.resize(o.turn_hist.size());
        for (size_t i = 0; i < o.turn_hist.size(); i++) turn_hist[i] += o.turn_hist[i];
    }

    json to_json(double secs) const {
        json ms = json::array();
        for (int i = 0; i < 2; i++) ms.push_back(bot_decisions[i] ? bot_ms[i] / bot_decisions[i] : 0.0);
        return { { "done", games }, { "decisions", decisions }, { "elapsed_s", secs },
                 { "games_per_s", secs > 0 ? games / secs : 0.0 },
                 { "decisions_per_s", secs > 0 ? decisions / secs : 0.0 },
                 { "wins", { wins[0], wins[1] } }, { "draws", draws }, { "forfeits", forfeits },
                 { "seat_wins", { seat_wins[0], seat_wins[1] } },
                 { "turns_mean", games ? (double)turns / games : 0.0 },
                 { "bot_ms_per_decision", ms }, { "reasons", reasons }, { "turn_hist", turn_hist } };
    }
};

// ------------------------------------------------------------------ runner --

struct Run {
    json                  spec;
    std::vector<Factory>  bots;
    json                  rules;
    int                   games = 0;
    uint64_t              seed = 1;
    int                   turn_limit = 0;
    bool                  fog = true;
    bool                  alternate = true;
    bool                  fixed_map = false;
    int                   max_per_turn = 2000;
    int                   replay_first = 0;
    std::string           replay_dir;
    std::ofstream         results;
    std::mutex            mu;
    Stats                 total;
    std::atomic<int>      next{ 0 };
    std::atomic<int>      running{ 0 };
    std::string           fatal;
};

struct ThreadBuffers {
    std::unique_ptr<GameState>    state = std::make_unique<GameState>();
    std::unique_ptr<GameState>    view = std::make_unique<GameState>();
    std::unique_ptr<lab::ViewMap> vm = std::make_unique<lab::ViewMap>();
    std::vector<Action>           legal, legal_view;
};

void play_game(Run& run, int g, ThreadBuffers& buf, Stats& stats) {
    const int bot_of_seat[2] = { (run.alternate && (g & 1)) ? 1 : 0, (run.alternate && (g & 1)) ? 0 : 1 };
    json start = run.spec.at("start");
    uint64_t map_seed = 0;
    if (start.value("kind", "generated") == "generated") {
        // Side-swapped pairs share a map so each bot plays both seats on it.
        map_seed = run.fixed_map ? start.value("seed", (uint64_t)1)
                                 : lab::derive_seed(run.seed, (uint64_t)(run.alternate ? g / 2 : g), 0x5eed);
        start["seed"] = map_seed;
    }
    GameState& s = *buf.state;
    lab::make_start(start, s);
    const int start_turn = s.get_turn();
    const int msz = s.map_size();
    const bool record = g < run.replay_first;

    json public_start = { { "kind", start.value("kind", "generated") }, { "size", msz } };
    json match = { { "start", public_start }, { "turn_limit", run.turn_limit }, { "observation", run.fog ? "fog" : "full" },
                   { "budget", run.spec.value("budget", json::object()) }, { "start_turn", start_turn },
                   { "first_player", s.current_player() } };

    std::unique_ptr<psb::Bot> seat[2];
    uint64_t bot_seed[2];
    int forfeit_seat = -1;
    std::string forfeit_kind, forfeit_msg;
    for (int p = 0; p < 2; p++) {
        const Factory& f = run.bots[bot_of_seat[p]];
        seat[p].reset(f.make());
        psb::Hello h;
        h.player = p;
        h.seed = bot_seed[p] = lab::derive_seed(run.seed, (uint64_t)g, (uint64_t)p + 1);
        h.mode = "selfplay";
        h.match = match;
        h.args = f.args;
        h.rules = run.rules;
        try {
            seat[p]->init(h);
        } catch (const std::exception& e) {
            if (forfeit_seat < 0) { forfeit_seat = p; forfeit_kind = "init"; forfeit_msg = e.what(); }
        }
    }

    json actions = record ? json::array() : json();
    json decisions = record ? json::array() : json();
    double ms_seat[2] = {};
    int dec_seat[2] = {};
    lab::Outcome out;
    int actions_this_turn = 0, last_player = -1, last_turn = -1, steps = 0;

    while (forfeit_seat < 0) {
        out = lab::outcome(s, start_turn, run.turn_limit);
        if (out.over) break;
        const int p = s.current_player();
        if (p != last_player || s.get_turn() != last_turn) { last_player = p; last_turn = s.get_turn(); actions_this_turn = 0; }
        lab::playable_actions(s, buf.legal);
        const Factory& f = run.bots[bot_of_seat[p]];
        psb::Observation obs;
        obs.player = p;
        obs.turn = s.get_turn();
        obs.step = steps;
        obs.legal_count = (int)buf.legal.size();
        if (run.fog && f.wants_view) {
            lab::build_view(s, p, *buf.view, *buf.vm);
            buf.legal_view.clear();
            for (const Action& a : buf.legal) buf.legal_view.push_back(lab::action_to_view(a, *buf.vm));
            obs.state = buf.view.get();
            obs.legal = buf.legal_view.data();
        } else {
            obs.state = &s;  // full observation (or a builtin that ignores the board)
            obs.legal = buf.legal.data();
        }
        auto t0 = Clock::now();
        psb::Decision d;
        try {
            d = seat[p]->act(obs);
        } catch (const std::exception& e) {
            forfeit_seat = p; forfeit_kind = "crash"; forfeit_msg = e.what();
            break;
        }
        double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        ms_seat[p] += ms;
        dec_seat[p]++;
        if (d.index < 0 || d.index >= obs.legal_count) {
            forfeit_seat = p; forfeit_kind = "illegal"; forfeit_msg = "index out of range";
            break;
        }
        const Action& a = buf.legal[d.index];
        if (record) {
            actions.push_back({ (int)a.type, a.from, a.to, a.param, a.path_bits, a.path_steps });
            json dj = { { "p", p }, { "ms", std::round(ms * 1000) / 1000 } };
            if (!std::isnan(d.eval)) dj["eval"] = d.eval;
            if (!d.info.is_null()) dj["info"] = d.info;
            decisions.push_back(std::move(dj));
        }
        lab::apply_inplace(s, a);
        steps++;
        if (++actions_this_turn > run.max_per_turn) {
            forfeit_seat = p; forfeit_kind = "action_limit"; forfeit_msg = "too many actions in one turn";
            break;
        }
    }

    int winner_seat = forfeit_seat >= 0 ? 1 - forfeit_seat : out.winner;
    std::string reason = forfeit_seat >= 0 ? "forfeit" : out.reason;
    int turns = s.get_turn() - start_turn;
    json end = { { "type", "end" }, { "winner", winner_seat >= 0 ? json(winner_seat) : json() },
                 { "reason", reason }, { "turns", turns } };
    for (int p = 0; p < 2; p++) {
        try { seat[p]->end(end); } catch (...) {}
        seat[p].reset();
    }

    stats.games++;
    stats.decisions += (uint64_t)steps;
    stats.turns += (uint64_t)std::max(0, turns);
    stats.reasons[reason]++;
    if (forfeit_seat >= 0) stats.forfeits++;
    if (winner_seat >= 0) { stats.wins[bot_of_seat[winner_seat]]++; stats.seat_wins[winner_seat]++; }
    else stats.draws++;
    for (int p = 0; p < 2; p++) { stats.bot_ms[bot_of_seat[p]] += ms_seat[p]; stats.bot_decisions[bot_of_seat[p]] += dec_seat[p]; }
    size_t bucket = (size_t)std::max(0, turns);
    if (stats.turn_hist.size() <= bucket) stats.turn_hist.resize(bucket + 1);
    stats.turn_hist[bucket]++;

    json line = { { "game", g }, { "map_seed", map_seed }, { "seats", { bot_of_seat[0], bot_of_seat[1] } },
                  { "winner_seat", winner_seat >= 0 ? json(winner_seat) : json() },
                  { "winner_bot", winner_seat >= 0 ? json(bot_of_seat[winner_seat]) : json() },
                  { "reason", reason }, { "turns", turns }, { "actions", steps },
                  { "ms", { ms_seat[0], ms_seat[1] } } };
    if (forfeit_seat >= 0) line["forfeit"] = { { "player", forfeit_seat }, { "kind", forfeit_kind }, { "message", forfeit_msg } };

    json replay;
    if (record) {
        json result = { { "status", "completed" }, { "winner", line["winner_seat"] }, { "reason", reason },
                        { "turns", turns }, { "actions", steps }, { "final_hash", lab::state_hash(s) },
                        { "summary", { lab::player_summary(s, 0), lab::player_summary(s, 1) } } };
        if (forfeit_seat >= 0) result["forfeit"] = line["forfeit"];
        json players = json::array();
        for (int p = 0; p < 2; p++)
            players.push_back({ { "name", run.bots[bot_of_seat[p]].name }, { "bot", bot_of_seat[p] }, { "seed", bot_seed[p] } });
        replay = { { "format", "polyshark-replay" }, { "version", 1 }, { "engine", POLYSHARK_ENGINE_HASH },
                   { "game_id", "selfplay-" + std::to_string(g) }, { "start", start },
                   { "turn_limit", run.turn_limit }, { "observation", run.fog ? "fog" : "full" },
                   { "players", players }, { "actions", actions }, { "decisions", decisions }, { "result", result } };
    }

    std::lock_guard<std::mutex> lock(run.mu);
    if (run.results) run.results << line.dump() << "\n";
    if (record && !run.replay_dir.empty()) {
        std::ofstream f(run.replay_dir + "/" + std::to_string(g) + ".json");
        f << replay.dump() << "\n";
    }
}

void* worker(void* arg) {
    Run& run = *static_cast<Run*>(arg);
    ThreadBuffers buf;
    Stats local;
    run.running++;
    for (;;) {
        if (g_stop) break;
        int g = run.next.fetch_add(1);
        if (g >= run.games) break;
        try {
            play_game(run, g, buf, local);
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(run.mu);
            run.fatal = std::string("game ") + std::to_string(g) + ": " + e.what();
            g_stop = true;
        }
        // Publish every game so progress stays live; merging is cheap.
        std::lock_guard<std::mutex> lock(run.mu);
        run.total.merge(local);
        local = Stats();
    }
    run.running--;
    return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
    signal(SIGPIPE, SIG_IGN);
    signal(SIGTERM, on_signal);
    signal(SIGINT, on_signal);
    if (argc < 2) {
        std::cerr << "usage: polyshark-selfplay <run.json>\n";
        return 2;
    }
    Run run;
    try {
        std::ifstream f(argv[1]);
        run.spec = json::parse(f);
    } catch (const std::exception& e) {
        std::cout << json({ { "type", "error" }, { "error", std::string("bad spec: ") + e.what() } }).dump() << std::endl;
        return 2;
    }
    const json& spec = run.spec;
    for (const auto& b : spec.at("bots")) {
        Factory f;
        std::string err;
        if (!load_factory(b, f, err)) {
            std::cout << json({ { "type", "error" }, { "error", err } }).dump() << std::endl;
            return 2;
        }
        run.bots.push_back(f);
    }
    if (run.bots.size() == 1) run.bots.push_back(run.bots[0]);
    run.rules = lab::rules_json();
    run.games = spec.value("games", 100);
    run.seed = spec.value("seed", (uint64_t)1);
    run.turn_limit = spec.value("turn_limit", 50);
    run.fog = spec.value("observation", "fog") != "full";
    run.alternate = spec.value("alternate_sides", true);
    run.fixed_map = spec.value("fixed_map", false);
    run.max_per_turn = spec.value("max_actions_per_turn", 2000);
    json rep = spec.value("replays", json::object());
    run.replay_first = rep.value("first", 0);
    run.replay_dir = rep.value("dir", "");
    if (spec.contains("results")) run.results.open(spec["results"].get<std::string>());
    int threads = std::max(1, spec.value("threads", (int)std::thread::hardware_concurrency()));
    int progress_ms = std::max(50, spec.value("progress_ms", 500));

    // A quick validity check before spawning threads: the start must build.
    try {
        GameState* probe = new GameState();
        json start = spec.at("start");
        if (start.value("kind", "generated") == "generated" && !start.contains("seed")) start["seed"] = 1;
        lab::make_start(start, *probe);
        delete probe;
    } catch (const std::exception& e) {
        std::cout << json({ { "type", "error" }, { "error", std::string("start: ") + e.what() } }).dump() << std::endl;
        return 2;
    }

    std::cout << json({ { "type", "start" }, { "engine", POLYSHARK_ENGINE_HASH }, { "threads", threads },
                        { "games", run.games } }).dump() << std::endl;
    const auto t0 = Clock::now();
    std::vector<pthread_t> pool(threads);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 16u << 20);  // engine helpers keep large arrays on the stack
    for (auto& t : pool) pthread_create(&t, &attr, worker, &run);
    pthread_attr_destroy(&attr);

    auto secs = [&]() { return std::chrono::duration<double>(Clock::now() - t0).count(); };
    auto next_report = Clock::now() + std::chrono::milliseconds(progress_ms);
    for (;;) {
        bool finished;
        {
            std::lock_guard<std::mutex> lock(run.mu);
            finished = run.total.games >= (uint64_t)run.games;
        }
        if (finished || g_stop || (run.running.load() == 0 && run.next.load() >= run.games)) break;
        // Short naps so the run ends (and is timed) as soon as the last game does.
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        if (Clock::now() < next_report) continue;
        next_report += std::chrono::milliseconds(progress_ms);
        json p;
        {
            std::lock_guard<std::mutex> lock(run.mu);
            p = run.total.to_json(secs());
        }
        p["type"] = "progress";
        p["games"] = run.games;
        std::cout << p.dump() << std::endl;
    }
    for (auto& t : pool) pthread_join(t, nullptr);
    run.results.close();

    json done = run.total.to_json(secs());
    done["type"] = "done";
    done["games"] = run.games;
    done["stopped"] = (bool)g_stop && run.total.games < (uint64_t)run.games;
    done["threads"] = threads;
    done["engine"] = POLYSHARK_ENGINE_HASH;
    if (!run.fatal.empty()) done["error"] = run.fatal;
    std::cout << done.dump() << std::endl;
    return run.fatal.empty() ? 0 : 1;
}
