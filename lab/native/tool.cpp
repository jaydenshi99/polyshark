// polyshark-tool: engine service for the Match Lab web server.
//
// Reads one JSON request per line on stdin and writes one JSON reply per line.
// All game logic the website needs (legal moves, applying moves, fog views,
// replay expansion) lives here so the server stays orchestration-only.
#include <signal.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>

#include "lab_core.h"
#include "unit_def.h"

using json = nlohmann::json;

namespace {

GameState& scratch_state(int slot) {
    static std::unique_ptr<GameState> slots[4] = {
        std::make_unique<GameState>(), std::make_unique<GameState>(),
        std::make_unique<GameState>(), std::make_unique<GameState>() };
    return *slots[slot];
}

Action action_from_array(const json& a) {
    Action x{};
    x.type = (ActionType)a.at(0).get<int>();
    x.from = a.at(1).get<int>();
    x.to = a.at(2).get<int>();
    x.param = a.at(3).get<int>();
    x.affordable = true;
    x.path_bits = a.size() > 4 ? a[4].get<uint32_t>() : 0;
    x.path_steps = a.size() > 5 ? (uint8_t)a[5].get<int>() : 0;
    return x;
}

json outcome_json(const GameState& s, const json& req) {
    lab::Outcome o = lab::outcome(s, req.value("start_turn", 0), req.value("turn_limit", 0));
    return { { "over", o.over }, { "winner", o.winner >= 0 ? json(o.winner) : json() }, { "reason", o.reason } };
}

// Attack previews: apply on a copy and read both units' hp.
json attack_preview(const GameState& s, const Action& a) {
    GameState& c = scratch_state(3);
    c = s;
    const Unit& atk0 = s.get_unit(s.tile_at(a.from).unit_id());
    const Unit& def0 = s.get_unit(s.tile_at(a.to).unit_id());
    int atk_id = s.tile_at(a.from).unit_id(), def_id = s.tile_at(a.to).unit_id();
    lab::apply_inplace(c, a);
    const Unit& atk1 = c.get_unit(atk_id);
    const Unit& def1 = c.get_unit(def_id);
    return { { "attacker_hp", { atk0.hp(), atk1.is_alive() ? atk1.hp() : 0 } },
             { "defender_hp", { def0.hp(), def1.is_alive() ? def1.hp() : 0 } },
             { "kills", !def1.is_alive() }, { "dies", !atk1.is_alive() },
             { "advances", c.tile_at(a.to).unit_id() == atk_id } };
}

json legal_list(const GameState& s, bool previews) {
    std::vector<Action> legal;
    lab::playable_actions(s, legal);
    json arr = json::array();
    for (const Action& a : legal) {
        json j = psb::action_to_json(a, s.map_size());
        j["label"] = lab::describe_action(s, a);
        if (previews && a.type == ActionType::Attack) j["preview"] = attack_preview(s, a);
        arr.push_back(std::move(j));
    }
    return arr;
}

json inspect(const GameState& s, const json& req) {
    json out = { { "ok", true } };
    if (req.value("include_state", true)) out["state"] = GameState::serialise(s);
    out["hash"] = lab::state_hash(s);
    out["turn"] = s.get_turn();
    out["player"] = s.current_player();
    out["phase"] = s.phase() == GameStateType::UpgradingCity ? "upgrading_city" : "idle";
    out["outcome"] = outcome_json(s, req);
    out["summary"] = { lab::player_summary(s, 0), lab::player_summary(s, 1) };
    if (req.value("legal", true)) out["legal"] = legal_list(s, req.value("previews", true));
    return out;
}

// Human-readable consequences of one action, for the analyser's log.
json diff_events(const GameState& a, const GameState& b) {
    static const char* names[] = { "?", "Warrior", "Archer", "Rider", "Defender", "Giant" };
    json ev = json::array();
    const int msz = a.map_size();
    const int ua = LabAccess::unit_count(a), ub = LabAccess::unit_count(b);
    for (int i = 0; i < std::max(ua, ub); i++) {
        const Unit& x = LabAccess::units(a)[i];
        const Unit& y = LabAccess::units(b)[i];
        bool alive_a = i < ua && x.is_alive(), alive_b = i < ub && y.is_alive();
        if (!alive_a && !alive_b) continue;
        const Unit& u = alive_b ? y : x;
        std::string who = std::string(names[(int)u.type()]) + " (P" + std::to_string(u.owner()) + ")";
        if (!alive_a) {
            ev.push_back({ { "kind", "spawn" }, { "text", who + " appears at " + lab::tile_label(y.tile_index(), msz) } });
            continue;
        }
        if (!alive_b) {
            ev.push_back({ { "kind", "death" }, { "text", who + " at " + lab::tile_label(x.tile_index(), msz) + " dies" } });
            continue;
        }
        if (x.tile_index() != y.tile_index())
            ev.push_back({ { "kind", "move" }, { "text", who + " " + lab::tile_label(x.tile_index(), msz) + " → " +
                                                             lab::tile_label(y.tile_index(), msz) } });
        if (x.hp() != y.hp())
            ev.push_back({ { "kind", "hp" }, { "text", who + " hp " + std::to_string(x.hp()) + " → " + std::to_string(y.hp()) } });
    }
    const int ca = LabAccess::city_count(a), cb = LabAccess::city_count(b);
    for (int i = 0; i < cb; i++) {
        const City& y = LabAccess::cities(b)[i];
        std::string at = lab::tile_label(y.tile_index(), msz);
        if (i >= ca) {
            ev.push_back({ { "kind", "city" }, { "text", "P" + std::to_string(y.owner()) + " founds a city at " + at } });
            continue;
        }
        const City& x = LabAccess::cities(a)[i];
        if (x.owner() != y.owner())
            ev.push_back({ { "kind", "capture" }, { "text", "P" + std::to_string(y.owner()) + " captures the city at " + at } });
        if (x.level() != y.level())
            ev.push_back({ { "kind", "level" }, { "text", "City " + at + " reaches level " + std::to_string(y.level()) } });
        if (x.population() != y.population() && x.level() == y.level())
            ev.push_back({ { "kind", "population" }, { "text", "City " + at + " population " + std::to_string(x.population()) +
                                                                  " → " + std::to_string(y.population()) } });
    }
    for (int p = 0; p < MAX_PLAYERS; p++) {
        int ds = b.get_stars(p) - a.get_stars(p);
        if (ds) ev.push_back({ { "kind", "stars" }, { "text", "P" + std::to_string(p) + (ds > 0 ? " +" : " ") + std::to_string(ds) + "★" } });
        int ea = 0, eb = 0;
        for (int i = 0; i < a.map_tiles(); i++) { ea += a.is_visible(p, i); eb += b.is_visible(p, i); }
        if (eb != ea) ev.push_back({ { "kind", "explore" }, { "text", "P" + std::to_string(p) + " explores " + std::to_string(eb - ea) + " tiles" } });
        uint32_t new_techs = b.techs_mask(p) & ~a.techs_mask(p);
        if (new_techs) ev.push_back({ { "kind", "tech" }, { "text", "P" + std::to_string(p) + " learns a tech" } });
    }
    if (a.current_player() != b.current_player())
        ev.push_back({ { "kind", "turn" }, { "text", "P" + std::to_string(b.current_player()) + " to move, turn " + std::to_string(b.get_turn()) } });
    return ev;
}

// Replays a game up to `step` actions into `s`. Returns false on a bad replay.
void replay_to(const json& replay, int step, GameState& s) {
    lab::make_start(replay.at("start"), s);
    const json& acts = replay.at("actions");
    int n = std::min<int>(step, (int)acts.size());
    for (int i = 0; i < n; i++) lab::apply_inplace(s, action_from_array(acts[i]));
}

json handle(const json& req) {
    const std::string cmd = req.value("cmd", "");
    if (cmd == "ping") return { { "ok", true }, { "engine", POLYSHARK_ENGINE_HASH } };
    if (cmd == "rules") return { { "ok", true }, { "rules", lab::rules_json() } };

    if (cmd == "new") {
        GameState& s = scratch_state(0);
        lab::make_start(req.at("start"), s);
        json out = inspect(s, req);
        return out;
    }
    if (cmd == "inspect") {
        GameState& s = scratch_state(0);
        s = GameState::deserialise(req.at("state"));
        return inspect(s, req);
    }
    if (cmd == "apply") {
        GameState& s = scratch_state(0);
        GameState& before = scratch_state(1);
        s = GameState::deserialise(req.at("state"));
        before = s;
        std::vector<Action> legal;
        lab::playable_actions(s, legal);
        Action want = psb::action_from_json(req.at("action"), s.map_size());
        const Action* chosen = nullptr;
        for (const Action& a : legal)
            if (psb::same_action(a, want)) { chosen = &a; break; }
        if (!chosen) return { { "ok", false }, { "error", "not a legal action here" } };
        Action a = *chosen;
        std::string label = lab::describe_action(s, a);
        lab::apply_inplace(s, a);
        json out = inspect(s, req);
        out["applied"] = psb::action_to_json(a, s.map_size());
        out["applied"]["label"] = label;
        out["applied_array"] = { (int)a.type, a.from, a.to, a.param, a.path_bits, a.path_steps };
        out["events"] = diff_events(before, s);
        return out;
    }
    if (cmd == "observe") {
        GameState& s = scratch_state(0);
        s = GameState::deserialise(req.at("state"));
        int player = req.value("player", s.current_player());
        bool fog = req.value("observation", "fog") != "full";
        auto vm = std::make_unique<lab::ViewMap>();
        json obs = lab::observation_json(s, player, fog, scratch_state(2), *vm);
        std::vector<Action> legal, view;
        if (player == s.current_player()) lab::playable_actions(s, legal);
        for (const Action& a : legal) view.push_back(fog ? lab::action_to_view(a, *vm) : a);
        json labels = json::array();
        for (const Action& a : legal) labels.push_back(lab::describe_action(s, a));
        return { { "ok", true }, { "obs", obs }, { "legal", lab::actions_json(view, s.map_size()) },
                 { "legal_real", lab::actions_json(legal, s.map_size()) }, { "labels", labels } };
    }
    if (cmd == "replay") {
        // States at steps from .. from+count-1 (state before action k = step k).
        const json& replay = req.at("replay");
        int total = (int)replay.at("actions").size();
        int from = std::max(0, req.value("from", 0));
        int count = std::max(1, req.value("count", 1));
        GameState& s = scratch_state(0);
        replay_to(replay, from, s);
        json states = json::array();
        const json& acts = replay["actions"];
        for (int k = from; k <= total && k < from + count; k++) {
            states.push_back(GameState::serialise(s));
            if (k < total) lab::apply_inplace(s, action_from_array(acts[k]));
        }
        return { { "ok", true }, { "from", from }, { "steps", total + 1 }, { "states", states } };
    }
    if (cmd == "timeline") {
        const json& replay = req.at("replay");
        GameState& s = scratch_state(0);
        lab::make_start(replay.at("start"), s);
        const json& acts = replay.at("actions");
        json labels = json::array(), movers = json::array(), turns = json::array();
        json stars = json::array(), inc = json::array(), cities = json::array(), units = json::array(), hp = json::array();
        auto sample = [&]() {
            json st = json::array(), in = json::array(), ci = json::array(), un = json::array(), h = json::array();
            for (int p = 0; p < 2; p++) {
                json sum = lab::player_summary(s, p);
                st.push_back(sum["stars"]); in.push_back(sum["income"]); ci.push_back(sum["cities"]);
                un.push_back(sum["units"]); h.push_back(sum["hp"]);
            }
            stars.push_back(st); inc.push_back(in); cities.push_back(ci); units.push_back(un); hp.push_back(h);
            turns.push_back(s.get_turn());
            movers.push_back(s.current_player());
        };
        for (const auto& aj : acts) {
            sample();
            Action a = action_from_array(aj);
            json lj = psb::action_to_json(a, s.map_size());
            lj["label"] = lab::describe_action(s, a);
            labels.push_back(lj);
            lab::apply_inplace(s, a);
        }
        sample();
        std::string final_hash = lab::state_hash(s);
        std::string expected = replay.value("result", json::object()).value("final_hash", "");
        return { { "ok", true }, { "steps", (int)acts.size() + 1 }, { "map_size", s.map_size() },
                 { "actions", labels }, { "movers", movers }, { "turns", turns },
                 { "stats", { { "stars", stars }, { "income", inc }, { "cities", cities }, { "units", units }, { "hp", hp } } },
                 { "final_hash", final_hash }, { "verified", expected.empty() ? json() : json(expected == final_hash) } };
    }
    return { { "ok", false }, { "error", "unknown cmd " + cmd } };
}

}  // namespace

int main() {
    signal(SIGPIPE, SIG_IGN);
    std::ios::sync_with_stdio(false);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        json reply;
        json req;
        try {
            req = json::parse(line);
            reply = handle(req);
        } catch (const std::exception& e) {
            reply = { { "ok", false }, { "error", e.what() } };
        }
        if (req.is_object() && req.contains("id")) reply["id"] = req["id"];
        std::cout << reply.dump() << "\n";
        std::cout.flush();
    }
    return 0;
}
