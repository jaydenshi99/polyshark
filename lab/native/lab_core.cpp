#include "lab_core.h"

#include <cstdio>
#include <memory>
#include <stdexcept>

#include "building_def.h"
#include "config.h"
#include "resource_def.h"
#include "tech_def.h"
#include "unit_def.h"

namespace lab {

uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

uint64_t derive_seed(uint64_t base, uint64_t a, uint64_t b) {
    uint64_t s = mix64(mix64(mix64(base) ^ a) ^ (b * 0x632be59bd9b4e019ULL));
    // MapGen treats 0 as "use the built-in constant"; never hand it a zero.
    return s ? s : 1;
}

// ------------------------------------------------------------------ starts --

void make_start(const json& start, GameState& out) {
    std::string kind = start.value("kind", "generated");
    if (kind == "generated") {
        std::string gen = start.value("generator", "drylands");
        if (gen != "drylands") throw std::runtime_error("unknown generator " + gen);
        int size = start.value("size", (int)cfg::map::DEFAULT_SIZE);
        if (size < 5 || size > MAX_MAP_SIZE) throw std::runtime_error("map size must be 5.." + std::to_string(MAX_MAP_SIZE));
        uint64_t seed = start.value("seed", (uint64_t)1);
        if (seed == 0) throw std::runtime_error("map seed 0 is reserved");
        MapGenParams p = MapGenParams::for_biome(BiomeType::Drylands, size);
        p.seed = seed;
        auto gen_result = std::make_unique<MapGenResult>(MapGen(p).generate());
        out = gen_result->state;
        return;
    }
    if (kind == "state") {
        if (!start.contains("state")) throw std::runtime_error("state start needs a \"state\" object");
        out = GameState::deserialise(start["state"]);
        if (!out.check_invariants()) throw std::runtime_error("start state fails engine invariants");
        return;
    }
    throw std::runtime_error("unknown start kind " + kind);
}

// ----------------------------------------------------------------- actions --

void playable_actions(const GameState& s, std::vector<Action>& out) {
    thread_local std::unique_ptr<Action[]> buf(new Action[LEGAL_CAP]);
    int n = 0;
    s.legal_actions(buf.get(), n);
    out.clear();
    for (int i = 0; i < n; i++)
        if (playable(buf[i])) out.push_back(buf[i]);
}

// Keep in sync with GameState::apply_action (apply_action.cpp).
void apply_inplace(GameState& s, const Action& a) {
    switch (a.type) {
        case ActionType::EndTurn:         s.apply_end_turn();          break;
        case ActionType::ResearchTech:    s.apply_research_tech(a);    break;
        case ActionType::TrainUnit:       s.apply_train_unit(a);       break;
        case ActionType::HarvestResource: s.apply_harvest_resource(a); break;
        case ActionType::DebugAddPop:     s.apply_debug_add_pop(a);    break;
        case ActionType::UpgradeCity:     s.apply_upgrade_city(a);     break;
        case ActionType::CaptureCity:     s.apply_capture_city(a);     break;
        case ActionType::Move:            s.apply_move(a);             break;
        case ActionType::Attack:          s.apply_attack(a);           break;
        case ActionType::Recover:         s.apply_recover(a);          break;
        default:                                                       break;
    }
    s.update_sieges();
}

// -------------------------------------------------------------------- view --

void build_view(const GameState& s, int player, GameState& out, ViewMap& vm) {
    const int msz  = s.map_size();
    const int mtsz = s.map_tiles();
    const Player& me = LabAccess::players(s)[player];
    const Tile* st = LabAccess::tiles(s);
    const Unit* su = LabAccess::units(s);
    const City* sc = LabAccess::cities(s);

    LabAccess::copy_header(s, out);

    // Cities on explored tiles keep their relative order under new positional ids.
    const int cities = LabAccess::city_count(s);
    City* oc = LabAccess::cities(out);
    int vc = 0;
    for (int i = 0; i < cities; i++) {
        if (me.is_visible(sc[i].tile_index())) {
            vm.city_to_view[i] = vc;
            oc[vc++] = sc[i];
        } else {
            vm.city_to_view[i] = -1;
        }
    }
    vm.city_count = vc;
    LabAccess::city_count(out) = vc;

    // Units keep their engine ids so bots can follow them across turns.
    const int units = LabAccess::unit_count(s);
    Unit* ou = LabAccess::units(out);
    int max_id = -1;
    for (int i = 0; i < units; i++) {
        const Unit& u = su[i];
        if (u.is_alive() && me.is_visible(u.tile_index())) {
            ou[i] = u;
            int home = u.city_id();
            ou[i].set_city_id(home >= 0 ? vm.city_to_view[home] : -1);
            max_id = i;
        } else {
            ou[i] = Unit();
        }
    }
    LabAccess::unit_count(out) = max_id + 1;

    Tile* ot = LabAccess::tiles(out);
    for (int i = 0; i < mtsz; i++) {
        if (!me.is_visible(i)) {
            ot[i] = Tile();
            ot[i].set_terrain(TerrainType::None);
            continue;
        }
        Tile t = st[i];
        if (t.has_city()) t.place_city(vm.city_to_view[t.city_id()]);
        if (t.border_city_id() >= 0) t.set_border_city(vm.city_to_view[t.border_city_id()]);
        ot[i] = t;
    }

    Player* op = LabAccess::players(out);
    for (int p = 0; p < MAX_PLAYERS; p++) {
        op[p] = LabAccess::players(s)[p];
        if (p != player) for (int w = 0; w < FOG_WORDS; w++) op[p].visible[w] = me.visible[w];
    }
    const int* caps = LabAccess::capitals(s);
    int* oc_caps = LabAccess::capitals(out);
    for (int p = 0; p < MAX_PLAYERS; p++)
        oc_caps[p] = caps[p] >= 0 ? vm.city_to_view[caps[p]] : -1;
    (void)msz;
}

Action action_to_view(const Action& a, const ViewMap& vm) {
    Action v = a;
    switch (a.type) {
        case ActionType::HarvestResource:
        case ActionType::UpgradeCity:
        case ActionType::DebugAddPop:
            v.from = vm.city_to_view[a.from];
            break;
        case ActionType::CaptureCity:
            if (a.from >= 0) v.from = vm.city_to_view[a.from];
            break;
        default:
            break;
    }
    return v;
}

json observation_json(const GameState& s, int player, bool fog, GameState& scratch, ViewMap& vm) {
    const GameState* src = &s;
    if (fog) {
        build_view(s, player, scratch, vm);
        src = &scratch;
    }
    json j = GameState::serialise(*src);
    j["player"] = player;
    j["observation"] = fog ? "fog" : "full";
    const Player& me = LabAccess::players(s)[player];
    const Tile* st = LabAccess::tiles(s);
    const City* sc = LabAccess::cities(s);
    auto& tiles = j["tiles"];
    for (int i = 0; i < s.map_tiles(); i++) {
        if (fog && !me.is_visible(i)) {
            tiles[i]["fog"] = true;
            continue;
        }
        int b = st[i].border_city_id();
        if (b >= 0) tiles[i]["territory"] = sc[b].owner();
    }
    return j;
}

json actions_json(const std::vector<Action>& actions, int map_size) {
    json arr = json::array();
    for (const Action& a : actions) arr.push_back(psb::action_to_json(a, map_size));
    return arr;
}

// ------------------------------------------------------------------- rules --

static const char* TERRAIN_NAMES[]  = { "None", "Field", "Forest", "Mountain", "Water", "Village" };
static const char* RESOURCE_NAMES[] = { "None", "Fruit", "Crop", "Animal", "Metal" };
static const char* BUILDING_NAMES[] = { "None", "Mine", "Farm", "Road" };
static const char* UNIT_NAMES[]     = { "None", "Warrior", "Archer", "Rider", "Defender", "Giant" };
static const char* UPGRADE_NAMES[]  = { "Workshop", "Explorer", "Resources", "Walls",
                                        "Border Growth", "Population Growth", "Park", "Super Unit" };

static const char* tech_name(int t) {
    return (t >= 0 && t < (int)TechType::Count) ? tech_def((TechType)t).name : "?";
}

json rules_json() {
    json r;
    r["protocol"] = psb::PROTOCOL_VERSION;
    r["engine"] = POLYSHARK_ENGINE_HASH;
    r["max_players"] = MAX_PLAYERS;
    r["terrain"] = TERRAIN_NAMES;
    r["buildings"] = BUILDING_NAMES;
    json actions = json::array();
    for (int t = 0; t <= (int)ActionType::EndTurn; t++) actions.push_back(psb::action_name((ActionType)t));
    r["actions"] = actions;

    json resources = json::array();
    for (int i = 0; i < 5; i++) {
        const ResourceDef& d = resource_def((ResourceType)i);
        resources.push_back({ { "id", i }, { "name", RESOURCE_NAMES[i] },
                              { "tech", d.required_tech == TechType::Count ? json() : json(tech_name((int)d.required_tech)) },
                              { "cost", d.star_cost }, { "population", d.pop_reward },
                              { "building", BUILDING_NAMES[(int)d.places_building] } });
    }
    r["resources"] = resources;

    json units = json::array();
    for (int i = 0; i < (int)UnitType::Count; i++) {
        const UnitDef& d = unit_def((UnitType)i);
        json abilities = json::array();
        if (d.abilities & ABILITY_FORTIFY) abilities.push_back("fortify");
        if (d.abilities & ABILITY_DASH)    abilities.push_back("dash");
        if (d.abilities & ABILITY_ESCAPE)  abilities.push_back("escape");
        if (d.abilities & ABILITY_RANGED)  abilities.push_back("ranged");
        if (d.abilities & ABILITY_STATIC)  abilities.push_back("static");
        json tech;
        if (d.required_tech == TechType::None) tech = "(super unit)";
        else if (d.required_tech != TechType::Count) tech = tech_name((int)d.required_tech);
        units.push_back({ { "id", i }, { "name", UNIT_NAMES[i] }, { "hp", d.hp }, { "attack", d.attack },
                          { "defense", d.defense }, { "movement", d.movement }, { "range", d.range },
                          { "cost", d.cost }, { "abilities", abilities }, { "tech", tech } });
    }
    r["units"] = units;

    json techs = json::array();
    for (int i = 0; i < (int)TechType::Count; i++) {
        const TechDef& d = tech_def((TechType)i);
        json unlocks = json::array(), res = json::array(), terr = json::array();
        for (int u = 0; u < (int)TechType::Count; u++)
            if (d.unlocks & (1u << u)) unlocks.push_back(tech_name(u));
        for (ResourceType rt : d.unlocks_resources)
            if (rt != ResourceType::None) res.push_back(RESOURCE_NAMES[(int)rt]);
        for (TerrainType tt : d.defense_terrains)
            if (tt != TerrainType::None) terr.push_back(TERRAIN_NAMES[(int)tt]);
        techs.push_back({ { "id", i }, { "name", d.name }, { "tier", d.tier }, { "unlocks", unlocks },
                          { "resources", res }, { "defence_terrain", terr } });
    }
    r["techs"] = techs;
    r["tech_cost"] = "tier 0: 0; otherwise 4 + tier + tier * max(owned_cities - 1, 0)";

    json upgrades = json::array();
    for (int i = 0; i < (int)CityUpgradeType::Count; i++)
        upgrades.push_back({ { "id", i }, { "name", UPGRADE_NAMES[i] }, { "level", i < 6 ? 2 + i / 2 : 5 } });
    r["upgrades"] = upgrades;

    r["constants"] = {
        { "default_map_size", cfg::map::DEFAULT_SIZE },
        { "capital_star_bonus", cfg::city::CAPITAL_STAR_BONUS },
        { "resources_star_reward", cfg::city::L2_RESOURCES_STAR_REWARD },
        { "population_growth_reward", cfg::city::L3_POP_GROWTH_REWARD },
        { "border_radius", { cfg::city::BORDER_RADIUS_BASE, cfg::city::BORDER_RADIUS_GROWN } },
        { "heal_friendly", cfg::combat::HEAL_FRIENDLY_TERRITORY },
        { "heal_neutral", cfg::combat::HEAL_NEUTRAL },
        { "kills_for_promotion", cfg::combat::KILLS_FOR_PROMOTION },
        { "unit_vision", cfg::vision::UNIT_RADIUS },
        { "mountain_vision", cfg::vision::MOUNTAIN_RADIUS },
        { "max_map_size", MAX_MAP_SIZE },
        { "max_path_steps", MAX_MOVE_PATH_STEPS },
    };
    r["directions"] = { { "dx", psb::DIR_DX }, { "dy", psb::DIR_DY },
                        { "names", { "N", "NE", "E", "SE", "S", "SW", "W", "NW" } } };
    return r;
}

std::string state_hash(const GameState& s) {
    std::string dump = GameState::serialise(s).dump();
    uint64_t h = 0xcbf29ce484222325ULL;
    for (unsigned char c : dump) { h ^= c; h *= 0x100000001b3ULL; }
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", (unsigned long long)h);
    return buf;
}

// ------------------------------------------------------------------ labels --

std::string tile_label(int tile, int map_size) {
    return "(" + std::to_string(tile % map_size) + "," + std::to_string(tile / map_size) + ")";
}

static std::string unit_on(const GameState& s, int tile) {
    const Tile& t = s.tile_at(tile);
    if (!t.has_unit()) return "unit";
    int type = (int)s.get_unit(t.unit_id()).type();
    return (type > 0 && type < (int)UnitType::Count) ? UNIT_NAMES[type] : "unit";
}

std::string describe_action(const GameState& s, const Action& a) {
    const int msz = s.map_size();
    switch (a.type) {
        case ActionType::Move:
            return unit_on(s, a.from) + " " + tile_label(a.from, msz) + " → " + tile_label(a.to, msz);
        case ActionType::Attack:
            return unit_on(s, a.from) + " " + tile_label(a.from, msz) + " attacks " + unit_on(s, a.to) + " " +
                   tile_label(a.to, msz);
        case ActionType::TrainUnit:
            return "Train " + std::string(a.param > 0 && a.param < (int)UnitType::Count ? UNIT_NAMES[a.param] : "?") +
                   " at " + tile_label(a.from, msz);
        case ActionType::ResearchTech:
            return "Research " + std::string(tech_name(a.param));
        case ActionType::HarvestResource:
            return "Harvest " + std::string(a.param >= 0 && a.param < 5 ? RESOURCE_NAMES[a.param] : "?") + " " +
                   tile_label(a.to, msz);
        case ActionType::CaptureCity:
            return std::string(a.from >= 0 ? "Capture city " : "Capture village ") + tile_label(a.to, msz);
        case ActionType::UpgradeCity: {
            int ct = (a.from >= 0 && a.from < LabAccess::city_count(s)) ? s.get_city(a.from).tile_index() : -1;
            return "City " + (ct >= 0 ? tile_label(ct, msz) : std::string("?")) + ": " +
                   (a.param >= 0 && a.param < (int)CityUpgradeType::Count ? UPGRADE_NAMES[a.param] : "?");
        }
        case ActionType::Recover:
            return "Recover " + unit_on(s, a.from) + " " + tile_label(a.from, msz);
        case ActionType::EndTurn:
            return "End turn";
        case ActionType::DebugAddPop:
            return "Debug +pop";
        case ActionType::ConstructBuilding:
            return "Build at " + tile_label(a.to, msz);
    }
    return "?";
}

// ----------------------------------------------------------------- summary --

int income(const GameState& s, int player) {
    int total = 0;
    const City* c = LabAccess::cities(s);
    for (int i = 0; i < LabAccess::city_count(s); i++)
        if (c[i].owner() == player) total += c[i].stars_per_turn();
    return total;
}

json player_summary(const GameState& s, int player) {
    int cities = 0, levels = 0, units = 0, hp = 0, explored = 0;
    int by_type[(int)UnitType::Count] = {};
    const City* c = LabAccess::cities(s);
    for (int i = 0; i < LabAccess::city_count(s); i++)
        if (c[i].owner() == player) { cities++; levels += c[i].level(); }
    const Unit* u = LabAccess::units(s);
    for (int i = 0; i < LabAccess::unit_count(s); i++)
        if (u[i].is_alive() && u[i].owner() == player) { units++; hp += u[i].hp(); by_type[(int)u[i].type()]++; }
    for (int i = 0; i < s.map_tiles(); i++)
        if (s.is_visible(player, i)) explored++;
    json types = json::object();
    for (int t = 1; t < (int)UnitType::Count; t++)
        if (by_type[t]) types[UNIT_NAMES[t]] = by_type[t];
    json techs = json::array();
    for (TechType t : s.get_techs(player))
        if (t != TechType::Origin) techs.push_back(tech_def(t).name);
    return { { "stars", s.get_stars(player) }, { "income", income(s, player) }, { "cities", cities },
             { "levels", levels }, { "units", units }, { "hp", hp }, { "unit_types", types },
             { "techs", techs }, { "explored", explored } };
}

Outcome outcome(const GameState& s, int start_turn, int turn_limit) {
    Outcome o;
    int w = s.winner();
    if (w >= 0) {
        o.over = true;
        o.winner = w;
        o.reason = "capital";
    } else if (turn_limit > 0 && s.get_turn() - start_turn >= turn_limit) {
        o.over = true;
        o.reason = "turn_limit";
    }
    return o;
}

}  // namespace lab
