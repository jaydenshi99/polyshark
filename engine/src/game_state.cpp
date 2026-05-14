#include <cstdio>
#include <cstring>
#include "game_state.h"
#include "unit_def.h"
#include "resource_def.h"
#include "tech_def.h"
#include "grid.h"

const int NO_WINNER = -1;
const int INVALID = -1;

// -------------------------------------------------------- movement helpers --

static bool tile_passable(const Tile& t, bool has_climbing) {
    if (t.terrain() == TerrainType::Water) return false;
    if (t.terrain() == TerrainType::Mountain && !has_climbing) return false;
    return true;
}

// Road tiles cost 0 movement to enter; everything else costs 1.
// Future: other buildings or unit abilities can also reduce this.
static int tile_entry_cost(const Tile& t) {
    if (t.building() == BuildingType::Road) return 0;
    return 1;
}

// Forest and mountain exhaust all remaining movement on entry.
static bool tile_exhausts_movement(const Tile& t) {
    return t.terrain() == TerrainType::Forest || t.terrain() == TerrainType::Mountain;
}

// ZOC: any of the 4 orthogonal neighbours of `tile` is an enemy unit.
static bool adjacent_to_enemy(const GameState& s, int tile, int my_player) {
    int x, y;
    to_coords(tile, x, y);
    const int DX[] = { 0, 1, 0, -1 };  // N E S W (clockwise)
    const int DY[] = { -1, 0, 1, 0 };
    for (int d = 0; d < 4; d++) {
        int nx = x + DX[d], ny = y + DY[d];
        if (!in_bounds(nx, ny)) continue;
        int ni = to_index(nx, ny);
        const Tile& nt = s.tile_at(ni);
        if (nt.has_unit() && s.get_unit(nt.unit_id()).owner() != my_player) return true;
    }
    return false;
}

// BFS flood-fill from unit's current position.
// out_mp[i] = remaining movement points after reaching tile i (-1 = unreachable).
// The unit's own tile is always set to its current move_points.
// Enemy-occupied tiles are NOT included (they are attack targets, not move destinations).
static void reachable_tiles(const GameState& s, int unit_id, int8_t out_mp[MAP_TILES]) {
    const Unit& u  = s.get_unit(unit_id);
    int          p  = u.owner();
    bool climbing   = s.has_tech(p, TechType::Climbing);

    memset(out_mp, -1, MAP_TILES);

    struct Node { int tile; int8_t mp; };
    // Upper bound: each tile can be re-pushed at most (max_mp+1) times.
    Node queue[MAP_TILES * 4];
    int head = 0, tail = 0;

    int src = u.tile_index();
    out_mp[src] = (int8_t)u.move_points();
    queue[tail++] = { src, (int8_t)u.move_points() };

    const int DX[] = { 0, 1, 1, 1, 0, -1, -1, -1 };  // N NE E SE S SW W NW (clockwise)
    const int DY[] = { -1, -1, 0, 1, 1,  1,  0, -1 };

    while (head < tail) {
        int tile  = queue[head].tile;
        int8_t mp = queue[head].mp;
        head++;

        if (mp <= 0) continue;

        int x, y;
        to_coords(tile, x, y);

        for (int d = 0; d < 8; d++) {
            int nx = x + DX[d], ny = y + DY[d];
            if (!in_bounds(nx, ny)) continue;
            int ntile = to_index(nx, ny);
            const Tile& nt = s.tile_at(ntile);

            if (!tile_passable(nt, climbing)) continue;
            // Any occupied tile (ally or enemy) blocks movement.
            if (nt.has_unit()) continue;

            int cost = tile_entry_cost(nt);
            if (cost > mp) continue;

            // ZOC and slow terrain both exhaust remaining movement after entry.
            bool zoc = adjacent_to_enemy(s, ntile, p);
            int8_t new_mp = (tile_exhausts_movement(nt) || zoc)
                            ? 0
                            : (int8_t)(mp - cost);

            if (new_mp > out_mp[ntile]) {
                out_mp[ntile] = new_mp;
                queue[tail++] = { ntile, new_mp };
            }
        }
    }
}

// Manhattan distance between two tile indices (used for range checks).
static int tile_distance(int t1, int t2) {
    int x1, y1, x2, y2;
    to_coords(t1, x1, y1);
    to_coords(t2, x2, y2);
    return (x1 > x2 ? x1 - x2 : x2 - x1) + (y1 > y2 ? y1 - y2 : y2 - y1);
}

// ---------------------------------------------------------- combat helpers --

// Defender's terrain/city bonus, applied only when the unit has FORTIFY.
static float defence_bonus(const Tile& def_tile, uint32_t def_abilities) {
    if (!(def_abilities & ABILITY_FORTIFY)) return 1.0f;
    TerrainType t = def_tile.terrain();
    if (t == TerrainType::Forest || t == TerrainType::Mountain) return 1.5f;
    if (def_tile.has_city()) return 1.5f;
    return 1.0f;
}

struct CombatResult {
    int  attacker_damage;
    int  defender_damage;
    bool attacker_dies;
    bool defender_dies;
};

static CombatResult compute_combat(
    const Unit& attacker, const Unit& defender,
    const Tile& def_tile, bool is_ranged)
{
    const UnitDef& adef = unit_def(attacker.type());
    const UnitDef& ddef = unit_def(defender.type());

    float db             = defence_bonus(def_tile, ddef.abilities);
    float attack_force   = adef.attack  * ((float)attacker.hp() / attacker.max_hp());
    float defence_force  = ddef.defense * ((float)defender.hp() / defender.max_hp()) * db;
    float total          = attack_force + defence_force;

    // defender takes attackResult; attacker takes defenceResult
    int def_dmg = (int)((attack_force  / total) * adef.attack  * 4.5f + 0.5f);
    int atk_dmg = is_ranged
                  ? 0
                  : (int)((defence_force / total) * ddef.defense * 4.5f + 0.5f);

    def_dmg = def_dmg < 1 ? 1 : def_dmg;
    if (!is_ranged) atk_dmg = atk_dmg < 1 ? 1 : atk_dmg;

    CombatResult r;
    r.defender_damage  = def_dmg;
    r.attacker_damage  = atk_dmg;
    r.defender_dies    = (defender.hp()  - def_dmg <= 0);
    r.attacker_dies    = (attacker.hp()  - atk_dmg <= 0);
    return r;
}

// Helpers use only the public GameState API — no private access needed.

static void reveal_around(GameState& s, int player, int tile_index, int radius) {
    int x, y;
    to_coords(tile_index, x, y);
    for (int dy = -radius; dy <= radius; dy++)
        for (int dx = -radius; dx <= radius; dx++)
            if (in_bounds(x + dx, y + dy))
                s.set_explored(player, to_index(x + dx, y + dy));
}

static void rebuild_visibility(GameState& s, int player) {
    s.clear_visible(player);
    for (int i = 0; i < MAP_TILES; i++) {
        const Tile& t = s.tile_at(i);
        if (t.has_unit() && s.get_unit(t.unit_id()).owner() == player) {
            int r = (t.terrain() == TerrainType::Mountain) ? 2 : 1;
            reveal_around(s, player, i, r);
        }
        if (t.has_city() && s.get_city(t.city_id()).owner() == player)
            reveal_around(s, player, i, 1);
    }
}
bool GameState::is_terminal() const {
    // TODO: define win conditions
    return false;
}

int GameState::winner() const {
    // TODO: return winning player index once win conditions are defined
    return NO_WINNER;
}

int GameState::current_player() const {
    return cur_player;
}

int GameState::spawn_unit(UnitType type, int owner, int tile_index) {
    if (unit_count >= MAX_UNITS) return INVALID;
    int id = unit_count++;
    Unit& u = units[id];
    const UnitDef& def = unit_def(type);
    u.set_type(type);
    u.set_owner(owner);
    u.set_tile(tile_index);
    u.set_max_hp(def.hp);
    u.set_hp(def.hp);
    u.reset_turn(def.movement);
    map[tile_index].place_unit(id);
    return id;
}

int GameState::spawn_city(int tile_index, int owner, bool is_capital) {
    if (city_count >= MAX_CITIES) return INVALID;
    int id = city_count++;
    City& c = cities[id];
    c.set_tile(tile_index);
    c.set_owner(owner);
    c.set_capital(is_capital);
    map[tile_index].place_city(id);
    return id;
}

void GameState::print_map() const {
    for (int y = 0; y < MAP_SIZE; y++) {
        for (int x = 0; x < MAP_SIZE; x++) {
            const Tile& t = map[to_index(x, y)];
            char c;
            switch (t.terrain()) {
                case TerrainType::Field:    c = '.'; break;
                case TerrainType::Forest:   c = 'F'; break;
                case TerrainType::Mountain: c = 'M'; break;
                case TerrainType::Water:    c = '~'; break;
                case TerrainType::Village:  c = 'V'; break;
                default:                    c = '?'; break;
            }
            if (t.has_city()) c = 'C';
            if (t.has_unit()) c = 'U';
            printf("%c ", c);
        }
        printf("\n");
    }
}


GameStateType GameState::phase() const {
    for (int i = 0; i < city_count; i++)
        if (cities[i].owner() == cur_player && cities[i].has_pending_upgrade())
            return GameStateType::UpgradingCity;
    return GameStateType::Idle;
}

// --------------------------------------------------------- public interface --

void GameState::legal_actions(Action out[], int& out_count) const {
    const GameState& s = *this;
    out_count = 0;
    int p = s.current_player();

    // --- UpgradeCity ---
    // All other actions are suppressed while any city has a pending upgrade.
    for (int i = 0; i < s.city_count; i++) {
        const City& city = s.cities[i];
        if (city.owner() != p || !city.has_pending_upgrade()) continue;

        // Pick the option pair for this level transition.
        // Level stored is the NEW level after the pop tick.
        CityUpgradeType opt_a, opt_b;
        switch (city.level()) {
            case 2:  opt_a = CityUpgradeType::L1_WORKSHOP;      opt_b = CityUpgradeType::L1_EXPLORER; break;
            case 3:  opt_a = CityUpgradeType::L2_RESOURCES;     opt_b = CityUpgradeType::L2_WALLS;       break;
            case 4:  opt_a = CityUpgradeType::L3_BORDER_GROWTH; opt_b = CityUpgradeType::L3_POPULATION_GROWTH;      break;
            default: opt_a = CityUpgradeType::L4_PARK;          opt_b = CityUpgradeType::L4_SUPERUNIT;       break;
        }
        out[out_count++] = { ActionType::UpgradeCity, i, -1, (int)opt_a, true };
        out[out_count++] = { ActionType::UpgradeCity, i, -1, (int)opt_b, true };
        break;
    }

    bool can_harvest = (s.phase() != GameStateType::UpgradingCity);

    if (can_harvest) {
        // --- Train Unit ---
        for (int i = 0; i < s.city_count; i++) {
            const City& city = s.cities[i];
            if (city.owner() != p) continue;
            if (city.has_pending_upgrade()) continue;
            int tile = city.tile_index();
            if (s.tile_at(tile).has_unit()) continue;  // tile occupied
            // Population cap: level + 1 population slots (same as pop bar)
            int pop_cap = city.level() + 1;
            if (city.population() >= pop_cap) continue;
            for (int ut = 1; ut < static_cast<int>(UnitType::Count); ut++) {
                const UnitDef& udef = unit_def(static_cast<UnitType>(ut));
                if (udef.required_tech != TechType::Count && !s.has_tech(p, udef.required_tech)) continue;
                bool can_afford = s.players[p].stars >= udef.cost;
                out[out_count++] = { ActionType::TrainUnit, tile, tile, ut, can_afford };
            }
        }

        // --- Harvest Resource ---
        for (int i = 0; i < s.city_count; i++) {
            const City& city = s.cities[i];
            if (city.owner() != p) continue;

            int ccx, ccy;
            to_coords(city.tile_index(), ccx, ccy);
            int r = city.border_radius();

            for (int dy = -r; dy <= r; dy++) {
                for (int dx = -r; dx <= r; dx++) {
                    int bx = ccx + dx, by = ccy + dy;
                    if (!in_bounds(bx, by)) continue;
                    // TODO: midpoint clipping — if another city is closer to (bx,by) than this one, skip it.
                    int bidx = to_index(bx, by);
                    const Tile& bt = s.tile_at(bidx);
                    ResourceType res = bt.resource();
                    if (res == ResourceType::None) continue;
                    if (bt.has_building()) continue;  // already developed
                    const ResourceDef& rdef = resource_def(res);
                    if (rdef.required_tech != TechType::Count && !s.has_tech(p, rdef.required_tech)) continue;
                    if (bt.has_unit() && s.get_unit(bt.unit_id()).owner() != p) continue;
                    bool can_afford = s.players[p].stars >= rdef.star_cost;
                    out[out_count++] = { ActionType::HarvestResource, i, bidx, (int)res, can_afford };
                }
            }
        }

        // --- ResearchTech ---
        // available_techs = (STARTING_TECHS | union of all owned.unlocks) & ~owned
        uint32_t avail = available_techs(s.get_techs(p));
        while (avail) {
            int t   = __builtin_ctz(avail);
            avail  &= avail - 1;
            bool can_afford = s.players[p].stars >= tech_def(static_cast<TechType>(t)).cost;
            out[out_count++] = { ActionType::ResearchTech, -1, -1, t, can_afford };
        }

        // --- DebugAddPop ---
        for (int i = 0; i < s.city_count; i++) {
            if (s.cities[i].owner() == p) {
                out[out_count++] = { ActionType::DebugAddPop, i, -1, 1,  true };
                out[out_count++] = { ActionType::DebugAddPop, i, -1, 10, true };
            }
        }

        // --- CaptureCity ---
        // from = city_id for enemy cities, -1 for uncaptured villages; to = tile index.
        for (int i = 0; i < MAP_TILES; i++) {
            const Tile& t = s.tile_at(i);
            if (!t.has_unit()) continue;
            if (s.get_unit(t.unit_id()).owner() != p) continue;

            if (t.has_city()) {
                // Enemy city: must have been sieged since last turn.
                const City& c = s.get_city(t.city_id());
                if (c.owner() != p && c.capture_ready())
                    out[out_count++] = { ActionType::CaptureCity, t.city_id(), i, 0, true };
            } else if (t.terrain() == TerrainType::Village && t.capture_ready()) {
                // Uncaptured village: no city object yet; spawn one on capture.
                out[out_count++] = { ActionType::CaptureCity, -1, i, 0, true };
            }
        }

        // --- Move and Attack ---
        for (int i = 0; i < MAP_TILES; i++) {
            const Tile& t = s.tile_at(i);
            if (!t.has_unit()) continue;
            const Unit& u = s.get_unit(t.unit_id());
            if (u.owner() != p) continue;

            const UnitDef& udef = unit_def(u.type());

            // Move: unit still has movement points (handles ESCAPE post-kill too).
            if (u.move_points() > 0) {
                int8_t mp_at[MAP_TILES];
                reachable_tiles(s, t.unit_id(), mp_at);
                for (int j = 0; j < MAP_TILES; j++) {
                    if (j == i) continue;
                    if (mp_at[j] < 0) continue;
                    out[out_count++] = { ActionType::Move, i, j, mp_at[j], true };
                }
            }

            // Attack: enemy within range that the current player can see.
            if (!u.has_attacked()) {
                for (int j = 0; j < MAP_TILES; j++) {
                    const Tile& dt = s.tile_at(j);
                    if (!dt.has_unit()) continue;
                    const Unit& target = s.get_unit(dt.unit_id());
                    if (target.owner() == p) continue;
                    if (!s.is_visible(p, j)) continue;
                    int dist = tile_distance(i, j);
                    if (dist < 1 || dist > udef.range) continue;
                    out[out_count++] = { ActionType::Attack, i, j, 0, true };
                }
            }
        }
    }

    // --- EndTurn (blocked while upgrades are pending) ---
    bool can_end = (s.phase() != GameStateType::UpgradingCity);
    if (can_end)
        out[out_count++] = { ActionType::EndTurn, -1, -1, 0, true };
}

GameState GameState::apply_action(Action a) const {
    GameState s = *this;
    switch (a.type) {

        case ActionType::EndTurn: {
            // Switch to next player
            s.cur_player = 1 - s.cur_player;
            int next = s.cur_player;

            // A full round completes when we wrap back to player 0
            if (next == 0) s.turn++;

            // Collect income — skip the very first round (turn 0) so neither player
            // gets free stars before they've had a chance to play.
            if (s.turn > 0) {
                for (int i = 0; i < MAP_TILES; i++) {
                    const Tile& t = s.tile_at(i);
                    if (t.has_city() && s.get_city(t.city_id()).owner() == next)
                        s.players[next].stars += s.get_city(t.city_id()).stars_per_turn();
                }
            }

            // Heal units that neither moved nor attacked last turn.
            // 4 HP in friendly city territory, 2 HP everywhere else. Capped at max HP.
            for (int i = 0; i < MAP_TILES; i++) {
                const Tile& t = s.tile_at(i);
                if (!t.has_unit()) continue;
                Unit& u = s.units[t.unit_id()];
                if (u.owner() != next) continue;
                const UnitDef& udef = unit_def(u.type());
                bool skipped = (u.move_points() == udef.movement) && !u.has_attacked();
                if (!skipped || u.hp() >= u.max_hp()) continue;

                // Check if the tile is within a friendly city's border
                int ux, uy;
                to_coords(i, ux, uy);
                bool friendly_territory = false;
                for (int c = 0; c < s.city_count && !friendly_territory; c++) {
                    const City& city = s.cities[c];
                    if (city.owner() != next) continue;
                    int cx, cy;
                    to_coords(city.tile_index(), cx, cy);
                    int r = city.border_radius();
                    if (ux >= cx - r && ux <= cx + r && uy >= cy - r && uy <= cy + r)
                        friendly_territory = true;
                }
                u.heal(friendly_territory ? 4 : 2);
            }

            for (int i = 0; i < MAP_TILES; i++) {
                const Tile& t = s.tile_at(i);
                if (t.has_unit()) {
                    Unit& u = s.units[t.unit_id()];
                    if (u.owner() == next)
                        u.reset_turn(unit_def(u.type()).movement);
                }
            }

            // Rebuild siege / capture_ready for real cities.
            for (int i = 0; i < s.city_count; i++) {
                City& c = s.cities[i];
                int ctile = c.tile_index();
                bool has_unit = s.map[ctile].has_unit();
                int unit_owner = has_unit ? s.units[s.map[ctile].unit_id()].owner() : -1;

                c.set_sieged(has_unit && unit_owner != c.owner());
                c.set_capture_ready(has_unit && unit_owner == next && c.owner() != next);
            }

            // Rebuild capture_ready for village tiles (no city object yet).
            for (int i = 0; i < MAP_TILES; i++) {
                Tile& t = s.map[i];
                if (t.terrain() != TerrainType::Village || t.has_city()) continue;
                bool attacker_here = t.has_unit() && s.units[t.unit_id()].owner() == next;
                t.set_capture_ready(attacker_here);
            }

            rebuild_visibility(s, next);
            return s;
        }

        case ActionType::ResearchTech: {
            TechType tt = static_cast<TechType>(a.param);
            s.players[s.cur_player].stars -= tech_def(tt).cost;
            s.research_tech(s.cur_player, tt);
            return s;
        }

        case ActionType::TrainUnit: {
            int p = s.cur_player;
            UnitType ut = static_cast<UnitType>(a.param);
            const UnitDef& udef = unit_def(ut);
            s.players[p].stars -= udef.cost;
            int uid = s.spawn_unit(ut, p, a.from);
            s.units[uid].spend_movement(s.units[uid].move_points());
            // Find which city owns this tile and add 1 population
            for (int i = 0; i < s.city_count; i++) {
                if (s.cities[i].tile_index() == a.from) {
                    s.cities[i].add_population(1);
                    break;
                }
            }
            return s;
        }

        case ActionType::HarvestResource: {
            int p = s.cur_player;
            ResourceType res = static_cast<ResourceType>(a.param);
            const ResourceDef& rdef = resource_def(res);
            s.players[p].stars -= rdef.star_cost;
            s.cities[a.from].add_population(rdef.pop_reward);
            s.tile_at(a.to).set_resource(ResourceType::None);
            if (rdef.places_building != BuildingType::None)
                s.tile_at(a.to).set_building(rdef.places_building);
            return s;
        }

        case ActionType::DebugAddPop: {
            s.cities[a.from].add_population(a.param);
            return s;
        }

        case ActionType::UpgradeCity: {
            int p    = s.cur_player;
            City& c  = s.cities[a.from];
            c.clear_pending_upgrade();
            // Re-check: excess population may push the city up another level immediately
            c.try_levelup();

            switch (static_cast<CityUpgradeType>(a.param)) {
                case CityUpgradeType::L1_WORKSHOP:
                    c.set_workshop(true);
                    break;
                case CityUpgradeType::L1_EXPLORER:
                    // TODO: add explorer logic
                    break;
                case CityUpgradeType::L2_WALLS:
                    c.set_walls(true);
                    break;
                case CityUpgradeType::L2_RESOURCES:
                    s.players[p].stars += 5;
                    break;
                case CityUpgradeType::L3_BORDER_GROWTH:
                    c.update_border(2);
                    reveal_around(s, p, c.tile_index(), 2);
                    break;
                case CityUpgradeType::L3_POPULATION_GROWTH:
                    c.add_population(3);
                    break;
                case CityUpgradeType::L4_PARK:
                    break;
                case CityUpgradeType::L4_SUPERUNIT:
                    // TODO: spawn a free Warrior at the city tile
                    break;
                default:
                    break;
            }
            return s;
        }

        case ActionType::CaptureCity: {
            int p = s.cur_player;
            Tile& ct = s.map[a.to];
            ct.set_capture_ready(false);

            if (a.from >= 0) {
                // Enemy city already exists — transfer ownership.
                City& c = s.cities[a.from];
                c.capture(p);   // sets owner, clears sieged
                c.set_capture_ready(false);
            } else {
                // Uncaptured village — spawn a brand new level-1 city (border radius 1).
                ct.set_terrain(TerrainType::Field);
                s.spawn_city(a.to, p, false);
            }

            // Reveal the city's initial border territory (3×3, radius 1).
            reveal_around(s, p, a.to, 1);
            rebuild_visibility(s, p);
            return s;
        }

        case ActionType::Move: {
            int src = a.from;
            int dst = a.to;
            int uid = s.map[src].unit_id();
            Unit& u = s.units[uid];

            s.map[src].remove_unit();
            s.map[dst].place_unit(uid);
            u.set_tile(dst);
            // a.param holds the remaining mp after this move (computed by BFS).
            u.spend_movement(u.move_points() - a.param);

            rebuild_visibility(s, u.owner());
            return s;
        }

        case ActionType::Attack: {
            int atk_tile = a.from;
            int def_tile = a.to;
            int atk_uid  = s.map[atk_tile].unit_id();
            int def_uid  = s.map[def_tile].unit_id();
            Unit& attacker = s.units[atk_uid];
            Unit& defender = s.units[def_uid];

            bool is_ranged = unit_def(attacker.type()).abilities & ABILITY_RANGED;
            CombatResult cr = compute_combat(attacker, defender, s.map[def_tile], is_ranged);

            attacker.take_damage(cr.attacker_damage);
            defender.take_damage(cr.defender_damage);

            if (cr.defender_dies) {
                s.map[def_tile].remove_unit();
                if (!cr.attacker_dies) {
                    attacker.add_kill();
                    // Melee kill: advance attacker into the vacated tile.
                    if (!is_ranged) {
                        s.map[attacker.tile_index()].remove_unit();
                        s.map[def_tile].place_unit(atk_uid);
                        attacker.set_tile(def_tile);
                    }
                }
            }

            if (cr.attacker_dies) {
                s.map[attacker.tile_index()].remove_unit();
            } else {
                attacker.mark_attacked();
                // ESCAPE: preserve remaining movement after a kill so the unit can still move.
                bool escape = unit_def(attacker.type()).abilities & ABILITY_ESCAPE;
                if (!(escape && cr.defender_dies))
                    attacker.spend_movement(attacker.move_points());
            }

            rebuild_visibility(s, s.cur_player);
            return s;
        }

        default:
            return s;
    }
}
