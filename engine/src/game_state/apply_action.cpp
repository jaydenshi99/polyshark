#include "game_state.h"
#include "unit_def.h"
#include "resource_def.h"
#include "tech_def.h"
#include "grid.h"
#include "damage_calculator.h"
#include "config.h"
#include "cassert"

// Helpers defined in game_state.cpp
extern const int MOVE_DX[8];
extern const int MOVE_DY[8];
bool tile_passable(const Tile& t, bool has_climbing);
void reveal_around(GameState& s, int player, int tile_index, int radius);
void rebuild_visibility(GameState& s, int player);
void reachable_tiles(const GameState& s, int unit_id, int8_t out_mp[], int out_parent[]);
bool encode_path_bits(const int parent[], int src, int dst, int msz,
                      uint32_t* out_bits, uint8_t* out_steps);
void decode_path_bits(int start, uint32_t bits, uint8_t steps, int msz,
                      int out_tiles[MAX_MOVE_PATH_STEPS + 1], int* out_count);

void GameState::apply_end_turn() {
    GameState& s = *this;
    const int mtsz = s.map_tiles();

    // Heal the player whose turn just ended: any of their units that neither
    // moved nor attacked this turn recovers HP. Runs BEFORE the cur_player
    // switch so the heal is visibly tied to "end of my turn" and lands before
    // the opponent can attack the unit.
    int prev = s.cur_player;
    for (int i = 0; i < mtsz; i++) {
        const Tile& t = s.tile_at(i);
        if (!t.has_unit()) continue;
        Unit& u = s.units[t.unit_id()];
        if (u.owner() != prev) continue;
        const UnitDef& udef = unit_def(u.type());
        bool skipped = (u.move_points() == udef.movement) && !u.has_attacked();
        if (!skipped || u.hp() >= u.max_hp()) continue;
        int bcid = s.map[i].border_city_id();
        bool friendly_territory = (bcid >= 0 && s.cities[bcid].owner() == prev);
        u.heal(friendly_territory ? cfg::combat::HEAL_FRIENDLY_TERRITORY
                                  : cfg::combat::HEAL_NEUTRAL);
    }

    s.cur_player = 1 - s.cur_player;
    int next = s.cur_player;

    if (next == 0) s.turn++;

    if (s.turn > 0) {
        for (int i = 0; i < mtsz; i++) {
            const Tile& t = s.tile_at(i);
            if (t.has_city() && s.get_city(t.city_id()).owner() == next)
                s.players[next].stars += s.get_city(t.city_id()).stars_per_turn();
        }
    }

    for (int i = 0; i < mtsz; i++) {
        const Tile& t = s.tile_at(i);
        if (t.has_unit()) {
            Unit& u = s.units[t.unit_id()];
            if (u.owner() == next)
                u.reset_turn(unit_def(u.type()).movement);
        }
    }

    for (int i = 0; i < s.city_count; i++) {
        City& c = s.cities[i];
        int ctile = c.tile_index();
        bool has_unit  = s.map[ctile].has_unit();
        int unit_owner = has_unit ? s.units[s.map[ctile].unit_id()].owner() : -1;
        c.set_sieged(has_unit && unit_owner != c.owner());
        c.set_capture_ready(has_unit && unit_owner == next && c.owner() != next);
    }

    for (int i = 0; i < mtsz; i++) {
        Tile& t = s.map[i];
        if (t.terrain() != TerrainType::Village || t.has_city()) continue;
        bool attacker_here = t.has_unit() && s.units[t.unit_id()].owner() == next;
        t.set_capture_ready(attacker_here);
    }

    rebuild_visibility(s, next);
}

void GameState::apply_research_tech(Action a) {
    GameState& s = *this;
    TechType tt = static_cast<TechType>(a.param);
    s.players[s.cur_player].stars -= tech_cost(tt, s.owned_cities(s.cur_player));
    s.research_tech(s.cur_player, tt);
}

void GameState::apply_train_unit(Action a) {
    GameState& s = *this;
    int p = s.cur_player;
    UnitType ut = static_cast<UnitType>(a.param);
    const UnitDef& udef = unit_def(ut);
    int city_id = s.tile_at(a.from).city_id();
    assert(city_id >= 0);
    assert(s.cities[city_id].can_spawn());
    s.players[p].stars -= udef.cost;
    int uid = s.spawn_unit(ut, p, a.from, city_id);
    // Freshly trained units can't act this turn: no movement, no attack.
    s.units[uid].end_turn_actions();
}

void GameState::apply_harvest_resource(Action a) {
    GameState& s = *this;
    int p = s.cur_player;
    ResourceType res = static_cast<ResourceType>(a.param);
    const ResourceDef& rdef = resource_def(res);
    s.players[p].stars -= rdef.star_cost;
    s.cities[a.from].add_population(rdef.pop_reward);
    s.tile_at(a.to).set_resource(ResourceType::None);
    if (rdef.places_building != BuildingType::None)
        s.tile_at(a.to).set_building(rdef.places_building);
}

void GameState::apply_debug_add_pop(Action a) {
    cities[a.from].add_population(a.param);
}

void GameState::apply_upgrade_city(Action a) {
    GameState& s = *this;
    int p   = s.cur_player;
    City& c = s.cities[a.from];
    c.clear_pending_upgrade();
    c.try_levelup();

    switch (static_cast<CityUpgradeType>(a.param)) {
        case CityUpgradeType::L1_WORKSHOP:          c.set_workshop(true);  break;
        case CityUpgradeType::L1_EXPLORER:          s.explore(c.tile_index(), p); break;
        case CityUpgradeType::L2_WALLS:             c.set_walls(true);     break;
        case CityUpgradeType::L2_RESOURCES:         s.players[p].stars += cfg::city::L2_RESOURCES_STAR_REWARD; break;
        case CityUpgradeType::L3_BORDER_GROWTH:
            c.update_border(cfg::city::BORDER_RADIUS_GROWN);
            reveal_around(s, p, c.tile_index(), cfg::city::BORDER_RADIUS_GROWN);
            s.claim_border_for_city(a.from);
            break;
        case CityUpgradeType::L3_POPULATION_GROWTH: c.add_population(cfg::city::L3_POP_GROWTH_REWARD);  break;
        case CityUpgradeType::L4_PARK:              break;
        case CityUpgradeType::L4_SUPERUNIT: {
            int uid = s.spawn_unit(UnitType::Giant, p, c.tile_index(), a.from);
            if (uid >= 0) s.units[uid].end_turn_actions();
            break;
        }
        default:                                    break;
    }
}

void GameState::apply_capture_city(Action a) {
    GameState& s = *this;
    int p = s.cur_player;
    Tile& ct = s.map[a.to];
    ct.set_capture_ready(false);

    int new_city_id;
    int prev_owner = -1;
    if (a.from >= 0) {
        City& c = s.cities[a.from];
        prev_owner = c.owner();
        c.capture(p);   // also clears capture_ready
        new_city_id = a.from;
    } else {
        ct.set_terrain(TerrainType::Field);
        new_city_id = s.spawn_city(a.to, p, false);
    }

    // Transfer the capturing unit's home to the captured city
    int uid = s.map[a.to].unit_id();
    assert(uid >= 0);
    Unit& u = s.units[uid];
    u.end_turn_actions();
    if (u.city_id() >= 0) s.cities[u.city_id()].remove_unit();
    u.set_city_id(new_city_id);
    s.cities[new_city_id].add_unit();

    // Re-home any other units that were registered to this city but still
    // belong to the previous owner — point them at the previous owner's capital
    // so unit.city_id always references a city owned by unit.owner.
    if (prev_owner >= 0 && prev_owner != p) {
        int rehome_to = s.capital_city[prev_owner];
        // If we just captured prev_owner's capital, or their capital pointer
        // is otherwise no longer theirs, there's no valid home to fall back on.
        if (rehome_to == new_city_id || rehome_to < 0
            || s.cities[rehome_to].owner() != prev_owner) {
            rehome_to = -1;
        }
        for (int i = 0; i < s.unit_count; i++) {
            Unit& ou = s.units[i];
            if (!ou.is_alive()) continue;
            if (i == uid) continue;
            if (ou.city_id() != new_city_id) continue;
            if (ou.owner() != prev_owner) continue;
            s.cities[new_city_id].remove_unit();
            if (rehome_to >= 0) {
                ou.set_city_id(rehome_to);
                s.cities[rehome_to].add_unit();
            } else {
                ou.set_city_id(-1);
            }
        }
    }

    reveal_around(s, p, a.to, 1);
    rebuild_visibility(s, p);
}

void GameState::apply_move(Action a) {
    GameState& s = *this;
    int src = a.from;
    int dst = a.to;
    int uid = s.map[src].unit_id();
    Unit& u = s.units[uid];
    int p   = u.owner();
    int msz = s.map_size();

    // Reveal fog along the path. If the action carries an encoded path, use it;
    // otherwise (e.g. legacy replay action with path_steps == 0) recompute via BFS
    // on the current state — deterministic, same result as the original emit.
    uint32_t pbits  = a.path_bits;
    uint8_t  psteps = a.path_steps;
    if (psteps == 0 && src != dst) {
        int8_t mp_at[MAX_MAP_TILES];
        int    parent[MAX_MAP_TILES];
        reachable_tiles(s, uid, mp_at, parent);
        encode_path_bits(parent, src, dst, msz, &pbits, &psteps);
    }
    int path_tiles[MAX_MOVE_PATH_STEPS + 1];
    int path_count = 0;
    decode_path_bits(src, pbits, psteps, msz, path_tiles, &path_count);
    for (int i = 0; i < path_count; i++) {
        // Standard vision radius (mountain bonus is handled by rebuild_visibility
        // when the unit settles at its destination).
        reveal_around(s, p, path_tiles[i], 1);
    }

    s.map[src].remove_unit();
    s.map[dst].place_unit(uid);
    u.set_tile(dst);
    // Record the final-hop direction (last 3-bit dir in path_bits) for
    // forced-spawn pushes. psteps > 0 here since src != dst.
    if (psteps > 0) {
        int last_hop = (int)((pbits >> (3 * (psteps - 1))) & 0x7u);
        u.set_last_dir((int8_t)last_hop);
    }
    // Single-move-per-turn rule: spend the entire remaining movement budget so a
    // unit that used part of its movement (e.g. a Rider's 2-mp Rider that only hopped
    // 1 tile) can't initiate another Move this turn. Note: Escape (Rider's post-attack
    // ability) still resets move points via reset_turn — that's intentional.
    u.spend_movement(u.move_points());
    rebuild_visibility(s, p);
}

void GameState::apply_attack(Action a) {
    GameState& s = *this;
    int atk_tile = a.from;
    int def_tile = a.to;
    int atk_uid  = s.map[atk_tile].unit_id();
    int def_uid  = s.map[def_tile].unit_id();
    Unit& attacker = s.units[atk_uid];
    Unit& defender = s.units[def_uid];

    int  dist     = chebyshev_distance(atk_tile, def_tile, s.map_size());
    bool opponent_sees_attacker = s.is_visible(defender.owner(), atk_tile);
    // Advance into the slain tile only if it was a melee attack (distance 1).
    bool melee_attack = !attacker.has_ability(ABILITY_RANGED);
    // Attack direction (atk → def), reduced to one of the 8 cardinal/diagonal
    // unit-vectors via sign. Used to update _last_dir on ranged shots and on
    // melee kill-advances (both effectively give the unit a "facing" for
    // forced-spawn push rules).
    int sz = s.map_size();
    int adx = (def_tile % sz) - (atk_tile % sz);
    int ady = (def_tile / sz) - (atk_tile / sz);
    int attack_dir = -1;
    int asx = (adx > 0) - (adx < 0);
    int asy = (ady > 0) - (ady < 0);
    for (int d = 0; d < 8; d++)
        if (MOVE_DX[d] == asx && MOVE_DY[d] == asy) { attack_dir = d; break; }
    float def_bonus = s.get_defense_bonus(s.players[defender.owner()], s.map[def_tile]);
    DamageCalculator::Result cr = DamageCalculator::resolve(
        attacker, defender, s.map[def_tile], def_bonus, dist, opponent_sees_attacker);

    attacker.take_damage(cr.attacker_damage);
    defender.take_damage(cr.defender_damage);

    if (cr.defender_dies) {
        if (defender.city_id() >= 0)
            s.cities[defender.city_id()].remove_unit();
        s.map[def_tile].remove_unit();
        if (!cr.attacker_dies) {
            attacker.add_kill();
            // Advance into the slain tile only if the attacker could
            // legally have walked onto it (passable terrain + tech).
            bool climbing  = s.has_tech(attacker.owner(), TechType::Climbing);
            bool can_enter = tile_passable(s.map[def_tile], climbing);
            if (melee_attack && can_enter) {
                s.map[attacker.tile_index()].remove_unit();
                s.map[def_tile].place_unit(atk_uid);
                attacker.set_tile(def_tile);
                // Kill-advance is effectively a move — record the direction.
                if (attack_dir >= 0) attacker.set_last_dir((int8_t)attack_dir);
            }
        }
    }
    attacker.mark_attacked();
    // Ranged attackers don't move but still register a "last direction" so
    // forced-spawn rule 3 (push ranged units along their last action) can use it.
    if (!melee_attack && !cr.attacker_dies && attack_dir >= 0)
        attacker.set_last_dir((int8_t)attack_dir);
    if (cr.attacker_dies) {
        if (attacker.city_id() >= 0)
            s.cities[attacker.city_id()].remove_unit();
        s.map[attacker.tile_index()].remove_unit();
    } else {
        if (attacker.has_ability(ABILITY_ESCAPE)) {
            attacker.reset_turn(unit_def(attacker.type()).movement);
            attacker.mark_attacked();
        } else{
            attacker.spend_movement(attacker.move_points());
        }
    }

    rebuild_visibility(s, s.cur_player);
}

void GameState::apply_recover(Action a) {
    GameState& s = *this;
    Tile& t = s.map[a.from];
    Unit& u = s.units[t.unit_id()];
    int bcid = t.border_city_id();
    bool friendly_territory = (bcid >= 0 && s.cities[bcid].owner() == u.owner());
    u.heal(friendly_territory ? cfg::combat::HEAL_FRIENDLY_TERRITORY
                              : cfg::combat::HEAL_NEUTRAL);
    u.spend_movement(unit_def(u.type()).movement);
    u.mark_attacked();
}

GameState GameState::apply_action(Action a) const {
    GameState s = *this;
    switch (a.type) {
        case ActionType::EndTurn:         s.apply_end_turn();           break;
        case ActionType::ResearchTech:    s.apply_research_tech(a);     break;
        case ActionType::TrainUnit:       s.apply_train_unit(a);        break;
        case ActionType::HarvestResource: s.apply_harvest_resource(a);  break;
        case ActionType::DebugAddPop:     s.apply_debug_add_pop(a);     break;
        case ActionType::UpgradeCity:     s.apply_upgrade_city(a);      break;
        case ActionType::CaptureCity:     s.apply_capture_city(a);      break;
        case ActionType::Move:            s.apply_move(a);              break;
        case ActionType::Attack:          s.apply_attack(a);            break;
        case ActionType::Recover:         s.apply_recover(a);           break;
        default:                                                        break;
    }
    return s;
}
