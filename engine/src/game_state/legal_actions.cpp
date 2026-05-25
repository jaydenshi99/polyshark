#include "game_state.h"
#include "unit_def.h"
#include "resource_def.h"
#include "tech_def.h"
#include "grid.h"

// Helpers defined in game_state.cpp
void reachable_tiles(const GameState& s, int unit_id, int8_t out_mp[]);
void reachable_tiles(const GameState& s, int unit_id, int8_t out_mp[], int out_parent[]);
bool encode_path_bits(const int parent[], int src, int dst, int msz,
                      uint32_t* out_bits, uint8_t* out_steps);

GameStateType GameState::phase() const {
    for (int i = 0; i < city_count; i++)
        if (cities[i].owner() == cur_player && cities[i].has_pending_upgrade())
            return GameStateType::UpgradingCity;
    return GameStateType::Idle;
}

void GameState::legal_actions(Action out[], int& out_count) const {
    const GameState& s    = *this;
    const int        mtsz = map_tiles();
    out_count = 0;
    int p = s.current_player();

    // --- UpgradeCity ---
    for (int i = 0; i < s.city_count; i++) {
        const City& city = s.cities[i];
        if (city.owner() != p || !city.has_pending_upgrade()) continue;

        CityUpgradeType opt_a, opt_b;
        switch (city.level()) {
            case 2:  opt_a = CityUpgradeType::L1_WORKSHOP;      opt_b = CityUpgradeType::L1_EXPLORER;            break;
            case 3:  opt_a = CityUpgradeType::L2_RESOURCES;     opt_b = CityUpgradeType::L2_WALLS;               break;
            case 4:  opt_a = CityUpgradeType::L3_BORDER_GROWTH; opt_b = CityUpgradeType::L3_POPULATION_GROWTH;   break;
            default: opt_a = CityUpgradeType::L4_PARK;          opt_b = CityUpgradeType::L4_SUPERUNIT;           break;
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
            if (city.owner() != p || city.has_pending_upgrade() || !city.can_spawn()) continue;
            int tile = city.tile_index();
            if (s.tile_at(tile).has_unit()) continue;
            for (int ut = 1; ut < static_cast<int>(UnitType::Count); ut++) {
                const UnitDef& udef = unit_def(static_cast<UnitType>(ut));
                if (udef.required_tech != TechType::Count && !s.has_tech(p, udef.required_tech)) continue;
                bool can_afford = s.get_stars(p) >= udef.cost;
                out[out_count++] = { ActionType::TrainUnit, tile, tile, ut, can_afford };
            }
        }

        // --- Harvest Resource ---
        for (int bidx = 0; bidx < mtsz; bidx++) {
            int cid = s.tile_at(bidx).border_city_id();
            if (cid < 0 || s.get_city(cid).owner() != p) continue;
            const Tile& bt = s.tile_at(bidx);
            ResourceType res = bt.resource();
            if (res == ResourceType::None) continue;
            if (bt.has_building()) continue;
            const ResourceDef& rdef = resource_def(res);
            if (rdef.required_tech != TechType::Count && !s.has_tech(p, rdef.required_tech)) continue;
            if (bt.has_unit() && s.get_unit(bt.unit_id()).owner() != p) continue;
            bool can_afford = s.get_stars(p) >= rdef.star_cost;
            out[out_count++] = { ActionType::HarvestResource, cid, bidx, (int)res, can_afford };
        }

        // --- ResearchTech ---
        uint32_t avail = available_techs(s.techs_mask(p));
        while (avail) {
            int t  = __builtin_ctz(avail);
            avail &= avail - 1;
            bool can_afford = s.get_stars(p) >= tech_cost(static_cast<TechType>(t), s.owned_cities(p));
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
        for (int i = 0; i < mtsz; i++) {
            const Tile& t = s.tile_at(i);
            if (!t.has_unit()) continue;
            if (s.get_unit(t.unit_id()).owner() != p) continue;

            if (t.has_city()) {
                const City& c = s.get_city(t.city_id());
                if (c.owner() != p && c.capture_ready())
                    out[out_count++] = { ActionType::CaptureCity, t.city_id(), i, 0, true };
            } else if (t.terrain() == TerrainType::Village && t.capture_ready()) {
                out[out_count++] = { ActionType::CaptureCity, -1, i, 0, true };
            }
        }

        // --- Move and Attack ---
        for (int i = 0; i < mtsz; i++) {
            const Tile& t = s.tile_at(i);
            if (!t.has_unit()) continue;
            const Unit& u = s.get_unit(t.unit_id());
            if (u.owner() != p) continue;

            const UnitDef& udef = unit_def(u.type());

            // Recover: only offered to a unit that has neither moved nor
            // attacked this turn (full MP, !has_attacked) and isn't already
            // at max HP. Mirrors the implicit end-of-turn skip-heal so the
            // player can explicitly mark a unit as resting.
            if (u.move_points() == udef.movement && !u.has_attacked()
                && u.hp() < u.max_hp()) {
                out[out_count++] = { ActionType::Recover, i, -1, 0, true };
            }

            if (u.move_points() > 0) {
                int8_t mp_at[MAX_MAP_TILES];
                int    parent[MAX_MAP_TILES];
                reachable_tiles(s, t.unit_id(), mp_at, parent);
                const int msz = s.map_size();
                for (int j = 0; j < mtsz; j++) {
                    if (j == i) continue;
                    if (mp_at[j] < 0) continue;
                    uint32_t pbits = 0;
                    uint8_t  psteps = 0;
                    encode_path_bits(parent, i, j, msz, &pbits, &psteps);
                    out[out_count++] = { ActionType::Move, i, j, mp_at[j], true, pbits, psteps };
                }
            }

            // Attack-after-move rule: a unit that has already moved this turn can
            // only attack if it has ABILITY_DASH (Warrior, Rider). Move spends the
            // full movement budget (apply_move), so move_points < full_movement
            // means the unit has moved this turn.
            bool has_moved = (u.move_points() < udef.movement);
            bool can_dash  = (udef.abilities & ABILITY_DASH) != 0;
            if (!u.has_attacked() && (!has_moved || can_dash)) {
                for (int j = 0; j < mtsz; j++) {
                    const Tile& dt = s.tile_at(j);
                    if (!dt.has_unit()) continue;
                    const Unit& target = s.get_unit(dt.unit_id());
                    if (target.owner() == p) continue;
                    if (!s.is_explored(p, j)) continue;  // units only hidden by fog
                    int dist = chebyshev_distance(i, j, s.map_size());
                    if (dist < 1 || dist > udef.range) continue;
                    out[out_count++] = { ActionType::Attack, i, j, 0, true };
                }
            }
        }
    }

    bool can_end = (s.phase() != GameStateType::UpgradingCity);
    if (can_end)
        out[out_count++] = { ActionType::EndTurn, -1, -1, 0, true };
}
