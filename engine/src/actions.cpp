#include "game_state.h"
#include "unit_def.h"
#include "grid.h"

// Helpers use only the public GameState API — no private access needed.

static void reveal_around(GameState& s, int player, int tile_index) {
    int x, y;
    to_coords(tile_index, x, y);
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
            if (in_bounds(x + dx, y + dy))
                s.set_explored(player, to_index(x + dx, y + dy));
}

static void rebuild_visibility(GameState& s, int player) {
    s.clear_visible(player);
    for (int i = 0; i < MAP_TILES; i++) {
        const Tile& t = s.tile_at(i);
        if (t.has_unit() && s.get_unit(t.unit_id()).owner() == player)
            reveal_around(s, player, i);
        if (t.has_city() && s.get_city(t.city_id()).owner() == player)
            reveal_around(s, player, i);
    }
}

// --------------------------------------------------------- public interface --

void legal_actions(const GameState& s, Action out[], int& out_count) {
    out_count = 0;
    int p = s.current_player();

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
                int bidx = to_index(bx, by);
                const Tile& bt = s.tile_at(bidx);
                ResourceType res = bt.resource();
                if (res == ResourceType::None) continue;

                // Tech requirements
                if (res == ResourceType::Metal && !s.has_tech(p, TechType::Mining))  continue;
                if (res == ResourceType::Game  && !s.has_tech(p, TechType::Hunting)) continue;

                // Star cost
                int cost = (res == ResourceType::Metal) ? 3 : 2;
                if (s.get_stars(p) < cost) continue;

                // Not blocked by enemy unit
                if (bt.has_unit() && s.get_unit(bt.unit_id()).owner() != p) continue;

                out[out_count++] = { ActionType::HarvestResource, i, bidx, (int)res };
            }
        }
    }

    // --- EndTurn (always legal) ---
    out[out_count++] = { ActionType::EndTurn, -1, -1, 0 };
}

// apply_action is a friend of GameState — direct private access is intentional here.
GameState apply_action(GameState s, Action a) {
    switch (a.type) {

        case ActionType::EndTurn: {
            // Switch to next player
            s.cur_player = 1 - s.cur_player;
            int next = s.cur_player;

            // Collect income for the player whose turn is starting
            for (int i = 0; i < MAP_TILES; i++) {
                const Tile& t = s.tile_at(i);
                if (t.has_city() && s.get_city(t.city_id()).owner() == next)
                    s.stars[next] += s.get_city(t.city_id()).stars_per_turn();
            }

            for (int i = 0; i < MAP_TILES; i++) {
                const Tile& t = s.tile_at(i);
                if (t.has_unit()) {
                    Unit& u = s.units[t.unit_id()];
                    if (u.owner() == next)
                        u.reset_turn(unit_def(u.type()).movement);
                }
            }

            rebuild_visibility(s, next);
            s.turn++;
            return s;
        }

        case ActionType::HarvestResource: {
            int p = s.cur_player;
            auto res = (ResourceType)a.param;
            int cost = (res == ResourceType::Metal) ? 3 : 2;

            s.stars[p] -= cost;

            if (res == ResourceType::Metal) {
                s.stars[p] += 2;                      // +2★ immediately
            } else {
                int cid = s.tile_at(a.from).city_id();
                s.cities[cid].add_population(1);      // +1 pop to city
            }

            s.tile_at(a.to).set_resource(ResourceType::None);  // one-time, tile cleared
            return s;
        }

        default:
            return s;
    }
}
