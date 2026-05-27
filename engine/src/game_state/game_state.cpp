#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "game_state.h"
#include "unit_def.h"
#include "resource_def.h"
#include "tech_def.h"
#include "grid.h"
#include "config.h"
#include "damage_calculator.h"
#include "cassert"
const int NO_WINNER = -1;
const int INVALID   = -1;

// -------------------------------------------------------- movement helpers --

// 8-neighbor direction tables. Indices 0..7 are also the 3-bit encoding used in
// Action::path_bits. Order: N, NE, E, SE, S, SW, W, NW.
extern const int MOVE_DX[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };
extern const int MOVE_DY[8] = { -1, -1, 0, 1, 1,  1,  0, -1 };

// BFS visit order: cardinals first (N, E, S, W) then diagonals (NE, SE, SW, NW).
// Only affects tie-breaking among equal-cost paths — preferring cardinal hops
// gives the unit a more "horizontal/vertical first" feel on the iso grid. Keep
// MOVE_DX/MOVE_DY indexing intact so the 3-bit path_bits encoding stays
// backward-compatible with existing replay files.
static const int BFS_DIR_ORDER[8] = { 0, 2, 4, 6, 1, 3, 5, 7 };

bool tile_passable(const Tile& t, bool has_climbing) {
    if (t.terrain() == TerrainType::Water) return false;
    if (t.terrain() == TerrainType::Mountain && !has_climbing) return false;
    return true;
}

int tile_entry_cost(const Tile& t) {
    if (t.building() == BuildingType::Road) return 0;
    return 1;
}

bool tile_exhausts_movement(const Tile& t) {
    return t.terrain() == TerrainType::Forest || t.terrain() == TerrainType::Mountain;
}

bool adjacent_to_enemy(const GameState& s, int tile, int my_player) {
    const int msz = s.map_size();
    int x, y;
    to_coords(tile, x, y, msz);
    const int DX[] = { 0, 1, 0, -1 };
    const int DY[] = { -1, 0, 1, 0 };
    for (int d = 0; d < 4; d++) {
        int nx = x + DX[d], ny = y + DY[d];
        if (!in_bounds(nx, ny, msz)) continue;
        int ni = to_index(nx, ny, msz);
        const Tile& nt = s.tile_at(ni);
        if (nt.has_unit() && s.get_unit(nt.unit_id()).owner() != my_player) return true;
    }
    return false;
}

// reachable_tiles with optional parent-pointer output. If `out_parent` is non-null,
// out_parent[tile] is the tile we stepped from to reach `tile` along the cheapest
// path (== -1 if unreached, == src for the source tile itself). Enables path
// reconstruction by walking parents back from any reachable tile to the source.
void reachable_tiles(const GameState& s, int unit_id, int8_t out_mp[], int out_parent[]) {
    const int msz  = s.map_size();
    const int mtsz = s.map_tiles();
    const Unit& u  = s.get_unit(unit_id);
    int         p  = u.owner();
    bool climbing  = s.has_tech(p, TechType::Climbing);

    memset(out_mp, -1, mtsz);
    if (out_parent) {
        for (int i = 0; i < mtsz; i++) out_parent[i] = -1;
    }

    struct Node { int tile; int8_t mp; };
    Node queue[MAX_MAP_TILES * 4];
    int head = 0, tail = 0;

    int src = u.tile_index();
    out_mp[src] = (int8_t)u.move_points();
    if (out_parent) out_parent[src] = src;
    queue[tail++] = { src, (int8_t)u.move_points() };

    while (head < tail) {
        int    tile = queue[head].tile;
        int8_t mp   = queue[head].mp;
        head++;

        if (mp <= 0) continue;

        int x, y;
        to_coords(tile, x, y, msz);

        for (int oi = 0; oi < 8; oi++) {
            int d = BFS_DIR_ORDER[oi];
            int nx = x + MOVE_DX[d], ny = y + MOVE_DY[d];
            if (!in_bounds(nx, ny, msz)) continue;
            int ntile = to_index(nx, ny, msz);
            if (!s.is_visible(p, ntile)) continue;  // can't step into fog
            const Tile& nt = s.tile_at(ntile);

            if (!tile_passable(nt, climbing)) continue;
            // Allies are passable as transit nodes; only enemy-occupied tiles
            // block the BFS outright. Ally tiles are stripped from the
            // destination set after the loop so callers see them as unlandable.
            if (nt.has_unit() && s.get_unit(nt.unit_id()).owner() != p) continue;

            int cost = tile_entry_cost(nt);
            if (cost > mp) continue;

            bool   zoc    = adjacent_to_enemy(s, ntile, p);
            int8_t new_mp = (tile_exhausts_movement(nt) || zoc)
                            ? 0
                            : (int8_t)(mp - cost);

            if (new_mp > out_mp[ntile]) {
                out_mp[ntile] = new_mp;
                if (out_parent) out_parent[ntile] = tile;
                queue[tail++] = { ntile, new_mp };
            }
        }
    }

    // Strip ally-occupied tiles from the destination set. They were kept in
    // the BFS so paths could route through them, but they're not landable.
    // (out_parent is left intact so path reconstruction through allies still
    // works.) Skip src — the source tile contains the moving unit itself and
    // is filtered out by callers anyway.
    for (int i = 0; i < mtsz; i++) {
        if (i == src) continue;
        if (out_mp[i] < 0) continue;
        const Tile& ti = s.tile_at(i);
        if (ti.has_unit() && s.get_unit(ti.unit_id()).owner() == p)
            out_mp[i] = -1;
    }
}

// Backwards-compat overload — callers that don't need parent pointers.
void reachable_tiles(const GameState& s, int unit_id, int8_t out_mp[]) {
    reachable_tiles(s, unit_id, out_mp, nullptr);
}

// Encode the path from src..dst (using `parent` from reachable_tiles) into a packed
// uint32 of 3-bit direction steps. Returns true on success. On failure (unreachable
// or path longer than MAX_MOVE_PATH_STEPS) sets both outputs to 0 and returns false.
bool encode_path_bits(const int parent[], int src, int dst, int msz,
                      uint32_t* out_bits, uint8_t* out_steps) {
    *out_bits  = 0;
    *out_steps = 0;
    if (src == dst) return true;             // zero-step "path"
    if (parent[dst] < 0) return false;       // unreachable

    // Walk back from dst to src, recording each step's child-tile (we'll need (parent → child)
    // to derive direction). Collect in reverse, then encode in forward order.
    int back[MAX_MOVE_PATH_STEPS + 1];
    int n = 0;
    int cur = dst;
    while (cur != src) {
        if (n >= MAX_MOVE_PATH_STEPS) return false;  // path too long
        back[n++] = cur;
        cur = parent[cur];
        if (cur < 0) return false;
    }
    // back[n-1..0] = first..last hop's destination; src is the implicit start.
    int prev = src;
    for (int i = 0; i < n; i++) {
        int child = back[n - 1 - i];
        int px, py, cx, cy;
        to_coords(prev,  px, py, msz);
        to_coords(child, cx, cy, msz);
        int dx = cx - px, dy = cy - py;
        int dir = -1;
        for (int d = 0; d < 8; d++)
            if (MOVE_DX[d] == dx && MOVE_DY[d] == dy) { dir = d; break; }
        if (dir < 0) return false;
        *out_bits |= ((uint32_t)dir & 0x7u) << (3 * i);
        prev = child;
    }
    *out_steps = (uint8_t)n;
    return true;
}

// Decode a packed path back into a sequence of tile indices, starting from `start`.
// Writes [start, step0_tile, step1_tile, ..., last_step_tile] into out_tiles, and
// the number of tiles written (steps + 1) into *out_count.
void decode_path_bits(int start, uint32_t bits, uint8_t steps, int msz,
                      int out_tiles[MAX_MOVE_PATH_STEPS + 1], int* out_count) {
    int cur = start;
    int n = 0;
    out_tiles[n++] = cur;
    for (int i = 0; i < steps; i++) {
        int dir = (int)((bits >> (3 * i)) & 0x7u);
        int x, y;
        to_coords(cur, x, y, msz);
        int nx = x + MOVE_DX[dir], ny = y + MOVE_DY[dir];
        if (!in_bounds(nx, ny, msz)) break;
        cur = to_index(nx, ny, msz);
        out_tiles[n++] = cur;
    }
    *out_count = n;
}

void reveal_around(GameState& s, int player, int tile_index, int radius) {
    const int msz = s.map_size();
    int x, y;
    to_coords(tile_index, x, y, msz);
    for (int dy = -radius; dy <= radius; dy++)
        for (int dx = -radius; dx <= radius; dx++)
            if (in_bounds(x + dx, y + dy, msz))
                s.set_visible(player, to_index(x + dx, y + dy, msz));
}

// --------------------------------------------------------- public interface --

int GameState::winner() const {
    for (int p = 0; p < MAX_PLAYERS; p++) {
        int cap_id = capital_city[p];
        if (cap_id == -1) continue;
        int captor = cities[cap_id].owner();
        if (captor != p) return captor;
    }
    return NO_WINNER;
}
bool GameState::is_terminal() const { return winner() != NO_WINNER; }
int  GameState::current_player() const { return cur_player; }

void GameState::set_visible(int player, int tile) {
    // Always re-mark visible (rebuild_visibility wipes it then expects each
    // reveal_around call to restore it). Only run the corner bonus on the
    // first-ever reveal.
    bool was_explored = players[player].is_visible(tile);
    players[player].set_visible(tile);  // sets both explored and visible bits
    if (was_explored) return;

    if (!is_lighthouse(tile, _map_size)) return;
    int cap = capital_city[player];
    if (cap < 0) return;
    // Only credit the bonus while the capital is still ours. Once captured,
    // the pointer is stale and would feed population to the enemy.
    if (cities[cap].owner() != player) return;
    cities[cap].add_population(1);
}

void GameState::move_unit(int unit_id, int dest_tile) {
    Unit& u = units[unit_id];
    int src = u.tile_index();
    if (src == dest_tile) return;
    int radius = (map[dest_tile].terrain() == TerrainType::Mountain) ? 2 : 1;
    reveal_around(*this, u.owner(), dest_tile, radius);
    // Ally pass-through: paths from reachable_tiles can route through friendly
    // tiles as transit nodes. Reveal as the unit "passes over" them but don't
    // physically place — the unit advances on the next call.
    if (map[dest_tile].has_unit() && units[map[dest_tile].unit_id()].owner() == u.owner())
        return;
    map[src].remove_unit();
    map[dest_tile].place_unit(unit_id);
    u.set_tile(dest_tile);
}

int GameState::spawn_unit(UnitType type, int owner, int tile_index, int city_id) {
    if (unit_count >= MAX_MAP_TILES) return INVALID;
    // If the destination tile already has a unit, push it out before placing.
    // This is the entry point for super-unit / Ruin / Polytaur-style spawns;
    // ordinary city training never reaches here (legal_actions guards on
    // !tile.has_unit()).
    if (map[tile_index].has_unit())
        push_unit_from(map[tile_index].unit_id(), tile_index, owner);
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
    if (city_id >= 0) {
        u.set_city_id(city_id);
        cities[city_id].add_unit();
    }
    return id;
}

int GameState::spawn_city(int tile_index, int owner, bool is_capital) {
    if (city_count >= MAX_MAP_TILES) return INVALID;
    int id = city_count++;
    City& c = cities[id];
    c.set_tile(tile_index);
    c.set_owner(owner);
    c.set_capital(is_capital);
    if (is_capital) capital_city[owner] = id;
    map[tile_index].place_city(id);
    claim_border_for_city(id);
    return id;
}

void GameState::claim_border_for_city(int city_id) {
    const City& city = cities[city_id];
    int cx, cy;
    to_coords(city.tile_index(), cx, cy, _map_size);
    int r = city.border_radius();

    for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++) {
            int bx = cx + dx, by = cy + dy;
            if (!in_bounds(bx, by, _map_size)) continue;
            int bidx = to_index(bx, by, _map_size);
            if (map[bidx].border_city_id() == -1)
                map[bidx].set_border_city(city_id);
        }
    }
}

void GameState::print_map() const {
    for (int y = 0; y < _map_size; y++) {
        for (int x = 0; x < _map_size; x++) {
            const Tile& t = map[to_index(x, y, _map_size)];
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


// ----------------------------------------------------------- explorer logic --
//
// Walks up to cfg::explorer::STEPS tiles from `start_tile`. At each step:
//   1. BFS up to SCAN_RADIUS over passable terrain (real map; ignores fog).
//      Records distance and which immediate-neighbour first-step leads to
//      each reachable tile.
//   2. Find the minimum distance to any fog tile. If none → step 4a.
//   3. For each first-step direction, score the BEST destination fog tile
//      reached through it at that minimum distance:
//          score = clear * CLEAR_WEIGHT
//                + (lighthouse_in_dest_3x3 ? LIGHTHOUSE_BONUS : 0)
//                - history_penalty(first_step)
//      where `clear` is the count of fog tiles in the destination's 3×3
//      (capped at FOG_REVEAL_CAP — the mountain-extended ring is excluded
//      by construction since we only count the radius-1 reveal).
//   4. Highest score wins; ties broken by a per-call deterministic RNG.
//   4a. No reachable fog: every passable neighbour scored equally except
//       tiles in the recent-history buffer, which get a small penalty.
//
// Lighthouses are the four map corners. They start fogged, are passable
// terrain like any other, and prioritising them falls naturally out of
// the LIGHTHOUSE_BONUS in the destination score.

namespace {

constexpr int DIR_DX[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };
constexpr int DIR_DY[8] = {-1,-1, 0, 1, 1,  1,  0, -1 };

inline bool explorer_passable(const Tile& t, bool climbing) {
    TerrainType ter = t.terrain();
    if (ter == TerrainType::Water) return false;
    if (ter == TerrainType::Mountain && !climbing) return false;
    return true;
}

// Cheap xorshift; reseeded per explore() call from state so MCTS rollouts
// stay deterministic without needing a stored RNG.
inline uint32_t xorshift32(uint32_t& s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return s;
}

inline uint32_t explorer_seed(int start_tile, int player, int turn) {
    uint32_t s = (uint32_t)start_tile * 73856093u
               ^ (uint32_t)player     * 19349663u
               ^ (uint32_t)turn       * 83492791u;
    return s ? s : 0xC0FFEEu;
}

// Count unexplored tiles in the 3×3 around `dest` (the radius-1 reveal),
// capped at FOG_REVEAL_CAP. Lighthouse priority is handled separately
// per-direction via dir_has_lighthouse() so the bonus can pull the path
// from farther than 1 tile away.
int dest_clear_count(const GameState& s, int player, int dest, int sz) {
    int dx, dy; to_coords(dest, dx, dy, sz);
    int clear = 0;
    for (int yy = -1; yy <= 1; yy++)
        for (int xx = -1; xx <= 1; xx++) {
            int fx = dx + xx, fy = dy + yy;
            if (!in_bounds(fx, fy, sz)) continue;
            int idx = to_index(fx, fy, sz);
            if (s.is_visible(player, idx)) continue;
            clear++;
        }
    return clear > cfg::explorer::FOG_REVEAL_CAP ? cfg::explorer::FOG_REVEAL_CAP : clear;
}

// For each first-step direction, true iff stepping that way leads — within
// SCAN_RADIUS — to an unexplored lighthouse via the recorded BFS shortest path.
// Indexed by `neighbour_dir[tile]`.
inline void mark_lighthouse_directions(const GameState& s, int player, int sz,
                                       const int8_t* dist, const int* first_step,
                                       const int* neighbour_dir,
                                       bool out_dir[8])
{
    for (int i = 0; i < 8; i++) out_dir[i] = false;
    int corners[4] = {
        to_index(0,      0,      sz),
        to_index(sz - 1, 0,      sz),
        to_index(0,      sz - 1, sz),
        to_index(sz - 1, sz - 1, sz),
    };
    for (int c : corners) {
        if (s.is_visible(player, c)) continue;
        if (dist[c] < 1 || dist[c] > cfg::explorer::SCAN_RADIUS) continue;
        int fs = first_step[c];
        if (fs < 0) continue;
        int idx = neighbour_dir[fs];
        if (idx < 0) continue;
        out_dir[idx] = true;
    }
}

inline int history_penalty(int tile, const int* hist, int hist_n) {
    for (int i = 0; i < hist_n; i++)
        if (hist[i] == tile) return cfg::explorer::BACKTRACK_PENALTY;
    return 0;
}

// Returns chosen tile to step onto (one of `cur`'s 8 passable neighbours),
// or -1 if no legal step exists (explorer goes poof).
int choose_explorer_step(const GameState& s, int player, int cur,
                         bool climbing, const int* hist, int hist_n,
                         uint32_t& rng)
{
    const int sz   = s.map_size();
    const int mtsz = s.map_tiles();

    int8_t dist[MAX_MAP_TILES];
    int    first_step[MAX_MAP_TILES];
    std::fill(dist,       dist       + mtsz, (int8_t)-1);
    std::fill(first_step, first_step + mtsz, -1);

    int queue[MAX_MAP_TILES];
    int head = 0, tail = 0;
    dist[cur] = 0;
    queue[tail++] = cur;

    while (head < tail) {
        int t = queue[head++];
        if (dist[t] >= cfg::explorer::SCAN_RADIUS) continue;
        int tx, ty; to_coords(t, tx, ty, sz);

        // Randomise neighbour order so first_step assignment is unbiased
        // when multiple equal-distance paths exist.
        int order[8] = {0,1,2,3,4,5,6,7};
        for (int i = 7; i > 0; i--) {
            int j = (int)(xorshift32(rng) % (uint32_t)(i + 1));
            std::swap(order[i], order[j]);
        }

        for (int k = 0; k < 8; k++) {
            int d = order[k];
            int nx = tx + DIR_DX[d], ny = ty + DIR_DY[d];
            if (!in_bounds(nx, ny, sz)) continue;
            int ni = to_index(nx, ny, sz);
            if (dist[ni] >= 0) continue;
            if (!explorer_passable(s.tile_at(ni), climbing)) continue;
            dist[ni] = dist[t] + 1;
            first_step[ni] = (t == cur) ? ni : first_step[t];
            queue[tail++] = ni;
        }
    }

    // Find minimum distance to any reachable fog tile.
    int min_d = INT_MAX;
    for (int i = 0; i < mtsz; i++) {
        if (dist[i] < 1) continue;
        if (s.is_visible(player, i)) continue;
        if (dist[i] < min_d) min_d = dist[i];
    }

    // Enumerate cur's immediate passable neighbours once — we score by
    // first-step direction, so the candidate set is at most 8.
    int cx, cy; to_coords(cur, cx, cy, sz);
    int  fs_tile[8];
    int  fs_score[8];
    bool fs_scored[8] = { false };
    int  n_neighbours = 0;
    int  neighbour_dir[MAX_MAP_TILES]; // tile -> index in fs_tile[]
    std::fill(neighbour_dir, neighbour_dir + mtsz, -1);
    for (int d = 0; d < 8; d++) {
        int nx = cx + DIR_DX[d], ny = cy + DIR_DY[d];
        if (!in_bounds(nx, ny, sz)) continue;
        int ni = to_index(nx, ny, sz);
        if (!explorer_passable(s.tile_at(ni), climbing)) continue;
        neighbour_dir[ni] = n_neighbours;
        fs_tile[n_neighbours] = ni;
        n_neighbours++;
    }
    if (n_neighbours == 0) return -1;

    // Lighthouse magnetism: any direction leading (within SCAN_RADIUS) to an
    // unexplored corner is boosted, even if no fog tile sits in the dest's 3×3.
    bool dir_lh[8];
    mark_lighthouse_directions(s, player, sz, dist, first_step, neighbour_dir, dir_lh);

    // ---- Step 4a: no reachable fog → equal scores; lighthouse bonus & history only ----
    if (min_d == INT_MAX) {
        int best = INT_MIN, ties[8], n_ties = 0;
        for (int i = 0; i < n_neighbours; i++) {
            int sc = -history_penalty(fs_tile[i], hist, hist_n)
                   + (dir_lh[i] ? cfg::explorer::LIGHTHOUSE_BONUS : 0);
            if (sc > best)      { best = sc; n_ties = 0; ties[n_ties++] = fs_tile[i]; }
            else if (sc == best) ties[n_ties++] = fs_tile[i];
        }
        return ties[xorshift32(rng) % (uint32_t)n_ties];
    }

    // ---- Steps 3 + 4: score each first-step direction by its best fog tile ----
    for (int i = 0; i < mtsz; i++) {
        if (dist[i] != min_d) continue;
        if (s.is_visible(player, i)) continue;
        int fs = first_step[i];
        if (fs < 0) continue;
        int idx = neighbour_dir[fs];
        if (idx < 0) continue;
        int sc = dest_clear_count(s, player, i, sz) * cfg::explorer::CLEAR_WEIGHT
               + (dir_lh[idx] ? cfg::explorer::LIGHTHOUSE_BONUS : 0)
               - history_penalty(fs, hist, hist_n);
        if (!fs_scored[idx] || sc > fs_score[idx]) {
            fs_scored[idx] = true;
            fs_score[idx]  = sc;
        }
    }

    int best = INT_MIN, ties[8], n_ties = 0;
    for (int i = 0; i < n_neighbours; i++) {
        if (!fs_scored[i]) continue;
        if (fs_score[i] > best)      { best = fs_score[i]; n_ties = 0; ties[n_ties++] = fs_tile[i]; }
        else if (fs_score[i] == best) ties[n_ties++] = fs_tile[i];
    }
    if (n_ties == 0) return -1;
    return ties[xorshift32(rng) % (uint32_t)n_ties];
}

}  // namespace

void GameState::explore(int start_tile, int player,
                        int* out_path, int* out_count) {
    if (out_count) *out_count = 0;
    if (start_tile < 0 || start_tile >= map_tiles()) return;
    if (out_path) { out_path[0] = start_tile; if (out_count) *out_count = 1; }

    reveal_around(*this, player, start_tile, 1);

    const bool climbing = has_tech(player, TechType::Climbing);
    uint32_t   rng      = explorer_seed(start_tile, player, turn);

    int history[cfg::explorer::HISTORY_LEN];
    int hist_n = 0;
    int cur    = start_tile;

    for (int step = 0; step < cfg::explorer::STEPS; step++) {
        int next = choose_explorer_step(*this, player, cur, climbing,
                                        history, hist_n, rng);
        if (next < 0) break;  // poof

        // Sliding window: drop oldest, append cur before moving.
        if (hist_n == cfg::explorer::HISTORY_LEN) {
            for (int i = 1; i < cfg::explorer::HISTORY_LEN; i++)
                history[i - 1] = history[i];
            history[cfg::explorer::HISTORY_LEN - 1] = cur;
        } else {
            history[hist_n++] = cur;
        }

        cur = next;
        reveal_around(*this, player, cur, 1);
        if (out_path && *out_count < 16) out_path[(*out_count)++] = cur;
    }
}

float GameState::get_defense_bonus(Player& p, Tile& t) {
    if (!t.has_unit()) {
        assert(false);
        return 1.0f;
    }
    if (!this->units[t.unit_id()].has_ability(ABILITY_FORTIFY)) return 1.0f;
    TerrainType terrain = t.terrain();
    float total_defense = 1.0f;

    for (TechType tech : p.get_techs()) {
        const auto& terrains = tech_def(tech).defense_terrains;
        if (std::find(terrains.begin(), terrains.end(), terrain) != terrains.end()) {
            total_defense = 1.5f;
            break;
        }
    }

    if (t.has_city()) {
        total_defense += 0.5f;
        if (this->cities[t.city_id()].has_walls()) total_defense += 0.5f;
    }
    return total_defense;
}

// ----------------------------------------------------------- invariants --

namespace {

#define INV(cond, ...) do {                                                 \
    if (!(cond)) {                                                          \
        std::fprintf(stderr, "[check_invariants] " __VA_ARGS__);            \
        std::fprintf(stderr, "\n");                                         \
        return false;                                                       \
    }                                                                       \
} while (0)

inline bool valid_player(int p)            { return p >= 0 && p < MAX_PLAYERS; }
inline bool valid_player_or_none(int p)    { return p == -1 || valid_player(p); }
inline bool valid_unit_type(UnitType t)    { return t > UnitType::None && t < UnitType::Count; }

}  // namespace

bool GameState::check_invariants() const {
    const int mtsz = map_tiles();
    INV(_map_size > 0 && _map_size <= MAX_MAP_SIZE,
        "map size %d out of bounds", _map_size);
    INV(cur_player >= 0 && cur_player < MAX_PLAYERS,
        "cur_player %d invalid", cur_player);
    INV(turn >= 0, "turn %d negative", turn);
    INV(unit_count >= 0 && unit_count <= MAX_MAP_TILES,
        "unit_count %d out of range", unit_count);
    INV(city_count >= 0 && city_count <= MAX_MAP_TILES,
        "city_count %d out of range", city_count);

    // ---- Players ----
    for (int p = 0; p < MAX_PLAYERS; p++) {
        INV(players[p].stars >= 0, "player %d has negative stars (%d)", p, players[p].stars);
        int cap = capital_city[p];
        INV(cap == -1 || (cap >= 0 && cap < city_count),
            "player %d capital_city=%d out of range", p, cap);
        if (cap >= 0) {
            const City& c = cities[cap];
            INV(c.is_capital(), "player %d capital_city=%d but city.is_capital=false", p, cap);
            // capital may have been captured: owner may differ from p; winner()
            // logic depends on that exact case, so don't require c.owner()==p here.
        }
    }

    // ---- Cities ----
    int city_unit_counts[MAX_MAP_TILES] = {};
    for (int i = 0; i < city_count; i++) {
        const City& c = cities[i];
        INV(valid_player(c.owner()), "city %d has invalid owner %d", i, c.owner());
        INV(c.tile_index() >= 0 && c.tile_index() < mtsz,
            "city %d has invalid tile_index %d", i, c.tile_index());
        INV(c.level() >= 1, "city %d level=%d < 1", i, c.level());
        INV(c.units_owned() >= 0 && c.units_owned() <= c.unit_capacity(),
            "city %d units_owned=%d exceeds capacity %d",
            i, c.units_owned(), c.unit_capacity());
        INV(c.population() >= 0, "city %d population=%d < 0", i, c.population());
        INV(c.border_radius() >= 1, "city %d border_radius=%d < 1", i, c.border_radius());

        const Tile& t = map[c.tile_index()];
        INV(t.city_id() == i,
            "city %d at tile %d but tile.city_id=%d", i, c.tile_index(), t.city_id());
    }

    // Each capital_city[p] backreference should be the unique capital owned by p
    // when the capital still belongs to its founder. Detect duplicate capitals.
    int found_capital[MAX_PLAYERS] = { -1, -1 };
    for (int i = 0; i < city_count; i++) {
        if (!cities[i].is_capital()) continue;
        // A captured capital still flags is_capital=true; map ownership to
        // its original founder via capital_city[].
        for (int p = 0; p < MAX_PLAYERS; p++) {
            if (capital_city[p] == i) {
                INV(found_capital[p] == -1,
                    "player %d has two cities marked capital (%d and %d)",
                    p, found_capital[p], i);
                found_capital[p] = i;
            }
        }
    }

    // ---- Units ----
    for (int i = 0; i < unit_count; i++) {
        const Unit& u = units[i];
        if (!u.is_alive()) continue;  // dead unit slots are sentinels, skip
        INV(valid_player(u.owner()), "unit %d invalid owner %d", i, u.owner());
        INV(valid_unit_type(u.type()), "unit %d invalid type %d", i, (int)u.type());
        INV(u.tile_index() >= 0 && u.tile_index() < mtsz,
            "unit %d invalid tile_index %d", i, u.tile_index());
        INV(u.max_hp() > 0, "unit %d max_hp=%d", i, u.max_hp());
        INV(u.hp() > 0 && u.hp() <= u.max_hp(),
            "unit %d hp=%d out of [1, %d]", i, u.hp(), u.max_hp());
        INV(u.move_points() >= 0, "unit %d move_points=%d < 0", i, u.move_points());

        const Tile& t = map[u.tile_index()];
        INV(t.unit_id() == i,
            "unit %d at tile %d but tile.unit_id=%d", i, u.tile_index(), t.unit_id());

        if (u.city_id() >= 0) {
            INV(u.city_id() < city_count,
                "unit %d home city_id=%d out of range", i, u.city_id());
            INV(cities[u.city_id()].owner() == u.owner(),
                "unit %d (owner %d) homed in city %d (owner %d)",
                i, u.owner(), u.city_id(), cities[u.city_id()].owner());
            city_unit_counts[u.city_id()]++;
        }
    }

    // ---- City units_owned matches alive units pointing to it ----
    for (int i = 0; i < city_count; i++) {
        INV(cities[i].units_owned() == city_unit_counts[i],
            "city %d units_owned=%d but %d alive units claim it",
            i, cities[i].units_owned(), city_unit_counts[i]);
    }

    // ---- Tile cross-refs ----
    for (int i = 0; i < mtsz; i++) {
        const Tile& t = map[i];
        if (t.has_city()) {
            INV(t.city_id() >= 0 && t.city_id() < city_count,
                "tile %d city_id=%d out of range", i, t.city_id());
            INV(cities[t.city_id()].tile_index() == i,
                "tile %d → city %d but city.tile_index=%d",
                i, t.city_id(), cities[t.city_id()].tile_index());
        }
        if (t.has_unit()) {
            INV(t.unit_id() >= 0 && t.unit_id() < unit_count,
                "tile %d unit_id=%d out of range", i, t.unit_id());
            const Unit& u = units[t.unit_id()];
            INV(u.is_alive(),
                "tile %d points to dead unit %d", i, t.unit_id());
            INV(u.tile_index() == i,
                "tile %d → unit %d but unit.tile_index=%d",
                i, t.unit_id(), u.tile_index());
        }
        int bcid = t.border_city_id();
        INV(bcid == -1 || (bcid >= 0 && bcid < city_count),
            "tile %d border_city_id=%d out of range", i, bcid);
    }

    return true;
}

#undef INV

