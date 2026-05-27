#include "game_state.h"
#include "unit.h"
#include "grid.h"
#include "types.h"

extern const int MOVE_DX[8];
extern const int MOVE_DY[8];

bool tile_passable(const Tile& t, bool has_climbing);

namespace {

int sgn(int x) { return (x > 0) - (x < 0); }

int dir_from_signed(int sx, int sy) {
    for (int d = 0; d < 8; d++)
        if (MOVE_DX[d] == sx && MOVE_DY[d] == sy) return d;
    return -1;
}

// Direction (0..7) from `tile` toward the map's centre tile. Spec: exact
// centre falls back to South (dir 4). For even-sized maps there is no single
// centre tile — use the floor-of-centre as the target.
int direction_toward_centre(int tile, int msz) {
    int x = tile % msz, y = tile / msz;
    int mid = (msz - 1) / 2;
    int dx = mid - x, dy = mid - y;
    if (dx == 0 && dy == 0) return 4;  // S
    int d = dir_from_signed(sgn(dx), sgn(dy));
    return d >= 0 ? d : 4;
}

// Rules 1-4 from the spec, in order.
int push_direction(const Unit& victim, int spawning_player,
                   int spawn_tile, int msz) {
    int last = victim.last_dir();
    bool is_ranged = victim.has_ability(ABILITY_RANGED);
    if (last >= 0) {
        // Rule 3: ranged units use last move/attack direction regardless of side.
        if (is_ranged) return last;
        // Rule 1: friendly (to the spawner) pushed same direction.
        if (victim.owner() == spawning_player) return last;
        // Rule 2: enemy pushed opposite direction.
        return (last + 4) & 7;
    }
    // Rule 4: no prior action — push toward map centre.
    return direction_toward_centre(spawn_tile, msz);
}

}  // namespace

void GameState::push_unit_from(int unit_id, int spawn_tile, int spawning_player) {
    Unit& u = units[unit_id];
    const int msz = map_size();
    const int base_dir = push_direction(u, spawning_player, spawn_tile, msz);
    const int sx = spawn_tile % msz, sy = spawn_tile / msz;
    const bool climbing = has_tech(u.owner(), TechType::Climbing);

    // Alternating fallback ring: dir, ccw1, cw1, ccw2, cw2, ccw3, cw3, ccw4.
    // (ccw4 == cw4 mod 8, so 8 unique directions in total.)
    static const int OFFSETS[8] = { 0, -1, +1, -2, +2, -3, +3, +4 };
    for (int i = 0; i < 8; i++) {
        int dir = (base_dir + OFFSETS[i] + 8) & 7;
        int nx = sx + MOVE_DX[dir], ny = sy + MOVE_DY[dir];
        if (nx < 0 || ny < 0 || nx >= msz || ny >= msz) continue;
        int n_idx = ny * msz + nx;
        const Tile& nt = map[n_idx];
        if (nt.has_unit()) continue;
        if (!tile_passable(nt, climbing)) continue;
        move_unit(unit_id, n_idx);
        return;
    }

    // No free / passable tile — remove from the game (no kill, no attack credit).
    if (u.city_id() >= 0) cities[u.city_id()].remove_unit();
    map[spawn_tile].remove_unit();
    u.set_hp(0);
}
