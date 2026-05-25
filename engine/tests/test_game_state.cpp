#include <cassert>
#include <cstdint>
#include <cstdio>
#include "game_state.h"
#include "mapgen.h"

static void test_new_game() {
    GameState s;
    assert(s.current_player() == 0);
    assert(s.winner() == -1);
    assert(!s.is_terminal());
    assert(s.check_invariants());
}

// Fresh generated game must satisfy every invariant before any action runs.
static void test_invariants_fresh_map() {
    for (uint64_t seed = 1; seed <= 8; seed++) {
        MapGenParams p = MapGen::drylands_defaults();
        p.seed = seed;
        GameState s = MapGen(p).generate().state;
        if (!s.check_invariants()) {
            std::fprintf(stderr, "fresh map invariants failed (seed=%llu)\n",
                         (unsigned long long)seed);
            assert(false);
        }
    }
}

// Random-playout fuzz: every applied action must leave the state consistent.
// We use a deterministic LCG so failures reproduce.
static uint32_t lcg(uint32_t& s) { s = s * 1664525u + 1013904223u; return s; }

static void test_invariants_random_playout() {
    for (uint64_t seed = 1; seed <= 30; seed++) {
        MapGenParams p = MapGen::drylands_defaults();
        p.seed = seed;
        GameState s = MapGen(p).generate().state;
        assert(s.check_invariants());

        uint32_t rng = (uint32_t)(seed * 2654435761u);
        constexpr int MAX_STEPS = 2000;
        for (int step = 0; step < MAX_STEPS && !s.is_terminal(); step++) {
            Action acts[256];
            int n = 0;
            s.legal_actions(acts, n);
            if (n == 0) break;

            // Prefer affordable actions but tolerate state with none.
            int affordable_idx[256]; int n_aff = 0;
            for (int i = 0; i < n; i++)
                if (acts[i].affordable) affordable_idx[n_aff++] = i;
            int pick = n_aff > 0
                ? affordable_idx[lcg(rng) % (uint32_t)n_aff]
                : (int)(lcg(rng) % (uint32_t)n);

            s = s.apply_action(acts[pick]);
            if (!s.check_invariants()) {
                std::fprintf(stderr, "invariant broke at seed=%llu step=%d "
                                     "(action type=%d from=%d to=%d param=%d)\n",
                             (unsigned long long)seed, step,
                             (int)acts[pick].type, acts[pick].from,
                             acts[pick].to, acts[pick].param);
                assert(false);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Forced-spawn / push tests
// ---------------------------------------------------------------------------

// Bare-bones GameState with an N×N Field map (default terrain) — no MapGen,
// so no cities/units exist until the test spawns them.
static GameState make_field_map(int msz) {
    GameState s;
    s.set_map_size(msz);
    return s;
}

// 1. spawn_unit onto an occupied tile pushes the existing unit out without
//    killing it — provided at least one neighbour is free/passable.
static void test_push_keeps_unit_alive() {
    GameState s = make_field_map(5);
    int centre = to_index(2, 2, 5);
    int uid_a = s.spawn_unit(UnitType::Warrior, 0, centre);
    int uid_b = s.spawn_unit(UnitType::Warrior, 0, centre);
    assert(uid_a >= 0 && uid_b >= 0);
    assert(s.get_unit(uid_a).is_alive());
    assert(s.get_unit(uid_a).tile_index() != centre);
    assert(s.tile_at(centre).unit_id() == uid_b);
}

// 2. A freshly-spawned unit has no last_dir, so the push direction falls
//    through to "toward the centre of the map". On a 7×7 map the centre is
//    (3,3); a unit pushed off (0,0) should land at (1,1) — one tile SE.
static void test_push_toward_centre_when_just_spawned() {
    GameState s = make_field_map(7);
    int corner = to_index(0, 0, 7);
    int uid_a  = s.spawn_unit(UnitType::Warrior, 0, corner);
    s.spawn_unit(UnitType::Warrior, 0, corner);
    assert(s.get_unit(uid_a).is_alive());
    assert(s.get_unit(uid_a).tile_index() == to_index(1, 1, 7));
}

// 3. When the natural push direction is blocked, the fallback ring alternates
//    CCW then CW. Setup: unit on the exact centre of a 5×5 (push dir = S per
//    spec). Block S and SE so the next free slot is SW (the CW-1 offset).
//    Expected pick order: S (blocked), CCW1=SE (blocked), CW1=SW (free).
static void test_push_alternating_ccw_cw_fallback() {
    GameState s = make_field_map(5);
    int centre = to_index(2, 2, 5);
    int south  = to_index(2, 3, 5);   // dir 4 (S)
    int se     = to_index(3, 3, 5);   // dir 3 (SE) — CCW1 from S
    int sw     = to_index(1, 3, 5);   // dir 5 (SW) — CW1 from S

    int uid_centre = s.spawn_unit(UnitType::Warrior, 0, centre);
    s.spawn_unit(UnitType::Warrior, 0, south);
    s.spawn_unit(UnitType::Warrior, 0, se);

    s.spawn_unit(UnitType::Warrior, 0, centre);  // triggers the push

    assert(s.get_unit(uid_centre).is_alive());
    assert(s.get_unit(uid_centre).tile_index() == sw);
}

// 4. When every neighbour is occupied, the pushed unit is removed from the
//    game entirely (hp = 0). Both friendly-to-spawner and enemy-to-spawner
//    victims hit the same removal path.
static void test_push_removes_unit_when_no_free_tile_friendly() {
    GameState s = make_field_map(3);
    int target = to_index(1, 1, 3);
    for (int i = 0; i < 9; i++)
        assert(s.spawn_unit(UnitType::Warrior, 0, i) >= 0);
    int victim_uid = s.tile_at(target).unit_id();
    assert(victim_uid >= 0);
    s.spawn_unit(UnitType::Warrior, 0, target);  // friendly forced spawn
    assert(!s.get_unit(victim_uid).is_alive());
}

static void test_push_removes_unit_when_no_free_tile_enemy() {
    GameState s = make_field_map(3);
    int target = to_index(1, 1, 3);
    for (int i = 0; i < 9; i++)
        assert(s.spawn_unit(UnitType::Warrior, 0, i) >= 0);
    int victim_uid = s.tile_at(target).unit_id();
    assert(victim_uid >= 0);
    s.spawn_unit(UnitType::Warrior, 1, target);  // enemy forced spawn
    assert(!s.get_unit(victim_uid).is_alive());
}

int main() {
    test_new_game();
    test_invariants_fresh_map();
    test_invariants_random_playout();
    test_push_keeps_unit_alive();
    test_push_toward_centre_when_just_spawned();
    test_push_alternating_ccw_cw_fallback();
    test_push_removes_unit_when_no_free_tile_friendly();
    test_push_removes_unit_when_no_free_tile_enemy();
    printf("All game_state tests passed.\n");

    printf("\nBlank map:\n");
    GameState s;
    s.print_map();

    return 0;
}
