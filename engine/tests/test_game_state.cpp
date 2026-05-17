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

int main() {
    test_new_game();
    test_invariants_fresh_map();
    test_invariants_random_playout();
    printf("All game_state tests passed.\n");

    printf("\nBlank map:\n");
    GameState s;
    s.print_map();

    return 0;
}
