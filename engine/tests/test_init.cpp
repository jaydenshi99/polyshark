#include <cassert>
#include <cstdio>
#include "init.h"
#include "unit_def.h"

int main() {
    GameState s = make_game();

    // Basic state
    assert(s.current_player() == 0);
    assert(s.winner() == -1);
    assert(!s.is_terminal());

    // Capitals exist and are on the right tiles
    int cap0 = to_index(1, 1);
    int cap1 = to_index(9, 9);
    assert(s.tile_at(cap0).has_city());
    assert(s.tile_at(cap1).has_city());

    // Starting warriors are placed
    assert(s.tile_at(cap0).has_unit());
    assert(s.tile_at(cap1).has_unit());

    // Stars start at 0
    assert(s.get_stars(0) == 0);
    assert(s.get_stars(1) == 0);

    // Fog: capitals and surrounding 3x3 are explored by their owner
    assert(s.is_explored(0, cap0));
    assert(s.is_explored(1, cap1));
    // Each player cannot see the other's capital
    assert(!s.is_explored(0, cap1));
    assert(!s.is_explored(1, cap0));

    // Water tiles have no resources
    assert(s.tile_at(to_index(5, 5)).resource() == ResourceType::None);

    printf("All init tests passed.\n\n");
    printf("Starting map (P0 capital at (1,1), P1 capital at (9,9)):\n\n");
    s.print_map();

    return 0;
}
