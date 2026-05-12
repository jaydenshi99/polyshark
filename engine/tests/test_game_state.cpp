#include <cassert>
#include <cstdio>
#include "game_state.h"

static void test_new_game() {
    GameState s;
    assert(s.current_player() == 0);
    assert(s.winner() == -1);
    assert(!s.is_terminal());
}

int main() {
    test_new_game();
    printf("All game_state tests passed.\n");

    printf("\nBlank map:\n");
    GameState s;
    s.print_map();

    return 0;
}
