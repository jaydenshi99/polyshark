#pragma once

#include "game_state.h"

// Returns a fully initialised starting GameState:
//   - hardcoded 11x11 map with terrain, resources, and villages
//   - player 0 capital at (1,1), player 1 capital at (9,9)
//   - one Warrior per player on their capital
//   - fog of war revealed in 3x3 around each starting position
//   - 0 starting stars (collected at start of turn 1)
GameState make_game();
