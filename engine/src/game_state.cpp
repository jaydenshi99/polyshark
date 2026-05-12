#include <cstdio>
#include "game_state.h"

bool GameState::is_terminal() const {
    // TODO: define win conditions
    return false;
}

int GameState::winner() const {
    // TODO: return winning player index once win conditions are defined
    return -1;
}

int GameState::current_player() const {
    return cur_player;
}

void GameState::print_map() const {
    for (int y = 0; y < MAP_SIZE; y++) {
        for (int x = 0; x < MAP_SIZE; x++) {
            const Tile& t = map[to_index(x, y)];
            char c;
            switch (t.terrain()) {
                case TerrainType::Flat:     c = '.'; break;
                case TerrainType::Forest:   c = 'F'; break;
                case TerrainType::Mountain: c = 'M'; break;
                case TerrainType::Water:    c = '~'; break;
                default:                    c = '?'; break;
            }
            if (t.has_city()) c = 'C';
            if (t.has_unit()) c = 'U';
            printf("%c ", c);
        }
        printf("\n");
    }
}
