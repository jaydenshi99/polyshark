#include <cstdio>
#include "game_state.h"
#include "unit_def.h"

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

int GameState::spawn_unit(UnitType type, int owner, int tile_index) {
    if (unit_count >= MAX_UNITS) return -1;
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
    return id;
}

int GameState::spawn_city(int tile_index, int owner, bool is_capital) {
    if (city_count >= MAX_CITIES) return -1;
    int id = city_count++;
    City& c = cities[id];
    c.set_tile(tile_index);
    c.set_owner(owner);
    c.set_capital(is_capital);
    map[tile_index].place_city(id);
    return id;
}

void GameState::print_map() const {
    for (int y = 0; y < MAP_SIZE; y++) {
        for (int x = 0; x < MAP_SIZE; x++) {
            const Tile& t = map[to_index(x, y)];
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
