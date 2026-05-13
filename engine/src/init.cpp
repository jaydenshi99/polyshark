#include "init.h"
#include "grid.h"

using T = TerrainType;
using R = ResourceType;

// 11x11 terrain layout, row-major (y=0 top, y=10 bottom)
// clang-format off
static const TerrainType TERRAIN[MAP_TILES] = {
//       x=0          x=1          x=2          x=3          x=4          x=5          x=6          x=7          x=8          x=9          x=10
/*y=0*/  T::Field,    T::Forest,   T::Field,    T::Field,    T::Mountain, T::Field,    T::Village,  T::Field,    T::Forest,   T::Field,    T::Field,
/*y=1*/  T::Forest,   T::Field,    T::Field,    T::Field,    T::Field,    T::Field,    T::Field,    T::Forest,   T::Field,    T::Field,    T::Forest,
/*y=2*/  T::Field,    T::Field,    T::Mountain, T::Forest,   T::Field,    T::Field,    T::Field,    T::Field,    T::Forest,   T::Field,    T::Field,
/*y=3*/  T::Field,    T::Village,  T::Field,    T::Field,    T::Forest,   T::Field,    T::Mountain, T::Field,    T::Field,    T::Forest,   T::Field,
/*y=4*/  T::Mountain, T::Field,    T::Forest,   T::Field,    T::Field,    T::Field,    T::Field,    T::Forest,   T::Field,    T::Field,    T::Mountain,
/*y=5*/  T::Field,    T::Field,    T::Field,    T::Field,    T::Field,    T::Water,    T::Water,    T::Field,    T::Field,    T::Field,    T::Field,
/*y=6*/  T::Mountain, T::Field,    T::Forest,   T::Field,    T::Field,    T::Water,    T::Water,    T::Field,    T::Field,    T::Forest,   T::Mountain,
/*y=7*/  T::Field,    T::Forest,   T::Field,    T::Field,    T::Mountain, T::Field,    T::Forest,   T::Field,    T::Field,    T::Village,  T::Field,
/*y=8*/  T::Field,    T::Field,    T::Forest,   T::Field,    T::Field,    T::Field,    T::Field,    T::Forest,   T::Mountain, T::Field,    T::Field,
/*y=9*/  T::Forest,   T::Field,    T::Field,    T::Forest,   T::Field,    T::Field,    T::Field,    T::Field,    T::Field,    T::Field,    T::Forest,
/*y=10*/ T::Field,    T::Field,    T::Forest,   T::Field,    T::Village,  T::Field,    T::Mountain, T::Field,    T::Field,    T::Forest,   T::Field,
};
// clang-format on

// Sparse resource placements — terrain type already verified by design
static const struct { int x, y; ResourceType r; } RESOURCES[] = {
    // Forests -> Game
    {1,0,R::Game},  {8,0,R::Game},
    {0,1,R::Game},  {7,1,R::Game},
    {3,2,R::Game},  {8,2,R::Game},
    {4,3,R::Game},  {9,3,R::Game},
    {2,4,R::Game},  {7,4,R::Game},
    {2,6,R::Game},  {9,6,R::Game},
    {1,7,R::Game},  {6,7,R::Game},
    {2,8,R::Game},  {7,8,R::Game},
    {0,9,R::Game},  {3,9,R::Game},
    {2,10,R::Game}, {9,10,R::Game},
    // Mountains -> Metal
    {4,0,R::Metal},
    {2,2,R::Metal},
    {6,3,R::Metal},
    {0,4,R::Metal}, {10,4,R::Metal},
    {0,6,R::Metal}, {10,6,R::Metal},
    {4,7,R::Metal},
    {8,8,R::Metal},
    {6,10,R::Metal},
    // Fields -> Fruit
    {2,1,R::Fruit}, {9,1,R::Fruit},
    {4,2,R::Fruit},
    {0,3,R::Fruit}, {3,3,R::Fruit},
    {4,4,R::Fruit},
    {1,5,R::Fruit}, {4,5,R::Fruit}, {7,5,R::Fruit},
    {8,6,R::Fruit},
    {3,7,R::Fruit},
    {1,8,R::Fruit}, {4,8,R::Fruit},
    {1,9,R::Fruit}, {7,9,R::Fruit},
    {1,10,R::Fruit},
};

GameState make_game() {
    GameState s;

    // Terrain
    for (int i = 0; i < MAP_TILES; i++)
        s.tile_at(i).set_terrain(TERRAIN[i]);

    // Resources
    for (auto& p : RESOURCES)
        s.tile_at(to_index(p.x, p.y)).set_resource(p.r);

    // Capitals: P0 at (1,1), P1 at (9,9)
    int cap0 = to_index(1, 1);
    int cap1 = to_index(9, 9);
    s.spawn_city(cap0, 0, true);
    s.spawn_city(cap1, 1, true);

    // Starting warriors
    s.spawn_unit(UnitType::Warrior, 0, cap0);
    s.spawn_unit(UnitType::Warrior, 1, cap1);

    // Initial fog: reveal 3x3 around each capital
    for (int player = 0; player < 2; player++) {
        int cx = (player == 0) ? 1 : 9;
        int cy = (player == 0) ? 1 : 9;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (in_bounds(cx + dx, cy + dy))
                    s.set_explored(player, to_index(cx + dx, cy + dy));
            }
        }
    }

    return s;
}
