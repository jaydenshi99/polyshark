#pragma once

#include "config.h"

constexpr int MAP_SIZE      = cfg::map::DEFAULT_SIZE;
constexpr int MAP_TILES     = MAP_SIZE * MAP_SIZE;
constexpr int MAX_MAP_SIZE  = 32;
constexpr int MAX_MAP_TILES = MAX_MAP_SIZE * MAX_MAP_SIZE;

inline int to_index(int x, int y, int sz = MAP_SIZE) {
    return y * sz + x;
}

inline void to_coords(int index, int& x, int& y, int sz = MAP_SIZE) {
    x = index % sz;
    y = index / sz;
}

inline bool in_bounds(int x, int y, int sz = MAP_SIZE) {
    return x >= 0 && x < sz && y >= 0 && y < sz;
}

static inline int abs_diff(int a, int b) { return a > b ? a - b : b - a; }

// 4-directional grid distance.
static inline int manhattan_distance(int x1, int y1, int x2, int y2) {
    return abs_diff(x1, x2) + abs_diff(y1, y2);
}
static inline int manhattan_distance(int t1, int t2, int sz) {
    return manhattan_distance(t1 % sz, t1 / sz, t2 % sz, t2 / sz);
}

// 8-directional (king's move) grid distance.
static inline int chebyshev_distance(int x1, int y1, int x2, int y2) {
    int dx = abs_diff(x1, x2), dy = abs_diff(y1, y2);
    return dx > dy ? dx : dy;
}
static inline int chebyshev_distance(int t1, int t2, int sz) {
    return chebyshev_distance(t1 % sz, t1 / sz, t2 % sz, t2 / sz);
}

void neighbors(int x, int y, int out_indices[4], int& out_count, int sz = MAP_SIZE);

// Lighthouses sit on the four map corners. They start fogged and grant a
// reveal bonus (+1 population to the capital that explores them).
inline bool is_lighthouse(int tile, int sz) {
    int x = tile % sz, y = tile / sz;
    return (x == 0 || x == sz - 1) && (y == 0 || y == sz - 1);
}
