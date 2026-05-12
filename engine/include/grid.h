#pragma once

constexpr int MAP_SIZE = 11;
constexpr int MAP_TILES = MAP_SIZE * MAP_SIZE;

inline int to_index(int x, int y) {
    return y * MAP_SIZE + x;
}

inline void to_coords(int index, int& x, int& y) {
    x = index % MAP_SIZE;
    y = index / MAP_SIZE;
}

inline bool in_bounds(int x, int y) {
    return x >= 0 && x < MAP_SIZE && y >= 0 && y < MAP_SIZE;
}

inline int distance(int x1, int y1, int x2, int y2) {
    return (x1 > x2 ? x1 - x2 : x2 - x1) +
           (y1 > y2 ? y1 - y2 : y2 - y1);
}

void neighbors(int x, int y, int out_indices[4], int& out_count);
