#pragma once

constexpr int MAP_SIZE      = 11;
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

inline int distance(int x1, int y1, int x2, int y2) {
    return (x1 > x2 ? x1 - x2 : x2 - x1) +
           (y1 > y2 ? y1 - y2 : y2 - y1);
}

void neighbors(int x, int y, int out_indices[4], int& out_count, int sz = MAP_SIZE);
