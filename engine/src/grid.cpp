#include "grid.h"

void neighbors(int x, int y, int out_indices[4], int& out_count, int sz) {
    out_count = 0;

    constexpr int dx[4] = { 0,  0, -1, 1 };
    constexpr int dy[4] = { -1, 1,  0, 0 };

    for (int i = 0; i < 4; i++) {
        int nx = x + dx[i];
        int ny = y + dy[i];
        if (in_bounds(nx, ny, sz))
            out_indices[out_count++] = to_index(nx, ny, sz);
    }
}
