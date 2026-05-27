#include <cassert>
#include <cstdio>
#include "grid.h"

static void test_to_index() {
    constexpr int N = MAP_SIZE;
    assert(to_index(0, 0)         == 0);
    assert(to_index(N - 1, 0)     == N - 1);
    assert(to_index(0, 1)         == N);
    assert(to_index(N - 1, N - 1) == N * N - 1);
}

static void test_to_coords() {
    constexpr int N = MAP_SIZE;
    int x, y;
    to_coords(0,           x, y); assert(x == 0     && y == 0);
    to_coords(N - 1,       x, y); assert(x == N - 1 && y == 0);
    to_coords(N,           x, y); assert(x == 0     && y == 1);
    to_coords(N * N - 1,   x, y); assert(x == N - 1 && y == N - 1);
}

static void test_in_bounds() {
    constexpr int N = MAP_SIZE;
    assert(in_bounds(0, 0));
    assert(in_bounds(N - 1, N - 1));
    assert(!in_bounds(-1, 0));
    assert(!in_bounds(0, -1));
    assert(!in_bounds(N, 0));
    assert(!in_bounds(0, N));
}

static void test_distance() {
    assert(manhattan_distance(0, 0, 0, 0) == 0);
    assert(manhattan_distance(0, 0, 1, 0) == 1);
    assert(manhattan_distance(0, 0, 0, 1) == 1);
    assert(manhattan_distance(0, 0, 3, 4) == 7);
    assert(manhattan_distance(5, 5, 2, 1) == 7);

    assert(chebyshev_distance(0, 0, 0, 0) == 0);
    assert(chebyshev_distance(0, 0, 1, 0) == 1);
    assert(chebyshev_distance(0, 0, 3, 4) == 4);
    assert(chebyshev_distance(5, 5, 2, 1) == 4);
}

static void test_neighbors() {
    int indices[4], count;

    // Corner tile — should have 2 neighbors
    neighbors(0, 0, indices, count);
    assert(count == 2);

    // Edge tile — should have 3 neighbors
    neighbors(5, 0, indices, count);
    assert(count == 3);

    // Center tile — should have 4 neighbors
    neighbors(5, 5, indices, count);
    assert(count == 4);
}

int main() {
    test_to_index();
    test_to_coords();
    test_in_bounds();
    test_distance();
    test_neighbors();
    printf("All grid tests passed.\n");
    return 0;
}
