#include <cassert>
#include <cstdio>
#include "grid.h"

static void test_to_index() {
    assert(to_index(0, 0) == 0);
    assert(to_index(10, 0) == 10);
    assert(to_index(0, 1) == 11);
    assert(to_index(10, 10) == 120);
}

static void test_to_coords() {
    int x, y;
    to_coords(0, x, y);   assert(x == 0 && y == 0);
    to_coords(10, x, y);  assert(x == 10 && y == 0);
    to_coords(11, x, y);  assert(x == 0 && y == 1);
    to_coords(120, x, y); assert(x == 10 && y == 10);
}

static void test_in_bounds() {
    assert(in_bounds(0, 0));
    assert(in_bounds(10, 10));
    assert(!in_bounds(-1, 0));
    assert(!in_bounds(0, -1));
    assert(!in_bounds(11, 0));
    assert(!in_bounds(0, 11));
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
