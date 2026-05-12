#include "city.h"

void City::add_population(int n) {
    _population += n;
    // TODO: trigger level-up logic when population threshold is reached
}

void City::capture(int new_owner) {
    _owner = new_owner;
    // TODO: additional capture effects (population reset, etc.)
}
