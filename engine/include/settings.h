#pragma once
#include <cstdint>
#include "config.h"
#include "mapgen.h"

// Runtime-mutable engine settings — the *only* knobs an AI / experiment is
// expected to change per game. Compile-time gameplay tuning (combat heals,
// explorer scoring, city upgrade rewards, etc.) lives in `cfg::*` and is
// deliberately NOT exposed here.
//
// Defaults pull from cfg::map::* so a default-constructed EngineSettings
// matches a default-built engine. Pass to `make_game(...)` in the bindings.
struct EngineSettings {
    int       map_size = cfg::map::DEFAULT_SIZE;
    uint64_t  seed     = cfg::map::DEFAULT_SEED;
    BiomeType biome    = BiomeType::Drylands;

    MapGenParams to_mapgen() const {
        MapGenParams p = MapGenParams::for_biome(biome, map_size);
        p.seed = seed;
        return p;
    }
};
