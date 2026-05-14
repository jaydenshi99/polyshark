#pragma once
#include "game_state.h"
#include <cstdint>

struct MapGenParams {
    uint64_t seed             = 0;
    float    forest_percent   = 0.15f;
    float    mountain_percent = 0.08f;
    // Bit i set → villages cannot be placed at edge-distance i.
    // Drylands: exclude 0,1,2,4 → only distance 3 and 5 are valid.
    uint8_t  village_edge_exclusion = 0b00010111; // bits 0,1,2,4
    int      village_min_spacing    = 2;           // Chebyshev distance
    float    climate_jitter   = 1.5f;
    float    fruit_rate       = 0.40f;
    float    animal_rate      = 0.50f;
    float    metal_rate       = 0.40f;
};

class MapGen {
public:
    explicit MapGen(MapGenParams p = {});
    GameState generate();

    static MapGenParams drylands_defaults();

private:
    MapGenParams _p;
    uint64_t     _rng;

    uint64_t rand64();
    int      randi(int lo, int hi);
    float    randf();

    void place_terrain (GameState& s, int cap0, int cap1);
    void fill_climate  (int climate[MAP_TILES], int cap0, int cap1);
    void place_villages(GameState& s, int cap0, int cap1);
    void place_resources(GameState& s, const int climate[MAP_TILES], int cap0, int cap1);
    void init_players  (GameState& s, int cap0, int cap1);
};
