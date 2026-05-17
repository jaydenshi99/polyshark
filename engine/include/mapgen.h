#pragma once
#include "game_state.h"
#include "config.h"
#include <cstdint>

enum class BiomeType {
    Drylands,
};

struct TileRates {
    float mountain = 0.08f;  // P(Mountain)
    float forest   = 0.15f;  // P(Forest)
    // field = remainder
    float fruit    = 0.20f;  // P(Fruit  | Field)  — mutually exclusive with crop
    float crop     = 0.20f;  // P(Crop   | Field, not Fruit)
    float metal    = 0.40f;  // P(Metal  | Mountain)
    float animal   = 0.50f;  // P(Animal | Forest)
};

struct TribeRates {
    TileRates outer;  // tiles not adjacent to any village
    TileRates inner;  // tiles Chebyshev-1 adjacent to any village
};

struct MapGenParams {
    BiomeType biome            = BiomeType::Drylands;
    int      map_size          = cfg::map::DEFAULT_SIZE;  // must be <= MAX_MAP_SIZE
    uint64_t seed              = cfg::map::DEFAULT_SEED;
    // Bit i set → villages cannot be placed at edge-distance i.
    uint8_t  village_edge_exclusion = cfg::map::VILLAGE_EDGE_EXCLUDE;
    int      village_min_spacing    = cfg::map::VILLAGE_MIN_SPACING;
    TribeRates tribe_rates[2];                     // [0] = P0 zone, [1] = P1 zone

    static MapGenParams for_biome(BiomeType b, int sz = MAP_SIZE);
};

struct MapGenResult {
    GameState state;
    int       climate[MAX_MAP_TILES]; // 0 = P0 tribe zone, 1 = P1 tribe zone
};

class MapGen {
public:
    explicit MapGen(MapGenParams p = {});
    MapGenResult generate();

    // Convenience: same as MapGenParams::for_biome(BiomeType::Drylands)
    static MapGenParams drylands_defaults();

private:
    MapGenParams _p;
    uint64_t     _rng;

    uint64_t rand64();
    int      randi(int lo, int hi);
    float    randf();

    void apply_tile_rates(GameState& s, int idx, const TileRates& r);
    void place_terrain   (GameState& s, const int climate[], int cap0, int cap1);
    void fill_climate    (int climate[], int cap0, int cap1);
    void place_villages  (GameState& s, int cap0, int cap1);
    void reroll_inner    (GameState& s, const int climate[], int cap0, int cap1);
    void clear_far_resources(GameState& s, int cap0, int cap1);
    void init_players    (GameState& s, int cap0, int cap1);
};
