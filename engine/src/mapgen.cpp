#include "mapgen.h"
#include "grid.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

MapGenParams MapGenParams::for_biome(BiomeType b, int sz) {
    MapGenParams p;
    p.map_size = sz;
    p.biome    = b;
    switch (b) {
        case BiomeType::Drylands:
        default:
            p.village_edge_exclusion = 0b00001011;
            p.village_min_spacing    = 3;

            // hardcoded for imperius tribe.
            for (int i = 0; i < 2; i++) {
                p.tribe_rates[i].outer.mountain = 0.14f;
                p.tribe_rates[i].outer.forest   = 0.38f;
                p.tribe_rates[i].outer.fruit     = 0.24f;
                p.tribe_rates[i].outer.metal     = 0.21f;
                p.tribe_rates[i].outer.animal    = 0.08f;

                p.tribe_rates[i].inner.mountain  = 0.14f;
                p.tribe_rates[i].inner.forest    = 0.38f;
                p.tribe_rates[i].inner.fruit     = 0.75f;
                p.tribe_rates[i].inner.metal     = 0.80f;
                p.tribe_rates[i].inner.animal    = 0.25f;
            }
            break;
    }
    return p;
}

// Quadrant centers — scaled to map size. Quadrants: TL TR BL BR.
static void quad_centers(int sz, int cx[4], int cy[4]) {
    int q = sz / 4;          // offset from corner
    cx[0] = q;     cy[0] = q;
    cx[1] = sz-1-q; cy[1] = q;
    cx[2] = q;     cy[2] = sz-1-q;
    cx[3] = sz-1-q; cy[3] = sz-1-q;
}

MapGen::MapGen(MapGenParams p)
    : _p(p), _rng(p.seed == 0 ? 0xdeadbeefcafe1234ULL : p.seed) {}

MapGenParams MapGen::drylands_defaults() {
    return MapGenParams::for_biome(BiomeType::Drylands);
}

uint64_t MapGen::rand64() {
    _rng ^= _rng << 13;
    _rng ^= _rng >> 7;
    _rng ^= _rng << 17;
    return _rng;
}

int MapGen::randi(int lo, int hi) {
    return lo + (int)(rand64() % (uint64_t)(hi - lo + 1));
}

float MapGen::randf() {
    return (float)(rand64() >> 11) * (1.0f / (float)(1ULL << 53));
}

MapGenResult MapGen::generate() {
    const int sz   = _p.map_size;
    const int mtsz = sz * sz;

    GameState s;
    s.set_map_size(sz);

    // Pick 2 distinct quadrants for capitals
    int qcx[4], qcy[4];
    quad_centers(sz, qcx, qcy);

    int quads[4] = {0, 1, 2, 3};
    int qa = randi(0, 3); std::swap(quads[0], quads[qa]);
    int qb = randi(1, 3); std::swap(quads[1], quads[qb]);
    int cap0 = to_index(qcx[quads[0]], qcy[quads[0]], sz);
    int cap1 = to_index(qcx[quads[1]], qcy[quads[1]], sz);

    MapGenResult result;
    fill_climate(result.climate, cap0, cap1);

    place_terrain(s, result.climate, cap0, cap1);
    place_villages(s, cap0, cap1);
    reroll_inner(s, result.climate, cap0, cap1);

    // TODO: ruins

    init_players(s, cap0, cap1);
    result.state = s;
    return result;
}

void MapGen::apply_tile_rates(GameState& s, int idx, const TileRates& r) {
    float tr = randf();
    TerrainType ter;
    if      (tr < r.mountain)              ter = TerrainType::Mountain;
    else if (tr < r.mountain + r.forest)   ter = TerrainType::Forest;
    else                                   ter = TerrainType::Field;
    s.tile_at(idx).set_terrain(ter);
    s.tile_at(idx).set_resource(ResourceType::None);
    float rr = randf();
    switch (ter) {
        case TerrainType::Field:    if (rr < r.fruit)  s.tile_at(idx).set_resource(ResourceType::Fruit);  break;
        case TerrainType::Mountain: if (rr < r.metal)  s.tile_at(idx).set_resource(ResourceType::Metal);  break;
        case TerrainType::Forest:   if (rr < r.animal) s.tile_at(idx).set_resource(ResourceType::Animal); break;
        default: break;
    }
}

void MapGen::place_terrain(GameState& s, const int climate[], int cap0, int cap1) {
    const int mtsz = _p.map_size * _p.map_size;
    for (int i = 0; i < mtsz; i++) {
        if (i == cap0 || i == cap1) continue;
        int tribe = (climate[i] >= 0 && climate[i] < 2) ? climate[i] : 0;
        apply_tile_rates(s, i, _p.tribe_rates[tribe].outer);
    }
}

void MapGen::fill_climate(int climate[], int cap0, int cap1) {
    const int sz   = _p.map_size;
    const int mtsz = sz * sz;

    std::fill(climate, climate + mtsz, -1);

    std::vector<int> frontier[2];
    frontier[0].push_back(cap0);
    frontier[1].push_back(cap1);
    climate[cap0] = 0;
    climate[cap1] = 1;

    auto expand = [&](int owner) {
        if (frontier[owner].empty()) return;
        int fi   = randi(0, (int)frontier[owner].size() - 1);
        int tile = frontier[owner][fi];
        frontier[owner][fi] = frontier[owner].back();
        frontier[owner].pop_back();

        int x, y;
        to_coords(tile, x, y, sz);
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                int nx = x + dx, ny = y + dy;
                if (!in_bounds(nx, ny, sz)) continue;
                int ni = to_index(nx, ny, sz);
                if (climate[ni] != -1) continue;
                climate[ni] = owner;
                frontier[owner].push_back(ni);
            }
        }
    };

    while (!frontier[0].empty() || !frontier[1].empty()) {
        expand(0);
        expand(1);
    }
}

void MapGen::place_villages(GameState& s, int cap0, int cap1) {
    const int sz      = _p.map_size;
    uint8_t   excl    = _p.village_edge_exclusion;
    int       spacing = _p.village_min_spacing;

    auto edge_dist = [&](int v) { return std::min(v, sz - 1 - v); };
    auto edge_ok   = [&](int x, int y) {
        int ex = edge_dist(x), ey = edge_dist(y);
        return !((excl >> ex) & 1) && !((excl >> ey) & 1);
    };

    std::vector<int> placed = {cap0, cap1};

    std::vector<int> candidates;
    for (int y = 0; y < sz; y++)
        for (int x = 0; x < sz; x++) {
            if (!edge_ok(x, y)) continue;
            int idx = to_index(x, y, sz);
            if (s.tile_at(idx).terrain() == TerrainType::Field)
                candidates.push_back(idx);
        }

    for (int i = (int)candidates.size() - 1; i > 0; i--)
        std::swap(candidates[i], candidates[randi(0, i)]);

    for (int idx : candidates) {
        int cx, cy;
        to_coords(idx, cx, cy, sz);

        bool ok = true;
        for (int p : placed) {
            int px, py;
            to_coords(p, px, py, sz);
            if (std::max(std::abs(cx - px), std::abs(cy - py)) < spacing) {
                ok = false;
                break;
            }
        }
        if (ok) {
            s.tile_at(idx).set_terrain(TerrainType::Village);
            placed.push_back(idx);
        }
    }
}

void MapGen::reroll_inner(GameState& s, const int climate[], int cap0, int cap1) {
    const int sz   = _p.map_size;
    const int mtsz = sz * sz;

    for (int i = 0; i < mtsz; i++) {
        if (i == cap0 || i == cap1) continue;
        if (s.tile_at(i).terrain() != TerrainType::Village) continue;
        int vx, vy;
        to_coords(i, vx, vy, sz);
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                int nx = vx + dx, ny = vy + dy;
                if (!in_bounds(nx, ny, sz)) continue;
                int ni = to_index(nx, ny, sz);
                if (ni == cap0 || ni == cap1) continue;
                if (s.tile_at(ni).terrain() == TerrainType::Village) continue;
                int tribe = (climate[ni] >= 0 && climate[ni] < 2) ? climate[ni] : 0;
                apply_tile_rates(s, ni, _p.tribe_rates[tribe].inner);
            }
    }
}

void MapGen::init_players(GameState& s, int cap0, int cap1) {
    const int sz = _p.map_size;
    s.spawn_city(cap0, 0, true);
    s.spawn_city(cap1, 1, true);
    s.spawn_unit(UnitType::Warrior, 0, cap0);
    s.spawn_unit(UnitType::Warrior, 1, cap1);
    s.set_stars(0, 5);
    s.set_stars(1, 5);
    s.research_tech(0, TechType::Origin);
    s.research_tech(1, TechType::Origin);

    int caps[2] = {cap0, cap1};
    for (int p = 0; p < 2; p++) {
        int cx, cy;
        to_coords(caps[p], cx, cy, sz);
        for (int dy = -2; dy <= 2; dy++)
            for (int dx = -2; dx <= 2; dx++)
                if (in_bounds(cx + dx, cy + dy, sz))
                    s.set_explored(p, to_index(cx + dx, cy + dy, sz));
    }
}
