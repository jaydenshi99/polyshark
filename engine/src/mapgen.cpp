#include "mapgen.h"
#include "grid.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <vector>

// Quadrant centers on 11x11: TL(2,2) TR(8,2) BL(2,8) BR(8,8)
static const int QUAD_CX[4] = {2, 8, 2, 8};
static const int QUAD_CY[4] = {2, 2, 8, 8};

MapGen::MapGen(MapGenParams p)
    : _p(p), _rng(p.seed == 0 ? 0xdeadbeefcafe1234ULL : p.seed) {}

MapGenParams MapGen::drylands_defaults() { return {}; }

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

GameState MapGen::generate() {
    GameState s;

    // Pick 2 distinct quadrants for capitals
    int quads[4] = {0, 1, 2, 3};
    int qa = randi(0, 3); std::swap(quads[0], quads[qa]);
    int qb = randi(1, 3); std::swap(quads[1], quads[qb]);
    int cap0 = to_index(QUAD_CX[quads[0]], QUAD_CY[quads[0]]);
    int cap1 = to_index(QUAD_CX[quads[1]], QUAD_CY[quads[1]]);

    place_terrain(s, cap0, cap1);

    int climate[MAP_TILES];
    fill_climate(climate, cap0, cap1);

    place_villages(s, cap0, cap1);
    place_resources(s, climate, cap0, cap1);

    // TODO: ruins

    init_players(s, cap0, cap1);
    return s;
}

void MapGen::place_terrain(GameState& s, int cap0, int cap1) {
    int forest_count   = (int)(_p.forest_percent   * MAP_TILES);
    int mountain_count = (int)(_p.mountain_percent * MAP_TILES);

    std::vector<int> tiles(MAP_TILES);
    std::iota(tiles.begin(), tiles.end(), 0);
    for (int i = MAP_TILES - 1; i > 0; i--)
        std::swap(tiles[i], tiles[randi(0, i)]);

    int forests = 0, mountains = 0;
    for (int idx : tiles) {
        if (idx == cap0 || idx == cap1) continue;
        if (forests < forest_count) {
            s.tile_at(idx).set_terrain(TerrainType::Forest);
            ++forests;
        } else if (mountains < mountain_count) {
            s.tile_at(idx).set_terrain(TerrainType::Mountain);
            ++mountains;
        } else {
            break;
        }
    }
}

void MapGen::fill_climate(int climate[MAP_TILES], int cap0, int cap1) {
    std::fill(climate, climate + MAP_TILES, -1);

    struct Entry {
        float priority;
        int   tile;
        int   owner;
        bool operator>(const Entry& o) const { return priority > o.priority; }
    };

    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq;
    pq.push({0.0f, cap0, 0});
    pq.push({0.0f, cap1, 1});

    while (!pq.empty()) {
        auto [pri, tile, owner] = pq.top();
        pq.pop();
        if (climate[tile] != -1) continue;
        climate[tile] = owner;

        int x, y;
        to_coords(tile, x, y);
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                int nx = x + dx, ny = y + dy;
                if (!in_bounds(nx, ny)) continue;
                int ni = to_index(nx, ny);
                if (climate[ni] != -1) continue;
                pq.push({pri + 1.0f + randf() * _p.climate_jitter, ni, owner});
            }
        }
    }
}

void MapGen::place_villages(GameState& s, int cap0, int cap1) {
    uint8_t excl    = _p.village_edge_exclusion;
    int     spacing = _p.village_min_spacing;

    auto edge_dist = [](int v) { return std::min(v, MAP_SIZE - 1 - v); };
    auto edge_ok   = [&](int x, int y) {
        int ex = edge_dist(x), ey = edge_dist(y);
        return !((excl >> ex) & 1) && !((excl >> ey) & 1);
    };

    std::vector<int> placed = {cap0, cap1};

    std::vector<int> candidates;
    for (int y = 0; y < MAP_SIZE; y++)
        for (int x = 0; x < MAP_SIZE; x++) {
            if (!edge_ok(x, y)) continue;
            int idx = to_index(x, y);
            if (s.tile_at(idx).terrain() == TerrainType::Field)
                candidates.push_back(idx);
        }

    for (int i = (int)candidates.size() - 1; i > 0; i--)
        std::swap(candidates[i], candidates[randi(0, i)]);

    for (int idx : candidates) {
        int cx, cy;
        to_coords(idx, cx, cy);

        bool ok = true;
        for (int p : placed) {
            int px, py;
            to_coords(p, px, py);
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

void MapGen::place_resources(GameState& s, const int climate[MAP_TILES], int cap0, int cap1) {
    for (int i = 0; i < MAP_TILES; i++) {
        if (i == cap0 || i == cap1) continue;
        float roll = randf();
        switch (s.tile_at(i).terrain()) {
            case TerrainType::Field:
                if (roll < _p.fruit_rate)  s.tile_at(i).set_resource(ResourceType::Fruit);
                break;
            case TerrainType::Forest:
                if (roll < _p.animal_rate) s.tile_at(i).set_resource(ResourceType::Animal);
                break;
            case TerrainType::Mountain:
                if (roll < _p.metal_rate)  s.tile_at(i).set_resource(ResourceType::Metal);
                break;
            default: break;
        }
    }
}

void MapGen::init_players(GameState& s, int cap0, int cap1) {
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
        to_coords(caps[p], cx, cy);
        for (int dy = -2; dy <= 2; dy++)
            for (int dx = -2; dx <= 2; dx++)
                if (in_bounds(cx + dx, cy + dy))
                    s.set_explored(p, to_index(cx + dx, cy + dy));
    }
}
