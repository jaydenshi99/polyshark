#pragma once

#include "types.h"

class Tile {
public:
    TerrainType  terrain()   const { return _terrain; }
    ResourceType resource()  const { return _resource; }
    int          unit_id()   const { return _unit_id; }
    int          city_id()   const { return _city_id; }
    void set_terrain(TerrainType t)   { _terrain = t; }
    void set_resource(ResourceType r) { _resource = r; }

    // Caller must also update Unit::tile_index when placing/removing a unit
    void place_unit(int id) { _unit_id = id; }
    void remove_unit()      { _unit_id = -1; }

    void place_city(int id) { _city_id = id; }
    void remove_city()      { _city_id = -1; }

    bool has_unit() const { return _unit_id != -1; }
    bool has_city() const { return _city_id != -1; }

private:
    TerrainType  _terrain  = TerrainType::Field;
    ResourceType _resource = ResourceType::None;
    int          _unit_id  = -1;
    int          _city_id  = -1;
};
