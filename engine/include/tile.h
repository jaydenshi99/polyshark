#pragma once

#include "types.h"
#include "building_def.h"

class Tile {
public:
    TerrainType  terrain()   const { return _terrain; }
    ResourceType resource()  const { return _resource; }
    BuildingType building()  const { return _building; }
    int          unit_id()   const { return _unit_id; }
    int          city_id()   const { return _city_id; }
    void set_terrain(TerrainType t)    { _terrain = t; }
    void set_resource(ResourceType r)  { _resource = r; }
    void set_building(BuildingType b)  { _building = b; }
    bool has_building() const { return _building != BuildingType::None; }

    // Caller must also update Unit::tile_index when placing/removing a unit
    void place_unit(int id) { _unit_id = id; }
    void remove_unit()      { _unit_id = -1; }

    void place_city(int id) { _city_id = id; }
    void remove_city()      { _city_id = -1; }

    bool has_unit() const { return _unit_id != -1; }
    bool has_city() const { return _city_id != -1; }

    // For village tiles (no city object): set at turn-switch when the active player
    // already has a unit here; cleared when the unit leaves or captures.
    bool capture_ready()          const { return _capture_ready; }
    void set_capture_ready(bool v)      { _capture_ready = v; }

private:
    TerrainType  _terrain   = TerrainType::Field;
    ResourceType _resource  = ResourceType::None;
    BuildingType _building  = BuildingType::None;
    int          _unit_id      = -1;
    int          _city_id      = -1;
    bool         _capture_ready = false;
};
