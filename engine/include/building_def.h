#pragma once

#include "types.h"

enum class BuildingType {
    None = 0,
    Mine,       // built on mountains after harvesting Metal
    Farm,       // built on fields after harvesting Crop
    Road,       // reduces tile entry cost to 0 for connected movement
    Count,
};

struct BuildingDef {
    const char* name;
    TerrainType required_terrain;  // TerrainType the building can sit on
};

static const BuildingDef BUILDING_DEFS[] = {
    { "None", TerrainType::Field    },
    { "Mine", TerrainType::Mountain },
    { "Farm", TerrainType::Field    },
    { "Road", TerrainType::Field    },  // placeholder; any terrain in practice
};

inline const BuildingDef& building_def(BuildingType b) {
    return BUILDING_DEFS[static_cast<int>(b)];
}
