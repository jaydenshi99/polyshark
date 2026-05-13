#pragma once

#include "types.h"
#include "building_def.h"

struct ResourceDef {
    const char*  name;
    TechType     required_tech;   // TechType::Count = no tech required
    int          star_cost;
    int          pop_reward;
    BuildingType places_building; // building placed on tile after harvest; None if nothing
};

static const ResourceDef RESOURCE_DEFS[] = {
    // None
    { "None",   TechType::Count,        0, 0, BuildingType::None },
    // Fruit
    { "Fruit",  TechType::Organisation, 2, 1, BuildingType::None },
    // Animal
    { "Animal", TechType::Hunting,      2, 1, BuildingType::None },
    // Metal
    { "Metal",  TechType::Mining,       5, 2, BuildingType::Mine },
};

inline const ResourceDef& resource_def(ResourceType r) {
    return RESOURCE_DEFS[static_cast<int>(r)];
}
