#pragma once

enum class TerrainType {
    Flat,
    Forest,
    Mountain,
    Water,
};

enum class ResourceType {
    None,
    Fruit,
    Game,
    Fish,
    Metal,
    Star,  // city ruins / ancient ruins TBD
};

enum class UnitType {
    None,  // unoccupied / sentinel
    // populated as tribes are added
};
