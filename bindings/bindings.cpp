#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "game_state.h"
#include "unit_def.h"
#include "mapgen.h"

namespace py = pybind11;

static std::vector<Action> legal_actions_vec(const GameState& s) {
    Action buf[256];
    int count = 0;
    s.legal_actions(buf, count);
    return std::vector<Action>(buf, buf + count);
}

PYBIND11_MODULE(polyshark, m) {

    // --- Enums ---

    py::enum_<TerrainType>(m, "TerrainType")
        .value("Field",    TerrainType::Field)
        .value("Forest",   TerrainType::Forest)
        .value("Mountain", TerrainType::Mountain)
        .value("Water",    TerrainType::Water)
        .value("Village",  TerrainType::Village);

    // ResourceType::None collides with Python None; renamed.
    py::enum_<ResourceType>(m, "ResourceType")
        .value("No_Resource", ResourceType::None)
        .value("Fruit",       ResourceType::Fruit)
        .value("Crop",        ResourceType::Crop)
        .value("Animal",      ResourceType::Animal)
        .value("Metal",       ResourceType::Metal);

    py::enum_<UnitType>(m, "UnitType")
        .value("No_Unit",  UnitType::None)
        .value("Warrior",  UnitType::Warrior)
        .value("Archer",   UnitType::Archer)
        .value("Rider",    UnitType::Rider);

    py::enum_<BuildingType>(m, "BuildingType")
        .value("No_Building", BuildingType::None)
        .value("Mine",        BuildingType::Mine)
        .value("Farm",        BuildingType::Farm)
        .value("Road",        BuildingType::Road);

    py::enum_<TechType>(m, "TechType")
        .value("Origin",       TechType::Origin)
        .value("Hunting",      TechType::Hunting)
        .value("Organisation", TechType::Organisation)
        .value("Farming",      TechType::Farming)
        .value("Riding",       TechType::Riding)
        .value("Climbing",     TechType::Climbing)
        .value("Archery",      TechType::Archery)
        .value("Mining",       TechType::Mining);

    // --- Tile ---

    py::class_<Tile>(m, "Tile")
        .def_property_readonly("terrain",        &Tile::terrain)
        .def_property_readonly("resource",       &Tile::resource)
        .def_property_readonly("building",       &Tile::building)
        .def_property_readonly("unit_id",        &Tile::unit_id)
        .def_property_readonly("city_id",        &Tile::city_id)
        .def_property_readonly("border_city_id", &Tile::border_city_id)
        .def_property_readonly("has_unit",       &Tile::has_unit)
        .def_property_readonly("has_city",       &Tile::has_city);

    // --- Unit ---

    py::class_<Unit>(m, "Unit")
        .def_property_readonly("type",            &Unit::type)
        .def_property_readonly("owner",           &Unit::owner)
        .def_property_readonly("tile_index",      &Unit::tile_index)
        .def_property_readonly("hp",              &Unit::hp)
        .def_property_readonly("max_hp",          &Unit::max_hp)
        .def_property_readonly("move_points",     &Unit::move_points)
        .def_property_readonly("has_attacked",    &Unit::has_attacked)
        .def_property_readonly("kills",           &Unit::kills)
        .def_property_readonly("promotion_ready", &Unit::promotion_ready)
        .def_property_readonly("is_alive",        &Unit::is_alive);

    // --- City ---

    py::class_<City>(m, "City")
        .def_property_readonly("owner",           &City::owner)
        .def_property_readonly("level",           &City::level)
        .def_property_readonly("population",      &City::population)
        .def_property_readonly("tile_index",      &City::tile_index)
        .def_property_readonly("is_capital",      &City::is_capital)
        .def_property_readonly("is_sieged",       &City::is_sieged)
        .def_property_readonly("has_walls",       &City::has_walls)
        .def_property_readonly("has_workshop",    &City::has_workshop)
        .def_property_readonly("pending_upgrade", &City::has_pending_upgrade)
        .def_property_readonly("stars_per_turn",  &City::stars_per_turn);

    // --- ActionType / Action ---

    py::enum_<ActionType>(m, "ActionType")
        .value("Move",              ActionType::Move)
        .value("Attack",            ActionType::Attack)
        .value("TrainUnit",         ActionType::TrainUnit)
        .value("ConstructBuilding", ActionType::ConstructBuilding)
        .value("ResearchTech",      ActionType::ResearchTech)
        .value("CaptureCity",       ActionType::CaptureCity)
        .value("HarvestResource",   ActionType::HarvestResource)
        .value("UpgradeCity",       ActionType::UpgradeCity)
        .value("Recover",           ActionType::Recover)
        .value("EndTurn",           ActionType::EndTurn)
        .export_values();

    py::class_<Action>(m, "Action")
        .def_readonly("type",       &Action::type)
        .def_readonly("src",        &Action::from)
        .def_readonly("dst",        &Action::to)
        .def_readonly("param",      &Action::param)
        .def_readonly("affordable", &Action::affordable)
        .def("__repr__", [](const Action& a) {
            return "<Action type=" + std::to_string((int)a.type)
                 + " src=" + std::to_string(a.from)
                 + " dst=" + std::to_string(a.to)
                 + " param=" + std::to_string(a.param) + ">";
        });

    // --- GameState ---

    py::class_<GameState>(m, "GameState")
        .def("legal_actions",  &legal_actions_vec)
        .def("apply_action",   &GameState::apply_action)
        .def("is_terminal",    &GameState::is_terminal)
        .def("winner",         &GameState::winner)
        .def("current_player", &GameState::current_player)
        .def("get_turn",       &GameState::get_turn)
        .def("map_size",       &GameState::map_size)
        .def("map_tiles",      &GameState::map_tiles)
        // Returned by value so callers don't hold dangling refs across apply_action.
        .def("tile_at",        [](const GameState& s, int i) { return s.tile_at(i); })
        .def("get_unit",       [](const GameState& s, int i) { return s.get_unit(i); })
        .def("get_city",       [](const GameState& s, int i) { return s.get_city(i); })
        .def("is_visible",     &GameState::is_visible)
        .def("get_stars",      &GameState::get_stars)
        .def("get_techs",      &GameState::get_techs)
        .def("techs_mask",     &GameState::techs_mask);

    // Base movement per unit type, indexed by UnitType int value.
    // Exposed so the encoder can normalise move_points without hardcoding.
    m.def("unit_base_move", [](int unit_type) {
        return unit_def(static_cast<UnitType>(unit_type)).movement;
    });

    m.def("make_random_game", [](uint64_t seed) {
        MapGenParams p = MapGen::drylands_defaults();
        p.seed = seed;
        return MapGen(p).generate().state;
    }, py::arg("seed") = 0);
}
