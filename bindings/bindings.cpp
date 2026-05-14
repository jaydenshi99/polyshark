#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "game_state.h"
#include "init.h"

namespace py = pybind11;

static std::vector<Action> legal_actions_vec(const GameState& s) {
    Action buf[256];
    int count = 0;
    s.legal_actions(buf, count);
    return std::vector<Action>(buf, buf + count);
}

PYBIND11_MODULE(polyshark, m) {
    py::enum_<ActionType>(m, "ActionType")
        .value("Move",              ActionType::Move)
        .value("Attack",            ActionType::Attack)
        .value("TrainUnit",         ActionType::TrainUnit)
        .value("ConstructBuilding", ActionType::ConstructBuilding)
        .value("ResearchTech",      ActionType::ResearchTech)
        .value("CaptureCity",       ActionType::CaptureCity)
        .value("HarvestResource",   ActionType::HarvestResource)
        .value("UpgradeCity",       ActionType::UpgradeCity)
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

    py::class_<GameState>(m, "GameState")
        .def("legal_actions",   &legal_actions_vec)
        .def("apply_action",    &GameState::apply_action)
        .def("is_terminal",     &GameState::is_terminal)
        .def("winner",          &GameState::winner)
        .def("current_player",  &GameState::current_player);

    m.def("make_game", &make_game);
}
