cmake_minimum_required(VERSION 3.16)
project(polyshark_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_library(engine
    src/grid.cpp
    src/unit.cpp
    src/city.cpp
    src/game_state/game_state.cpp
    src/game_state/legal_actions.cpp
    src/game_state/apply_action.cpp
    src/game_state/damage_calculator.cpp
    src/game_state/forced_spawn.cpp
    src/game_state/state_serialiser.cpp
    src/mapgen.cpp
)

# game_state.h includes <nlohmann/json.hpp>, so it must propagate to consumers.
target_link_libraries(engine PUBLIC nlohmann_json::nlohmann_json)

add_executable(test_grid
    tests/test_grid.cpp
)
target_link_libraries(test_grid engine)

add_executable(test_game_state
    tests/test_game_state.cpp
)
target_link_libraries(test_game_state engine)

