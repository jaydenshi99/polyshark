# Match Lab native tools (developer build). The lab server builds the same
# sources itself, per frozen engine hash; this target set is for IDEs and tests.
#
# The engine is compiled again here with -ffp-contract=off so floating-point
# results never depend on FMA availability (replays must match across machines).

set(LAB_ENGINE_SOURCES
    ${CMAKE_SOURCE_DIR}/engine/src/grid.cpp
    ${CMAKE_SOURCE_DIR}/engine/src/unit.cpp
    ${CMAKE_SOURCE_DIR}/engine/src/city.cpp
    ${CMAKE_SOURCE_DIR}/engine/src/game_state/game_state.cpp
    ${CMAKE_SOURCE_DIR}/engine/src/game_state/legal_actions.cpp
    ${CMAKE_SOURCE_DIR}/engine/src/game_state/apply_action.cpp
    ${CMAKE_SOURCE_DIR}/engine/src/game_state/damage_calculator.cpp
    ${CMAKE_SOURCE_DIR}/engine/src/game_state/forced_spawn.cpp
    ${CMAKE_SOURCE_DIR}/engine/src/game_state/state_serialiser.cpp
    ${CMAKE_SOURCE_DIR}/engine/src/mapgen.cpp
)

add_library(polyshark_lab STATIC ${LAB_ENGINE_SOURCES} native/lab_core.cpp)
target_include_directories(polyshark_lab PUBLIC
    ${CMAKE_SOURCE_DIR}/engine/include
    ${CMAKE_SOURCE_DIR}/bots/sdk/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/native)
target_link_libraries(polyshark_lab PUBLIC nlohmann_json::nlohmann_json)
target_compile_options(polyshark_lab PUBLIC -ffp-contract=off)
set_target_properties(polyshark_lab PROPERTIES POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)

add_executable(polyshark-referee native/referee.cpp native/process.cpp)
target_link_libraries(polyshark-referee PRIVATE polyshark_lab)

add_executable(polyshark-tool native/tool.cpp)
target_link_libraries(polyshark-tool PRIVATE polyshark_lab)

add_executable(polyshark-selfplay native/selfplay.cpp)
target_link_libraries(polyshark-selfplay PRIVATE polyshark_lab Threads::Threads ${CMAKE_DL_LIBS})

add_executable(test_lab native/test_lab.cpp)
target_link_libraries(test_lab PRIVATE polyshark_lab)
