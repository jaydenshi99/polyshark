cmake_minimum_required(VERSION 3.16)
project(polyshark_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_library(engine
    src/grid.cpp
    src/game_state.cpp
)

add_executable(test_grid
    tests/test_grid.cpp
)
target_link_libraries(test_grid engine)

add_executable(test_game_state
    tests/test_game_state.cpp
)
target_link_libraries(test_game_state engine)
