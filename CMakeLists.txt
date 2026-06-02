cmake_minimum_required(VERSION 3.16)
project(polyshark CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# Python Development component is required by the visualizer's embedded
# interpreter (pybind11::embed). The bindings module only needs the
# Interpreter+Module sides, but finding Development covers both.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

add_subdirectory(engine)
add_subdirectory(bindings)
add_subdirectory(visualizer)
