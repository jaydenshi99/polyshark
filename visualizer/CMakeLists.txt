find_package(raylib REQUIRED)

add_executable(visualizer visualizer.cpp)
# pybind11::embed links libpython so the visualizer can host an interpreter
# (used to call heuristics.py from a UI button). pybind11 itself is fetched by
# bindings/CMakeLists.txt, which is configured before this directory.
target_link_libraries(visualizer engine raylib pybind11::embed)
target_include_directories(visualizer PRIVATE ${CMAKE_SOURCE_DIR}/engine/include)
