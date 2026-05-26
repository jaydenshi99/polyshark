include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(polyshark bindings.cpp ../ai/tdl/mcts.cpp)
target_link_libraries(polyshark PRIVATE engine)
target_include_directories(polyshark PRIVATE
    ${CMAKE_SOURCE_DIR}/engine/include
    ${CMAKE_SOURCE_DIR}/ai/tdl)
