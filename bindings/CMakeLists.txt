include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
)
# Use the modern find_package(Python) finder instead of pybind11's bundled
# PythonLibsNew (the latter sets PYTHON_LIBRARY incorrectly for libpython on
# some macOS installs, breaking pybind11::embed linking).
set(PYBIND11_FINDPYTHON ON)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(polyshark bindings.cpp ../ai/tdl/mcts.cpp)
target_link_libraries(polyshark PRIVATE engine)
target_include_directories(polyshark PRIVATE
    ${CMAKE_SOURCE_DIR}/engine/include
    ${CMAKE_SOURCE_DIR}/ai/tdl)
