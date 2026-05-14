find_package(raylib REQUIRED)

add_executable(visualizer visualizer.cpp)
target_link_libraries(visualizer engine raylib)
target_include_directories(visualizer PRIVATE ${CMAKE_SOURCE_DIR}/engine/include)
