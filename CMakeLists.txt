cmake_minimum_required(VERSION 3.8)
PROJECT(zzz)

find_package(ament_cmake_auto REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)

set(SOURCES
    src/main.cpp 
    src/use_onnx.cpp 
)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable(Ause ${SOURCES})
ament_target_dependencies(
    Ause
    "OpenCV"
)