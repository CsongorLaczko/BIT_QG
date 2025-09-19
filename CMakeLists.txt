cmake_minimum_required(VERSION 3.28)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(QGFEM)

# run/test Neumann-Neumann
set(SOURCES
  src/mf_quantum_graph.cpp
  src/measure_nn.cpp
)

add_executable(measure_nn ${SOURCES})

target_include_directories(measure_nn
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/external/eigen-3.4.0
)
