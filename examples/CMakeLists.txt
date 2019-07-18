cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-std=c++11")

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/external/Code/BEM/2DParametricBEM/include)

add_executable(dinteg double_integral.cpp)

target_link_libraries(dinteg parametrizations adj_double_layer hypersingular single_layer double_layer quadrature CppHilbert)



