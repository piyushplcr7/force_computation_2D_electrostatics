cmake_minimum_required(VERSION 2.8)

include(ExternalProject)

project(FCSC)

ExternalProject_Add(Code
  GIT_REPOSITORY "https://gitlab.math.ethz.ch/AdvNumCSE/Code.git"
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/parametricbem2d
)

add_subdirectory(examples)

