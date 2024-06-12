# Set the minimum required cmake version
cmake_minimum_required(VERSION 3.1)

# Set cache variables for C compilers and flags
# Specify project name and languages
project(SurfaceODESolver)
include(CMakePrintHelpers)


set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(INCLUDE_DIR, "headers")


add_subdirectory("../lib/pybind11" pybind11)
include_directories("../lib")





# Specify the path to SUNDIALSConfig.cmake
if (DEFINED ENV{CVODE_PATH})
  set (SUNDIALS_DIR $ENV{CVODE_PATH})
else()
  set (SUNDIALS_DIR /path/to/env)
endif()
find_package(SUNDIALS REQUIRED PATHS  ${SUNDIALS_DIR})

link_libraries(SUNDIALS::cvode SUNDIALS::nvecserial SUNDIALS::sunlinsoldense SUNDIALS::core SUNDIALS::sunmatrixdense)
pybind11_add_module(SurfaceODESolver SHARED ODE_simulator_sundials.cpp headers/functions.cpp)
target_include_directories(SurfaceODESolver  PRIVATE
                           ${INCLUDE_DIR})