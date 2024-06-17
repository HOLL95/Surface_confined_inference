# Set the minimum required CMake version
cmake_minimum_required(VERSION 3.1)

# Specify project name and languages
project(SurfaceODESolver)
include(CMakePrintHelpers)

# Set C++ compiler flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Set the include directory
set(INCLUDE_DIR "headers")


if (DEFINED ENV{IN_VIKING})
  set(IN_VIKING $ENV{IN_VIKING})
else()
  set(IN_VIKING "false")
endif()

if (IN_VIKING STREQUAL "true")
  set(CMAKE_PREFIX_PATH "/opt/apps/eb/software/SUNDIALS/2.7.0-foss-2021b" ${CMAKE_PREFIX_PATH})
  set(CMAKE_LIBRARY_PATH "/opt/apps/eb/software/SUNDIALS/2.7.0-foss-2021b/lib" ${CMAKE_LIBRARY_PATH})
  set(CMAKE_INCLUDE_PATH "/opt/apps/eb/software/SUNDIALS/2.7.0-foss-2021b/include" ${CMAKE_INCLUDE_PATH})
  find_package(SUNDIALS REQUIRED)
  set(PYBIND11_DIR  /opt/apps/eb/software/pybind11/2.11.1-GCCcore-12.3.0/share/cmake/pybind11)
  find_package(PYBIND11 REQUIRED PATHS ${PYBIND11_DIR})
else()
  add_subdirectory("../lib/pybind11" pybind11)
  include_directories("../lib")

  # Specify the path to SUNDIALSConfig.cmake
  if (DEFINED ENV{CVODE_PATH})
    set(SUNDIALS_DIR $ENV{CVODE_PATH})
  else()
    set(SUNDIALS_DIR /path/to/env)
  endif()

  find_package(SUNDIALS REQUIRED PATHS ${SUNDIALS_DIR})
endif()

# Link SUNDIALS libraries
link_libraries(SUNDIALS::cvode SUNDIALS::nvecserial SUNDIALS::sunlinsoldense SUNDIALS::core SUNDIALS::sunmatrixdense)

# Add pybind11 module
pybind11_add_module(SurfaceODESolver SHARED ODE_simulator_sundials.cpp headers/functions.cpp)

# Specify include directories
target_include_directories(SurfaceODESolver PRIVATE ${INCLUDE_DIR})
