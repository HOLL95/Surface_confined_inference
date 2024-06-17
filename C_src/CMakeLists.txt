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
  set(SUNDIALS_INCLUDE_DIRS "/opt/apps/eb/software/SUNDIALS/2.7.0-foss-2021b/include")
  set(SUNDIALS_LIBRARY_DIRS "/opt/apps/eb/software/SUNDIALS/2.7.0-foss-2021b/lib")
  set(pybind11_DIR  /opt/apps/eb/software/pybind11/2.11.1-GCCcore-12.3.0/share/cmake/pybind11)
  find_library(SUNDIALS_NVECSERIAL_LIB NAMES sundials_nvecserial PATHS ${SUNDIALS_LIBRARY_DIRS})
  find_library(SUNDIALS_SUNLINSOLDENSE_LIB NAMES sundials_sunlinsoldense PATHS ${SUNDIALS_LIBRARY_DIRS})
  find_library(SUNDIALS_SUNMATRIXDENSE_LIB NAMES sundials_sunmatrixdense PATHS ${SUNDIALS_LIBRARY_DIRS})
  find_library(SUNDIALS_CVODE_LIB NAMES sundials_cvode PATHS ${SUNDIALS_LIBRARY_DIRS})
  find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})
  link_libraries(SUNDIALS::cvode SUNDIALS::nvecserial SUNDIALS::sunlinsoldense SUNDIALS::core SUNDIALS::sunmatrixdense)

  # Add pybind11 module
  pybind11_add_module(SurfaceODESolver SHARED ODE_simulator_sundials.cpp headers/functions.cpp)

  # Specify include directories
  target_include_directories(SurfaceODESolver PRIVATE ${INCLUDE_DIR})
  target_link_libraries(your_target_name
    PUBLIC
        ${SUNDIALS_NVECSERIAL_LIB}
        ${SUNDIALS_SUNLINSOLDENSE_LIB}
        ${SUNDIALS_CVODE_LIB}
)
else()
  add_subdirectory("../lib/pybind11" pybind11)
  include_directories("../lib")

  # Specify the path to SUNDIALSConfig.cmake
  if (DEFINED ENV{CVODE_PATH})
    set(SUNDIALS_DIR $ENV{CVODE_PATH})
  else()
    set(SUNDIALS_DIR /path/to/env)
  endif()

  find_package(SUNDIALS REQUIRED PATHS ${SUNDIALS_DIR})#
  # Link SUNDIALS libraries
  link_libraries(SUNDIALS::cvode SUNDIALS::nvecserial SUNDIALS::sunlinsoldense SUNDIALS::core SUNDIALS::sunmatrixdense)

  # Add pybind11 module
  pybind11_add_module(SurfaceODESolver SHARED ODE_simulator_sundials.cpp headers/functions.cpp)

  # Specify include directories
  target_include_directories(SurfaceODESolver PRIVATE ${INCLUDE_DIR})
endif()


