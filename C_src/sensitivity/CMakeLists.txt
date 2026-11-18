# CVODES forward-sensitivity module SurfaceODESensitivity. Built by `pip install -e .` (setup.py, next to SurfaceODESolver, which passes
# CMAKE_LIBRARY_OUTPUT_DIRECTORY so the .so lands in the repository root), or standalone:
#   cmake -S C_src/sensitivity -B build -DPYTHON_EXECUTABLE=$(which python) -DSUNDIALS_DIR=<dir with SUNDIALSConfig.cmake>
# SUNDIALS discovery mirrors ../CMakeLists.txt: IN_VIKING / IN_ARC environment variables select the cluster module, otherwise CVODE_PATH
# (or -DSUNDIALS_DIR) names the directory holding SUNDIALSConfig.cmake. Works with SUNDIALS 6.x and 7.x (the source detects the version).
cmake_minimum_required(VERSION 3.12)
project(SurfaceODESensitivity LANGUAGES C CXX)   # C is needed: SUNDIALS built with MPI links MPI::MPI_C, so find_package(MPI) must find the C component
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

foreach(CLUSTER IN_ARC IN_VIKING)
  set(${CLUSTER} "false")
  if(DEFINED ENV{${CLUSTER}})
    set(${CLUSTER} $ENV{${CLUSTER}})
  endif()
endforeach()
find_package(MPI COMPONENTS C CXX)

if(IN_VIKING STREQUAL "true")
  set(SUNDIALS_DIR /opt/apps/eb/software/SUNDIALS/7.0.0-foss-2023a//lib64/cmake/sundials/)
  set(pybind11_DIR /opt/apps/eb/software/pybind11/2.11.1-GCCcore-12.3.0/share/cmake/pybind11)
  find_package(SUNDIALS REQUIRED PATHS ${SUNDIALS_DIR})
  find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})
elseif(IN_ARC STREQUAL "true")
  set(SUNDIALS_DIR /apps/system/easybuild/software/SUNDIALS/6.3.0-foss-2021b/lib64/cmake/sundials)
  set(pybind11_DIR /apps/system/easybuild/software/pybind11/2.11.1-GCCcore-12.3.0/share/cmake/pybind11)
  find_package(SUNDIALS REQUIRED PATHS ${SUNDIALS_DIR})
  find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})
else()
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../lib/pybind11" pybind11)
  if(DEFINED ENV{CVODE_PATH} AND NOT SUNDIALS_DIR)
    set(SUNDIALS_DIR $ENV{CVODE_PATH})
  endif()
  find_package(SUNDIALS REQUIRED PATHS ${SUNDIALS_DIR})
endif()

pybind11_add_module(SurfaceODESensitivity SHARED sens_simulator.cpp)
find_package(Threads REQUIRED)
target_link_libraries(SurfaceODESensitivity PRIVATE Threads::Threads SUNDIALS::cvodes SUNDIALS::nvecserial SUNDIALS::sunlinsoldense SUNDIALS::sunmatrixdense)
