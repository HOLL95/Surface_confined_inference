# Set the minimum required cmake version
cmake_minimum_required(VERSION 3.1)

# Set cache variables for C compilers and flags
# Specify project name and languages
project(CVODE_test)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(PYBIND11_PYTHON_VERSION "3.12")
set(INCLUDE_DIR, "headers")
add_subdirectory("/home/henryll/Documents/General_electrochemistry/lib/pybind11" pybind11)
include_directories("/home/henryll/Documents/General_electrochemistry/lib")

# Enable testing
include(CTest)
#include(CMakePrintHelpers)

#@EXAMPLES_FIND_PACKAGE@
# ------------------------------------------------------------------------------
#add_executable(myexec BV_test.cpp)

# Specify the path to SUNDIALSConfig.cmake
set(SUNDIALS_DIR
  "/home/henryll/Documents/Software/cvode-7.0.0/builddir")
#cmake_print_variables(SUNDIALS_DIR)
# Find SUNDIALS
find_package(SUNDIALS
             REQUIRED NO_DEFAULT_PATH
             PATHS "/home/henryll/Software/cvode-7.0.0/buildir")

link_libraries(SUNDIALS::cvode SUNDIALS::nvecserial SUNDIALS::sunlinsoldense SUNDIALS::core SUNDIALS::sunmatrixdense)
pybind11_add_module(CVODE_test SHARED BV_test.cpp headers/functions.cpp)
target_include_directories(CVODE_test  PRIVATE
                           ${INCLUDE_DIR})