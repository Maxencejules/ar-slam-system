cmake_minimum_required(VERSION 3.16)
project(ar_slam_system VERSION 0.2.0 LANGUAGES CXX C)

# --- Language standard ---------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Build type ----------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- Options -------------------------------------------------------------
option(BUILD_TESTS "Build the unit/integration test suite" ON)
option(BUILD_BENCHMARKS "Build the standalone benchmark executables" OFF)
option(ENABLE_NATIVE_ARCH "Optimise for the local CPU (-march=native)" OFF)
option(WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

# --- Warnings / optimisation flags (GCC/Clang only) ----------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra)
    if(WARNINGS_AS_ERRORS)
        add_compile_options(-Werror)
    endif()
    if(ENABLE_NATIVE_ARCH)
        add_compile_options($<$<CONFIG:Release>:-march=native>)
    endif()
endif()

# --- Dependencies --------------------------------------------------------
find_package(OpenCV REQUIRED)
find_package(Eigen3 QUIET)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Threads REQUIRED)

message(STATUS "OpenCV version:  ${OpenCV_VERSION}")
if(Eigen3_FOUND)
    message(STATUS "Eigen3 found:    ${EIGEN3_VERSION_STRING}")
else()
    message(STATUS "Eigen3 not found (optional; core builds without it)")
endif()

# --- Targets -------------------------------------------------------------
add_subdirectory(src)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
