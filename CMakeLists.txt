cmake_minimum_required(VERSION 3.16)
project(ar_slam_system VERSION 0.1.0 LANGUAGES CXX C)

# Project settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Find packages - OpenCV is REQUIRED now
find_package(OpenCV REQUIRED)  # Changed from QUIET to REQUIRED
find_package(Eigen3 QUIET)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Threads REQUIRED)

# Include directories
include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIRS}
        ${GLEW_INCLUDE_DIRS}
)

# Status messages
message(STATUS "OpenCV found: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# Add Eigen3 if found
if(Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
    message(STATUS "Eigen3 found")
else()
    message(WARNING "Eigen3 not found. Some features will be disabled.")
endif()

# Add subdirectories
add_subdirectory(src)

# Enable testing
enable_testing()

# Option to build tests
option(BUILD_TESTS "Build test programs" ON)

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()