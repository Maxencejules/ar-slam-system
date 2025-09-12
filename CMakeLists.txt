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

# Find packages (make them optional for now)
find_package(OpenCV QUIET)
find_package(Eigen3 QUIET)
find_package(OpenGL QUIET)
find_package(GLEW QUIET)
find_package(glfw3 QUIET)
find_package(Threads REQUIRED)

# Include directories
include_directories(
        ${PROJECT_SOURCE_DIR}/include
)

# Add OpenCV if found
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
else()
    message(WARNING "OpenCV not found. Some features will be disabled.")
endif()

# Add Eigen3 if found
if(Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
    message(STATUS "Eigen3 found")
else()
    message(WARNING "Eigen3 not found. Some features will be disabled.")
endif()

# Add subdirectories
add_subdirectory(src)

# Only add tests if they exist
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
endif()