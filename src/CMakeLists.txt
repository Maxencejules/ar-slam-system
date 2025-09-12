# Create core library
add_library(slam_core STATIC
        core/frame.cpp
        core/feature_tracker.cpp
)

# Add include directories for the library
target_include_directories(slam_core PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

# Link OpenCV to slam_core
target_link_libraries(slam_core PUBLIC
        ${OpenCV_LIBS}
)

# Create main executable (ONLY main.cpp)
add_executable(ar_slam_app main.cpp)

# Link libraries to main executable
target_link_libraries(ar_slam_app
        slam_core
        ${OpenCV_LIBS}
        Threads::Threads
)

# Create SEPARATE camera test executable
add_executable(camera_test camera_test.cpp)

# Link libraries to camera test
target_link_libraries(camera_test
        slam_core
        ${OpenCV_LIBS}
        Threads::Threads
)