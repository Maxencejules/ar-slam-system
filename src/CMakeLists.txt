# Create core library
add_library(slam_core STATIC
        core/frame.cpp
        core/feature_tracker.cpp
        # NO .h files here!
)

# Add include directories for the library
target_include_directories(slam_core PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

# Link OpenCV to slam_core
target_link_libraries(slam_core PUBLIC
        ${OpenCV_LIBS}
)

# Create main executable
add_executable(ar_slam_app main.cpp)

# Link libraries to executable
target_link_libraries(ar_slam_app
        slam_core
        ${OpenCV_LIBS}
        Threads::Threads
)