# Create rendering library
add_library(rendering STATIC
        rendering/gl_viewer.cpp
)

target_include_directories(rendering PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(rendering PUBLIC
        ${OPENGL_LIBRARIES}
        GLEW::GLEW
        glfw
)

# Create core library
add_library(slam_core STATIC
        core/frame.cpp
        core/feature_tracker.cpp
)

target_include_directories(slam_core PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(slam_core PUBLIC
        ${OpenCV_LIBS}
)

# Main executable
add_executable(ar_slam_app main.cpp)
target_link_libraries(ar_slam_app
        slam_core
        ${OpenCV_LIBS}
        Threads::Threads
)

# Camera test - THIS NEEDS rendering library too!
add_executable(camera_test camera_test.cpp)
target_link_libraries(camera_test
        slam_core
        rendering  # ADD THIS LINE
        ${OpenCV_LIBS}
        ${OPENGL_LIBRARIES}
        GLEW::GLEW
        glfw
        Threads::Threads
)

# 3D Camera test
add_executable(camera_3d camera_3d_test.cpp)
target_link_libraries(camera_3d
        slam_core
        rendering
        ${OpenCV_LIBS}
        ${OPENGL_LIBRARIES}
        GLEW::GLEW
        glfw
        Threads::Threads
)