# --- Core SLAM library (tracking front-end + reconstruction back-end) ----
add_library(slam_core STATIC
        core/frame.cpp
        core/feature_tracker.cpp
        core/reconstruction.cpp
        core/incremental_mapper.cpp
)

target_include_directories(slam_core PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(slam_core PUBLIC
        ${OpenCV_LIBS}
)

# --- OpenGL point-cloud renderer -----------------------------------------
add_library(rendering STATIC
        rendering/gl_viewer.cpp
)

target_include_directories(rendering PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIRS}
        ${GLEW_INCLUDE_DIRS}
)

target_link_libraries(rendering PUBLIC
        ${OpenCV_LIBS}
        ${OPENGL_LIBRARIES}
        GLEW::GLEW
        glfw
)

# --- Applications --------------------------------------------------------
# Full monocular mapping demo: tracking + two-view reconstruction + 3D cloud.
add_executable(camera_3d camera_3d_test.cpp)
target_link_libraries(camera_3d
        slam_core
        rendering
        ${OpenCV_LIBS}
        Threads::Threads
)

# Lightweight real-time tracking viewer (2D overlays, features on a plane).
add_executable(camera_test camera_test.cpp)
target_link_libraries(camera_test
        slam_core
        rendering
        ${OpenCV_LIBS}
        Threads::Threads
)
