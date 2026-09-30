add_library(slam_core STATIC
    core/frame.cpp
    core/feature_tracker.cpp
    core/reconstruction.cpp
    core/incremental_mapper.cpp
)
target_include_directories(slam_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(slam_core SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(slam_core PUBLIC ${OpenCV_LIBS})
add_executable(offline_demo offline_demo.cpp)
target_include_directories(offline_demo PRIVATE ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(offline_demo PRIVATE slam_core)
if(BUILD_VIEWERS)
    add_library(rendering STATIC rendering/gl_viewer.cpp)
    target_include_directories(rendering PUBLIC ${PROJECT_SOURCE_DIR}/include)
    target_include_directories(rendering SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(rendering PUBLIC ${OpenCV_LIBS} OpenGL::GL GLEW::GLEW glfw glm::glm)
    add_executable(camera_3d camera_3d_test.cpp)
    target_link_libraries(camera_3d PRIVATE slam_core rendering Threads::Threads)
    add_executable(camera_test camera_test.cpp)
    target_link_libraries(camera_test PRIVATE slam_core rendering Threads::Threads)
endif()
