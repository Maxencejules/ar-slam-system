# tests/CMakeLists.txt

# Find Google Test (optional)
find_package(GTest QUIET)

# Main benchmark executable
add_executable(benchmark_slam
        benchmark/benchmark_main.cpp
)

target_link_libraries(benchmark_slam
        slam_core
        rendering
        ${OpenCV_LIBS}
        ${OPENGL_LIBRARIES}
        GLEW::GLEW
        glfw
        Threads::Threads
)

# Performance stress test
add_executable(performance_test
        benchmark/performance_test.cpp
)

target_link_libraries(performance_test
        slam_core
        ${OpenCV_LIBS}
        Threads::Threads
)

# Memory pool test
add_executable(memory_test
        unit/memory_test.cpp
)

target_link_libraries(memory_test
        slam_core
        ${OpenCV_LIBS}
        Threads::Threads
)

# Feature extraction test
add_executable(test_features
        unit/test_features.cpp
)

target_link_libraries(test_features
        slam_core
        ${OpenCV_LIBS}
        Threads::Threads
)

# Add tests to CTest
add_test(NAME BenchmarkSLAM COMMAND benchmark_slam)
add_test(NAME PerformanceTest COMMAND performance_test)
add_test(NAME MemoryTest COMMAND memory_test)
add_test(NAME FeatureTest COMMAND test_features)

# If Google Test is found, add individual unit tests
if(GTest_FOUND)
    message(STATUS "Google Test found - building unit tests")

    # Add individual test files with Google Test
    add_executable(gtest_features
            unit/test_features.cpp
    )

    target_link_libraries(gtest_features
            slam_core
            ${OpenCV_LIBS}
            GTest::gtest
            GTest::gtest_main
            Threads::Threads
    )

    # Enable Google Test discovery
    include(GoogleTest)
    gtest_discover_tests(gtest_features)
else()
    message(STATUS "Google Test not found - basic tests will still run")
endif()

# Custom target to run all tests
add_custom_target(run_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS benchmark_slam performance_test memory_test test_features
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running all tests..."
)

# Set test properties
set_tests_properties(BenchmarkSLAM PROPERTIES TIMEOUT 60)
set_tests_properties(PerformanceTest PROPERTIES TIMEOUT 30)
set_tests_properties(MemoryTest PROPERTIES TIMEOUT 30)
set_tests_properties(FeatureTest PROPERTIES TIMEOUT 30)