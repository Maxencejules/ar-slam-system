# Core tests
add_executable(test_features unit/test_features.cpp)
target_link_libraries(test_features
        slam_core
        ${OpenCV_LIBS}
)

# Benchmark tests
add_executable(benchmark_slam benchmark/benchmark_main.cpp)
target_link_libraries(benchmark_slam
        slam_core
        ${OpenCV_LIBS}
)

# Performance test
add_executable(performance_test benchmark/performance_test.cpp)
target_link_libraries(performance_test
        slam_core
        ${OpenCV_LIBS}
        Threads::Threads
)

# Memory test
add_executable(memory_test unit/memory_test.cpp)
target_link_libraries(memory_test
        slam_core
        ${OpenCV_LIBS}
)

# Remove or comment out these lines that reference non-existent files:
# add_executable(test_real_sequences unit/test_real_sequences.cpp)
# add_executable(test_failure_modes unit/test_failure_modes.cpp)

# GTest if you're using it
if(GTEST_FOUND)
    add_executable(gtest_features unit/test_features.cpp)
    target_link_libraries(gtest_features
            slam_core
            ${OpenCV_LIBS}
            GTest::gtest
            GTest::gtest_main
    )
endif()