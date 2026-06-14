# Headless, deterministic test suite. Every test is a standalone executable that
# returns non-zero on failure so CTest (and CI) can gate on it.

# --- Pure-C++ unit tests (no third-party dependencies) -------------------
foreach(pure_test test_geometry test_memory_pool)
    add_executable(${pure_test} unit/${pure_test}.cpp)
    target_include_directories(${pure_test} PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_test(NAME ${pure_test} COMMAND ${pure_test})
endforeach()

# --- Tests that exercise the OpenCV-backed pipeline ----------------------
foreach(cv_test test_reconstruction test_tracking)
    add_executable(${cv_test} unit/${cv_test}.cpp)
    target_include_directories(${cv_test} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(${cv_test} PRIVATE slam_core ${OpenCV_LIBS})
    add_test(NAME ${cv_test} COMMAND ${cv_test})
endforeach()

# --- Optional standalone benchmarks (built with -DBUILD_BENCHMARKS=ON) ----
if(BUILD_BENCHMARKS)
    add_executable(benchmark_slam benchmark/benchmark_main.cpp)
    target_link_libraries(benchmark_slam PRIVATE slam_core ${OpenCV_LIBS})

    add_executable(performance_test benchmark/performance_test.cpp)
    target_link_libraries(performance_test PRIVATE slam_core ${OpenCV_LIBS} Threads::Threads)
endif()
