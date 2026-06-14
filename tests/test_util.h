#pragma once

// Minimal, dependency-free assertion helpers for the headless test suite.
// Each test is a standalone executable that returns non-zero on failure, which
// CTest (and the CI workflow) use to determine pass/fail.

#include <cmath>
#include <cstdio>

namespace artest {

inline int& failures() {
    static int f = 0;
    return f;
}

inline void check(bool cond, const char* expr, const char* file, int line) {
    if (!cond) {
        std::printf("  [FAIL] %s:%d: %s\n", file, line, expr);
        ++failures();
    }
}

inline void check_near(double a, double b, double tol, const char* expr, const char* file,
                       int line) {
    if (std::fabs(a - b) > tol) {
        std::printf("  [FAIL] %s:%d: %s  (|%.6g - %.6g| = %.3g > %.3g)\n", file, line, expr, a, b,
                    std::fabs(a - b), tol);
        ++failures();
    }
}

inline int report(const char* name) {
    if (failures() == 0) {
        std::printf("PASSED: %s\n", name);
        return 0;
    }
    std::printf("FAILED: %s (%d check(s))\n", name, failures());
    return 1;
}

}  // namespace artest

#define CHECK(cond) ::artest::check((cond), #cond, __FILE__, __LINE__)
#define CHECK_NEAR(a, b, tol) ::artest::check_near((a), (b), (tol), #a " ~= " #b, __FILE__, __LINE__)
