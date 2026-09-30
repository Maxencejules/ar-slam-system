#pragma once

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
    inline void check_near(
        double a, double b, double tol, const char* expr, const char* file, int line) {
        if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(tol) || tol < 0 ||
            std::fabs(a - b) > tol) {
            std::printf("  [FAIL] %s:%d: %s  (%.6g vs %.6g; tolerance %.3g)\n", file, line, expr, a,
                        b, tol);
            ++failures();
        }
    }
    template <typename Exception, typename Function>
    bool throws(Function&& function) {
        try {
            function();
        } catch (const Exception&) {
            return true;
        } catch (...) {
            return false;
        }
        return false;
    }
    inline int report(const char* name) {
        std::printf("%s: %s (%d failed checks)\n", failures() == 0 ? "PASSED" : "FAILED", name,
                    failures());
        return failures() == 0 ? 0 : 1;
    }
}  // namespace artest

#define CHECK(cond) ::artest::check((cond), #cond, __FILE__, __LINE__)
#define CHECK_NEAR(a, b, tol) \
    ::artest::check_near((a), (b), (tol), #a " ~= " #b, __FILE__, __LINE__)
