#pragma once

#include <iostream>

namespace ar_slam {

/// Process-wide verbosity flag for the core library. Off by default so the
/// library stays silent unless an application explicitly opts in.
inline bool& verbose_logging() {
    static bool enabled = false;
    return enabled;
}

/// Enable or disable diagnostic logging from the core library.
inline void set_verbose_logging(bool enabled) { verbose_logging() = enabled; }

}  // namespace ar_slam

/// Stream a diagnostic line only when verbose logging is enabled, e.g.
/// `AR_LOG("tracked " << n << " features");`
#define AR_LOG(stream_expr)                          \
    do {                                             \
        if (::ar_slam::verbose_logging()) {          \
            std::cout << stream_expr << std::endl;   \
        }                                            \
    } while (0)
