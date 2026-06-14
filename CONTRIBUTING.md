# Contributing

Thanks for your interest in the project. This is a personal portfolio project, but
issues and pull requests that improve correctness, clarity, or test coverage are
welcome.

## Development setup

See [README.md](README.md#building) for dependencies and build instructions. In
short:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build --parallel
cd build && ctest --output-on-failure
```

## Coding standards

- **Language:** C++17 (`-DCMAKE_CXX_STANDARD=17`, no compiler extensions).
- **Style:** formatted with `clang-format` using the repository's
  [`.clang-format`](.clang-format). Before sending a PR:
  ```bash
  clang-format -i $(find src include tests -name '*.cpp' -o -name '*.h')
  ```
- **Warnings:** the code builds clean under `-Wall -Wextra`. Configure with
  `-DWARNINGS_AS_ERRORS=ON` to enforce this locally.
- Keep the **geometry core (`include/core/geometry.h`) dependency-free** so it
  remains unit-testable in isolation.

## Tests

Every behavioral change should come with a test. Tests are headless and
deterministic (synthetic data, fixed seeds) so they run in CI without a camera.

- Pure-C++ logic → add to `tests/unit/test_geometry.cpp` /
  `tests/unit/test_memory_pool.cpp` (no third-party deps).
- OpenCV-backed pipeline → add to `tests/unit/test_reconstruction.cpp` /
  `tests/unit/test_tracking.cpp`.

Register new test executables in [`tests/CMakeLists.txt`](tests/CMakeLists.txt) with
`add_test(...)` so CTest and CI pick them up.

## Pull requests

1. Branch from `main`.
2. Keep changes focused; describe the motivation in the PR body.
3. Ensure `ctest` passes and the build is warning-clean.
4. CI (build + tests on Ubuntu) must be green.
