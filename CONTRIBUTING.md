# Contributing

Use the [README](README.md) build instructions. Keep changes scoped to the visual
frontend/two-view prototype, and describe coordinate frames, scale, calibration
and temporal assumptions when changing a contract.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

C++17 is required. Warnings are errors by default. Keep `geometry.h` and pool
tests standard-library-only. Format C++ sources with the exact version CI uses:

```sh
python -m venv build/format-env
build/format-env/bin/python -m pip install clang-format==22.1.5
build/format-env/bin/clang-format -i $(find src include tests -type f \( -name '*.cpp' -o -name '*.h' \))
```

Add meaningful regression cases for changed behavior. Geometry assertions should
use independent truth/reference calculations, not only an implementation's own
residuals. Statistical or performance claims need workload, raw trials, compiler,
environment and limitations. Do not label generated synthetic data as real
measurements. Recorded demo updates must come from an actual passing executable;
retain compiler/OpenCV provenance and verify CSV column alignment.

CI compiles viewers on Ubuntu Release and runs bounded headless tests, with a
separate Clang sanitizer job and pinned formatting. Green compilation does not
establish webcam compatibility or real-time behavior.
