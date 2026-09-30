# AR SLAM System

A C++17 visual frontend and **two-view reconstruction prototype**. OpenCV supplies
ORB, optical flow, robust model estimation and pose decomposition; the project
implements track bookkeeping, pair selection, geometric quality gates and a
dependency-free DLT triangulator. The default build runs offline without a
camera, display, GPU, downloaded dataset or network access.

[![CI](https://github.com/Maxencejules/ar-slam-system/actions/workflows/ci.yml/badge.svg)](https://github.com/Maxencejules/ar-slam-system/actions/workflows/ci.yml)

This is a research/educational prototype. Each cloud belongs to one camera pair
and has its own unknown scale. It does **not** estimate a persistent world
trajectory, align consecutive clouds, perform bundle adjustment, relocalize or
close loops. The optional webcam viewer is a qualitative illustration; no
real-time rate or real-camera accuracy is claimed.

## Build and run offline

Ubuntu/Debian needs a C++17 compiler, CMake 3.16+ and OpenCV 4:

```sh
sudo apt-get update
sudo apt-get install -y build-essential cmake libopencv-dev
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
./build/src/offline_demo --output build/my-demo
python3 scripts/verify_demo.py build/my-demo  # optional standard-library artifact audit
```

On Windows use a Visual Studio developer shell and a matching MSVC OpenCV SDK:

```powershell
cmake -S . -B build -DOpenCV_DIR="C:/opencv/build/x64/vc16/lib"
cmake --build build --config Release --parallel 2
$env:PATH = "C:/opencv/build/x64/vc16/bin;" + $env:PATH
ctest --test-dir build -C Release --output-on-failure
./build/src/Release/offline_demo.exe --output build/my-demo
```

If a Windows launcher does not forward the DLL search path, copy the release
`opencv_world*.dll` beside the disposable Release test/demo executables. Match
Debug executables to the debug DLL. The recorded run used the official OpenCV
4.14.0 MSVC SDK and MSVC 19.44.35228, CMake 4.2.1, Ninja, Release, `/W4 /WX`.
No GUI or webcam was used.

Options:

| Option | Default | Effect |
|---|---|---|
| `BUILD_TESTS` | ON | Six bounded headless CTests |
| `BUILD_VIEWERS` | OFF | Optional camera/OpenGL executables and their dependencies |
| `WARNINGS_AS_ERRORS` | ON | `-Wall -Wextra -Werror` or MSVC `/W4 /WX` |
| `ENABLE_SANITIZERS` | OFF | GCC/Clang AddressSanitizer + UndefinedBehaviorSanitizer |
| `ENABLE_NATIVE_ARCH` | OFF | Local CPU tuning for GCC/Clang Release |
| `BUILD_BENCHMARKS` | OFF | Legacy timing harnesses; requires `BUILD_TESTS=ON` |

## Recorded synthetic demonstration

[examples/offline](examples/offline) contains actual output from
`offline_demo --output examples/offline`, with seed 2026, one OpenCV thread and
OpenCL disabled. These are **generated synthetic observations**, not a camera
recording. The executable verifies accuracy bounds before exporting:

| Experiment | Accepted observations | Rotation error | Translation-direction error | 3D RMSE |
|---|---:|---:|---:|---:|
| Clean known non-planar scene | 240/240 | 0.0000064° | 0.0000245° | 0.0000042 scene units |
| ±0.15 px uniform noise + 35 mismatches | 205/240 | 0.113° | 2.549° | 0.0943 scene units |
| Separate (4,3) px image warp | 500 temporal tracks | — | — | 0.00110 px median displacement error |

The geometry experiment uses 240 known non-planar 3D points at depths 4–8
arbitrary scene units, calibrated pinhole projection, a 5° rotation and known
translation (-0.6, 0.02, 0.04). The mismatch case replaces every seventh
second-view observation with another point's pixel. Its initial RANSAC consensus
is 206 observations; 205 survive pose/depth/angle/reprojection gates, including
zero known mismatches. RMSE counts retained true correspondences only; rejected
points and retained mismatches are reported separately.

Recovered translation has norm 1. Multiplying estimated structure by the
**known synthetic baseline** 0.601664 scene units permits the ground-truth RMSE.
This scale is available to the experiment only; a monocular deployment cannot
infer it from these pixels alone.

The frontend experiment independently tracks a known 2D affine warp. The mapper
experiment receives known correspondence IDs from the geometric fixture. This
demonstration does not claim an end-to-end reconstruction from those warped
images.

| Artifact | Content |
|---|---|
| `scene.csv` | Known 3D scene, noisy pixels, explicit mismatch flags |
| `estimated_points.csv` | Accepted input indices and estimated camera-1 coordinates |
| `cloud.ply` | Same estimated synthetic cloud, in unit-baseline coordinates |
| `tracking.csv` | Previous/current pixels, IDs and temporal-match flags |
| `metrics.csv` | Model consensus, accepted counts, ground-truth and quality metrics |
| `report.json` | Seed, noise, baseline, compiler/OpenCV and frontend/pair provenance |

Repeated runs in the recorded environment produce the same six files. The optional Python standard-library audit recomputes pose,
structure, reprojection and warp metrics from the exported records and checks
CSV/PLY consistency. OpenCV
versions, floating-point implementations and robust estimation may change
results; tests check independent tolerances rather than exact snapshot bytes.
Fresh CI outputs are uploaded as artifacts; the recorded files are a labeled
reference run.

## Contracts and evidence

- **Tracking:** timestamps increase strictly; image resolution stays fixed until
  `reset()`. Inputs are nonempty 8-bit gray/BGR/BGRA images. Every observation has
  a pixel and unique ID. IDs persist for surviving tracks and are never reused
  within one tracker instance, including after reset. Replacing the instance
  requires resetting the mapper.
- **New observations:** all result arrays have the same size. Newly detected
  features use previous=current as a placeholder and `inliers=false`; they are
  not temporal correspondences. Quality is retained previous observations divided
  by the previous count, before top-up. Initialization has quality zero.
- **Reconstruction:** matched pixels must already be undistorted and use a known,
  positive-focal, zero-skew pinhole calibration at the same image resolution.
  Before model fitting, median observed pixel displacement must exceed the
  configured RANSAC pixel threshold (default 1 px). This conservative gate rejects
  stationary noise and a stationary majority with moving outliers before an
  arbitrary estimated rotation can fabricate ray separation. It may also skip
  valid motion with very small or cancelled image displacement.
  RANSAC's five-point model is refined from its consensus with calibrated
  eight-point fitting and essential singular values (s,s,0) when at least eight
  inliers exist. Pose recovery and custom DLT then require positive depth in both
  views, a 1° ray angle and at most 2 px reprojection error. Depth limit 100 is in
  baseline units. Consensus count and final accepted count are distinct.
- **Pair mapper:** pixel displacement triggers an attempt; rotation can cause
  displacement without observable depth. The reconstruction angle gate handles
  the tested pure-rotation/low-baseline cases. Clouds carry reference/current
  accepted update indices; later updates mark cached clouds stale. Track loss
  clears the old cloud. Successful new pairs replace it without global alignment.
- **Memory utility:** `MemoryPool<T>` bounds its slab by the requested byte budget,
  including alignment; tiny budgets have zero capacity. Raw-slot operations are
  O(1), single-threaded and require owned slots returned exactly once. Callers
  manage object lifetime; constructors/destructors can allocate. The pool is a
  standalone utility and does not bound OpenCV/frontend heap use. Frame memory
  reporting is approximate owned payload, excluding allocator/temporary overhead.

The tests use independent analytic pinhole projection and known non-planar 3D
truth across three seeds; custom DLT is also compared with OpenCV's separate SVD
triangulator. They check noisy/mismatched scenes, pure rotation, inadequate
baseline, invalid calibration/configuration/nonfinite points, fresh IDs,
timestamps, alignment after top-up/loss, two distinct mapper coordinate frames,
cached-cloud invalidation, strict pool budgets and over-aligned storage.
`CHECK_NEAR` rejects NaNs instead of silently passing them.

These checks are bounded evidence, not a general observability proof. Planar
scenes, poor calibration, repeated texture, large outlier rates and weak baselines
can still give failure or unreliable pose. The normal-equation DLT solver is
educational and less well conditioned than direct SVD in difficult cases; there
is no nonlinear bundle adjustment or general degeneracy classifier.

## Optional viewers and timing harnesses

```sh
sudo apt-get install -y libgl1-mesa-dev libglew-dev libglfw3-dev libglm-dev
cmake -S . -B build-viewers -DBUILD_VIEWERS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-viewers --parallel 2
./build-viewers/src/camera_test
./build-viewers/src/camera_3d
```

A display, OpenGL 3.3 and webcam are required to run these targets. Ubuntu CI
compiles the viewers and runs only headless tests; hardware interaction is not
validated. The 3D viewer guesses intrinsics and centers/rescales each pair cloud
for display. Its initial frontal-plane fallback, cached cloud and new pair cloud
are labeled; display coordinates are not a metric world map.

Legacy `benchmark_slam`/`performance_test` are exploratory synthetic timing
harnesses, not a reproducible accuracy dataset or a real-time guarantee. They
include time-seeded/random workloads and do not provide controlled raw repeated
trials or hardware-normalized comparisons. No performance recommendation follows
from their timings.

See [architecture](docs/ARCHITECTURE.md) and [contributing](CONTRIBUTING.md).
The calibration, pose and triangulation conventions follow the
[OpenCV calib3d documentation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).
