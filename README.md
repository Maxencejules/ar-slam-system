# AR SLAM System

A real-time **monocular visual tracking and structure-from-motion** system written
from scratch in modern C++. It implements the geometric core of an augmented-reality
tracker — feature detection, optical-flow tracking, robust outlier rejection, and
**two-view 3D reconstruction** — without relying on an existing SLAM library, to
demonstrate the underlying computer-vision and multi-view-geometry fundamentals.

[![CI](https://github.com/Maxencejules/ar-slam-system/actions/workflows/ci.yml/badge.svg)](https://github.com/Maxencejules/ar-slam-system/actions/workflows/ci.yml)
![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![OpenGL](https://img.shields.io/badge/OpenGL-3.3-red.svg)
![License](https://img.shields.io/badge/license-MIT-purple.svg)

---

## What this is (and what it isn't)

A precise breakdown of what the system does and does not do:

| ✅ Implemented | ⛔ Not (yet) implemented |
|---|---|
| ORB feature detection | Pose graph / global bundle adjustment |
| Pyramidal KLT optical-flow tracking | Loop-closure detection |
| RANSAC fundamental/essential-matrix outlier rejection | Dense reconstruction |
| Two-view relative pose (essential matrix + cheirality) | IMU / inertial fusion |
| **Real DLT triangulation of 3D structure** | Metric scale (monocular is scale-ambiguous) |
| Keyframe-based incremental mapping | Relocalization after total tracking loss |
| Fixed-capacity O(1) object pool | |
| OpenGL 3.3 point-cloud visualization | |

This is the **visual front-end and two-view reconstruction back-end** of a
monocular SLAM pipeline — the layer that detects features, tracks them, and
triangulates 3D structure from camera motion. The points in the demo are
triangulated from recovered motion, so the cloud is real geometry. The
components a production SLAM/VIO system adds on top — global optimization, loop
closure, and metric scale — are outlined in the [roadmap](#roadmap).

## Pipeline

```
 camera frame
      │
      ▼
┌───────────────┐   ORB keypoints + descriptors
│  Frame        │──────────────────────────────┐
└───────────────┘                               │
      │ grayscale                               │
      ▼                                         ▼
┌───────────────┐   tracked correspondences   ┌──────────────────────┐
│ FeatureTracker│────────────────────────────▶│ IncrementalMapper    │
│ (KLT + RANSAC)│      (id-stable tracks)      │  • keyframe selection │
└───────────────┘                             │  • parallax gating     │
                                              └──────────┬───────────┘
                                                         │ matched views
                                                         ▼
                                              ┌──────────────────────┐
                                              │ TwoViewReconstruction │
                                              │  • essential matrix    │
                                              │  • recoverPose         │
                                              │  • DLT triangulation   │
                                              └──────────┬───────────┘
                                                         │ 3D points + pose
                                                         ▼
                                                  ┌──────────────┐
                                                  │  GLViewer    │
                                                  └──────────────┘
```

## Key components

- **`core/geometry.h`** — dependency-free multi-view geometry: a 4×4 symmetric
  Jacobi eigensolver and DLT triangulation (Hartley & Zisserman). Pure C++ so the
  math is unit-tested in isolation.
- **`core/feature_tracker`** — ORB detection, pyramidal Lucas–Kanade optical flow,
  RANSAC outlier rejection, automatic re-detection and feature top-up.
- **`core/reconstruction`** — estimates the essential matrix with RANSAC, decomposes
  it into a relative pose via the cheirality (positive-depth) constraint, and
  triangulates inliers using the geometry core.
- **`core/incremental_mapper`** — keyframe management: matches tracks by id, gates
  on parallax, and triggers reconstruction once the baseline is wide enough.
- **`core/memory_pool.h`** — a fixed-capacity object pool backed by a single
  contiguous slab with an intrusive free-list: **true O(1)** allocate/deallocate and
  a hard, enforced capacity (suitable for latency- and memory-constrained pipelines).
- **`rendering/gl_viewer`** — OpenGL 3.3 core-profile point-cloud renderer with
  depth-based coloring, a ground-plane grid, and orbit controls.

## Building

### Prerequisites (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake \
    libopencv-dev libgl1-mesa-dev libglew-dev libglfw3-dev libglm-dev libeigen3-dev
```

The **core library and its tests depend only on OpenCV** and standard C++17; OpenGL,
GLFW and GLEW are only needed for the interactive viewers.

### Configure & build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Useful options: `-DBUILD_TESTS=ON` (default), `-DBUILD_BENCHMARKS=ON`,
`-DENABLE_NATIVE_ARCH=ON` (adds `-march=native` for the local CPU),
`-DWARNINGS_AS_ERRORS=ON`.

### Run

```bash
./build/src/camera_3d     # full mapping demo: tracking + two-view reconstruction
./build/src/camera_test   # lightweight real-time tracking viewer
```

| Key | Action |
|-----|--------|
| Arrow keys | Orbit / zoom the 3D view |
| `R` | Reset the tracker (camera_test) |
| Space | Print statistics |
| `Q` / Esc | Quit |

## Testing

The suite is **headless and deterministic** — it uses synthetic images and
synthetic two-view scenes, so it runs unattended in CI (no webcam required):

```bash
cd build && ctest --output-on-failure
```

| Test | Verifies |
|------|----------|
| `test_geometry` | Jacobi eigensolver; DLT triangulation recovers known 3D points to numerical precision, and stays accurate under sub-pixel noise |
| `test_memory_pool` | Capacity derivation, O(1) slab reuse, enforced exhaustion, construction/destruction, move semantics |
| `test_reconstruction` | End-to-end: synthetic scene → projected into two cameras → recovered pose and structure match ground truth (up to scale) |
| `test_tracking` | ORB extraction counts; KLT tracking quality under known motion; tracker reset |

Standalone benchmarks (`-DBUILD_BENCHMARKS=ON`) report mean/stddev/min/max timings
for feature extraction, tracking, the memory pool, and the full pipeline under
synthetic motion with noise, blur and lighting variation. Run them to reproduce
performance numbers on your own hardware.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module responsibilities, the
data-flow walkthrough, and the rationale behind the main design decisions.

## Technical notes

**Feature tracking.** ORB keypoints are tracked frame-to-frame with pyramidal
Lucas–Kanade optical flow over a 4-level (`maxLevel = 3`) image pyramid. Tracks failing the optical-flow status/error checks or
leaving the image are dropped; a fundamental-matrix RANSAC pass removes
epipolar-inconsistent matches. When tracked count or quality falls below threshold,
features are re-detected, and a masked detector tops the track set back up so the
distribution stays even.

**Two-view geometry.** Relative motion is recovered from the essential matrix
(RANSAC) and decomposed with the cheirality constraint so the solution places
points in front of both cameras. Each inlier is triangulated with a row-normalized
DLT solved as the smallest-eigenvalue null space of `AᵀA`. Because monocular
reconstruction is scale-ambiguous, translation is unit-length and structure is
defined up to a global scale.

**Memory pool.** A single over-aligned slab is carved into slots threaded onto an
intrusive free-list, giving constant-time allocation/deallocation with zero
post-construction heap traffic and a hard capacity ceiling.

## Roadmap

The natural path from this front-end to a complete SLAM system:

1. **Local bundle adjustment** over a sliding keyframe window.
2. **PnP-based pose tracking** against the existing map (frame-to-map, not just
   frame-to-frame).
3. **Loop-closure detection** (e.g. DBoW) with pose-graph optimization.
4. **IMU pre-integration** for metric scale and robustness (visual-inertial odometry).
5. **Mobile deployment** (Android NDK / ARM NEON).

## License

MIT — see [LICENSE](LICENSE).

## Contact

**Maxence Jules** · [GitHub @Maxencejules](https://github.com/Maxencejules) ·
[LinkedIn](https://linkedin.com/in/julesmax)
