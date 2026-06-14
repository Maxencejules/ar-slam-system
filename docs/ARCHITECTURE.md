# Architecture

This document describes how the system is structured, how data flows through it,
and the reasoning behind the main design decisions.

## Module map

```
include/core/
  frame.h               Frame: owns an image + extracted ORB features
  feature_tracker.h     FeatureTracker: KLT tracking + RANSAC + re-detection
  geometry.h            Dependency-free multi-view geometry (eigensolver, DLT)
  reconstruction.h      TwoViewReconstruction: essential matrix -> pose -> 3D
  incremental_mapper.h  IncrementalMapper: keyframes + parallax gating
  memory_pool.h         MemoryPool<T>: fixed-capacity O(1) object pool
  log.h                 Opt-in verbose logging for the core library
include/rendering/
  gl_viewer.h           GLViewer: OpenGL 3.3 point-cloud renderer
include/camera/
  camera_interface.h    Abstract capture interface
  v4l2_camera.h         Linux V4L2 capture declaration (interface stub)
src/
  camera_3d_test.cpp    Full mapping demo (tracking + reconstruction + 3D)
  camera_test.cpp       Lightweight tracking-only viewer
  core/*.cpp            Implementations of the core modules
  rendering/gl_viewer.cpp
tests/
  test_util.h           Minimal assertion helpers
  unit/                 Headless, deterministic tests (run in CI)
  benchmark/            Standalone performance harnesses (opt-in)
```

## Data flow

1. **Capture.** A frame arrives from `cv::VideoCapture` (or a `CameraInterface`
   implementation) as a `cv::Mat`.
2. **Frame.** Wrapped in `ar_slam::Frame`, which converts to grayscale and, on
   demand, extracts ORB keypoints and descriptors.
3. **Tracking.** `FeatureTracker` propagates features from the previous frame with
   pyramidal Lucas–Kanade optical flow, rejects outliers with a fundamental-matrix
   RANSAC pass, and assigns each surviving feature a **stable track id**. When
   quality drops it re-detects and tops the track set back up.
4. **Mapping.** `IncrementalMapper` keeps a reference keyframe (track id → pixel).
   Each update it matches the current tracks to the reference by id, measures the
   median parallax, and once the baseline is wide enough hands the matched
   correspondences to reconstruction. A successful reconstruction promotes the
   current frame to the new keyframe.
5. **Reconstruction.** `TwoViewReconstruction` estimates the essential matrix
   (RANSAC), recovers relative pose under the cheirality constraint, and
   triangulates inliers via the DLT solver in `geometry.h`.
6. **Visualization.** `GLViewer` renders the resulting point cloud with depth-based
   coloring; the demo also draws 2D overlays (tracks, trails, quality, mapping
   status) on the camera image.

## Design decisions

**Dependency-free geometry core.** The eigensolver and triangulation live in a
header that depends only on the standard library. This makes the numerically
delicate math (null-space extraction, DLT conditioning) unit-testable without a
camera, an image, or even OpenCV — and the reconstruction back-end reuses exactly
the code that the tests exercise.

**Track ids as the matching primitive.** Tracking emits a stable id per feature, so
the mapper can associate observations across frames by id rather than re-matching
descriptors. This keeps keyframe correspondence cheap and unambiguous.

**Parallax-gated keyframes.** Triangulation is ill-conditioned with a short
baseline, so the mapper waits for sufficient median parallax before reconstructing
and only then advances the keyframe. This avoids triangulating noise on
near-stationary frames.

**Real, honest depth.** Earlier iterations visualized heuristic/synthetic depth.
The current pipeline shows only depth that was actually triangulated from recovered
motion; until enough parallax accrues, the viewer shows tracked features on a
frontal plane rather than inventing depth.

**Fixed-capacity pool.** `MemoryPool<T>` pre-allocates one contiguous slab and hands
out slots from an intrusive free-list. Allocation and deallocation are O(1) and
never touch the heap after construction, and the capacity is a hard ceiling — the
behavior expected in a real-time, memory-constrained perception pipeline.

## Coordinate conventions

Following Hartley & Zisserman: a world point `X` projects to image point `x` via
`x ~ P X` with `P = K [R | t]`. View 1 is the world origin (`R = I, t = 0`); a world
point maps into view 2 by `X₂ = R X + t`. Monocular reconstruction is defined up to
a global scale, so `t` is unit-length and triangulated structure is scaled to the
baseline.
