# Architecture

The default target is a headless visual frontend/two-view reconstruction
prototype. Interactive viewers are optional; a global SLAM estimator is absent.

## Data and ownership

`Frame` clones a nonempty 8-bit image, owns grayscale pixels and extracted ORB
features, and exposes a steady-clock timestamp. Its payload memory estimate
includes stored features/descriptors but excludes allocator overhead and OpenCV
temporary allocations.

`FeatureTracker` accepts strictly ordered, same-resolution frames. Forward/backward
Lucas–Kanade flow checks status, error, bounds and a 1 px round trip. A valid
fundamental-matrix RANSAC mask further rejects flow tracks; if model estimation
fails, flow-validated tracks remain, so no universal epipolar guarantee is implied.
Survivors keep their IDs. Masked ORB detection tops up to at most 500 observations
with new IDs and no previous-frame match. Four result arrays remain aligned.
Retention is measured before top-up; reset clears frame state, not the ID counter.
These stateful components are single-threaded.

`IncrementalMapper` keeps an ID-to-pixel reference. Unique nonnegative IDs and
finite matching arrays are required; rejected invalid updates do not change state.
The median pixel displacement triggers a two-view attempt, with an optional wide
movement re-anchor on failure. This is a selection heuristic, not a
rotation-compensated baseline estimate. Too little shared identity re-anchors and
clears the old cloud. A successful pair promotes the reference and replaces the
cloud. Later updates mark a retained cloud stale, even when reconstruction has
not been attempted. Pair indices count accepted mapper updates, not timestamps.

`TwoViewReconstruction` validates pinhole calibration and thresholds at
construction. Invalid observations/failed model estimation return an empty,
unsuccessful result. Five-point essential RANSAC supplies a consensus. When eight
or more points are available, a normalized calibrated eight-point fit and
(s,s,0) SVD projection refine the minimal model from that consensus. Recovery
then checks cheirality; custom row-normalized DLT checks finite results, both-view
depth, ray angle and both-view reprojection error. Initial model consensus and
accepted point counts are reported separately. This is linear refinement, not
bundle adjustment; a small image residual alone does not establish accurate
structure.

## Coordinates and limitations

For each pair, `X2 = R*X1 + t`, `|t| = 1` and pixels satisfy
`x ~ K[R|t]X`. Output structure is expressed in the first camera's coordinate
frame and baseline units. A later pair has a different frame and unknown scale;
no pose composition or scale alignment occurs. The optional OpenGL viewer
centers/rescales a cloud for display only, and its webcam calibration is guessed.

`geometry.h` is standard-library-only: a symmetric Jacobi eigensolver solves
normal-equation DLT. Independent pinhole truth and OpenCV SVD tests check selected
well-conditioned cases. Extreme conditioning and general degeneracy detection
remain limitations.

`MemoryPool<T>` is a standalone single-threaded raw-slot utility. Its slab is
bounded by the byte budget including alignment. Allocate/deallocate do not call
the heap after construction, but object constructors/destructors may. Only
owned, allocated slots may be returned once; callers destroy live objects before
destroying/replacing the pool. It is not used to cap all pipeline/OpenCV memory.

## Offline evidence

`tests/synthetic_scene.h` provides independently projected non-planar ground truth,
a known mismatch mask and a separate 2D warped texture. `offline_demo` checks
ground-truth tolerances before exporting CSV/PLY/JSON. The geometric mapper test
uses known IDs; the frontend warp test is a separate observation experiment.
Neither represents an end-to-end real-camera evaluation. CTest covers three
geometric seeds, temporal bookkeeping, pair freshness, failure contracts and
memory boundaries. CI compiles optional viewers without requesting a camera or
display and separately runs headless sanitizer tests.
