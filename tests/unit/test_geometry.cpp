// Unit tests for the dependency-free multi-view geometry core.
// Deterministic and headless: builds synthetic two-view scenes, projects known
// 3D points, triangulates them back, and checks recovery to numerical precision.

#include <algorithm>
#include <cmath>
#include <vector>

#include "core/geometry.h"
#include "test_util.h"

using namespace ar_slam::geometry;

namespace {

constexpr double kPi = 3.14159265358979323846;

Mat3 rot_y(double deg) {
    double r = deg * kPi / 180.0;
    double c = std::cos(r);
    double s = std::sin(r);
    Mat3 R = Mat3::identity();
    R.m[0][0] = c;
    R.m[0][2] = s;
    R.m[2][0] = -s;
    R.m[2][2] = c;
    return R;
}

Mat3 default_intrinsics() {
    Mat3 K = Mat3::identity();
    K.m[0][0] = 525.0;
    K.m[1][1] = 525.0;
    K.m[0][2] = 320.0;
    K.m[1][2] = 240.0;
    return K;
}

const std::vector<Vec3>& scene() {
    static const std::vector<Vec3> pts = {{0.2, 0.1, 3.0},  {-0.5, 0.3, 4.5}, {0.8, -0.4, 5.0},
                                          {-0.1, -0.2, 3.7}, {0.0, 0.0, 6.0},  {1.2, 0.9, 4.2}};
    return pts;
}

void test_eigensolver() {
    // Symmetric matrix with a 2x2 coupled block {{2,1},{1,2}} (eigenvalues 1,3)
    // and two isolated eigenvalues 4 and 7.
    double A[4][4] = {{2, 1, 0, 0}, {1, 2, 0, 0}, {0, 0, 4, 0}, {0, 0, 0, 7}};
    Eigen4 e = symmetric_eig4(A);

    bool f1 = false, f3 = false, f4 = false, f7 = false;
    for (int i = 0; i < 4; ++i) {
        if (std::fabs(e.values[i] - 1.0) < 1e-6) f1 = true;
        if (std::fabs(e.values[i] - 3.0) < 1e-6) f3 = true;
        if (std::fabs(e.values[i] - 4.0) < 1e-6) f4 = true;
        if (std::fabs(e.values[i] - 7.0) < 1e-6) f7 = true;
    }
    CHECK(f1 && f3 && f4 && f7);

    // Eigenvectors must satisfy A v = lambda v.
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            double Av = 0.0;
            for (int k = 0; k < 4; ++k) Av += A[i][k] * e.vectors[k][j];
            CHECK_NEAR(Av, e.values[j] * e.vectors[i][j], 1e-9);
        }
    }
}

double reconstruct_max_error(const Mat3& R2, const Vec3& t2, double noise) {
    Mat3 K = default_intrinsics();
    Mat34 P1 = make_projection(K, Mat3::identity(), {0, 0, 0});
    Mat34 P2 = make_projection(K, R2, t2);

    // Deterministic LCG noise so the test is reproducible across machines.
    unsigned seed = 12345u;
    auto nz = [&]() {
        seed = seed * 1103515245u + 12345u;
        return (static_cast<int>((seed >> 16) & 0x7fff) / 32767.0 - 0.5);
    };

    double max_err = 0.0;
    for (const auto& X : scene()) {
        auto x1 = project(P1, X);
        auto x2 = project(P2, X);
        auto tri = triangulate(P1, P2, x1[0] + nz() * noise, x1[1] + nz() * noise,
                               x2[0] + nz() * noise, x2[1] + nz() * noise);
        CHECK(tri.valid);
        double e = std::sqrt((tri.point[0] - X[0]) * (tri.point[0] - X[0]) +
                             (tri.point[1] - X[1]) * (tri.point[1] - X[1]) +
                             (tri.point[2] - X[2]) * (tri.point[2] - X[2]));
        max_err = std::max(max_err, e);
    }
    return max_err;
}

void test_triangulation() {
    // Noise-free recovery should be at numerical precision.
    CHECK(reconstruct_max_error(Mat3::identity(), {-0.6, 0.0, 0.0}, 0.0) < 1e-6);
    CHECK(reconstruct_max_error(rot_y(8.0), {-0.6, 0.05, 0.1}, 0.0) < 1e-6);
    // With sub-pixel noise the error stays small.
    CHECK(reconstruct_max_error(rot_y(10.0), {-0.7, 0.0, 0.05}, 0.5) < 0.1);
}

void test_projection_roundtrip() {
    Mat3 K = default_intrinsics();
    Mat34 P = make_projection(K, Mat3::identity(), {0, 0, 0});
    bool behind = false;
    auto px = project(P, {0.0, 0.0, 5.0}, &behind);
    CHECK(!behind);
    CHECK_NEAR(px[0], 320.0, 1e-9);  // On-axis point lands at the principal point.
    CHECK_NEAR(px[1], 240.0, 1e-9);
}

}  // namespace

int main() {
    test_eigensolver();
    test_triangulation();
    test_projection_roundtrip();
    return artest::report("test_geometry");
}
