#pragma once

#include <array>
#include <cmath>
#include <cstddef>

/**
 * @file geometry.h
 * @brief Dependency-free multi-view geometry primitives.
 *
 * This header implements the structure-from-motion math used by the
 * reconstruction front-end without pulling in OpenCV or Eigen, which keeps the
 * core algorithms unit-testable in isolation. It provides:
 *   - small fixed-size matrix/vector types (Mat3, Mat34, Vec3);
 *   - a Jacobi eigen-decomposition for 4x4 symmetric matrices;
 *   - linear (DLT) triangulation of a 3D point from two calibrated views.
 *
 * Conventions follow Hartley & Zisserman, "Multiple View Geometry": a camera
 * projects a homogeneous world point X to an image point via x ~ P X, with
 * P = K [R | t] a 3x4 projection matrix.
 */
namespace ar_slam::geometry {

using Vec3 = std::array<double, 3>;

/// Row-major 3x3 matrix.
struct Mat3 {
    double m[3][3] = {{0}};

    static Mat3 identity() {
        Mat3 r;
        r.m[0][0] = r.m[1][1] = r.m[2][2] = 1.0;
        return r;
    }
};

/// Row-major 3x4 projection matrix.
struct Mat34 {
    double m[3][4] = {{0}};
};

/// Build a projection matrix P = K [R | t].
inline Mat34 make_projection(const Mat3& K, const Mat3& R, const Vec3& t) {
    // Rt = [R | t] (3x4)
    double Rt[3][4];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Rt[i][j] = R.m[i][j];
        }
        Rt[i][3] = t[i];
    }
    Mat34 P;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) {
                sum += K.m[i][k] * Rt[k][j];
            }
            P.m[i][j] = sum;
        }
    }
    return P;
}

/// Project a 3D world point with a 3x4 projection matrix, returning pixel (u,v).
/// `behind` is set true when the point lies on/behind the principal plane.
inline std::array<double, 2> project(const Mat34& P, const Vec3& X, bool* behind = nullptr) {
    double x = P.m[0][0] * X[0] + P.m[0][1] * X[1] + P.m[0][2] * X[2] + P.m[0][3];
    double y = P.m[1][0] * X[0] + P.m[1][1] * X[1] + P.m[1][2] * X[2] + P.m[1][3];
    double w = P.m[2][0] * X[0] + P.m[2][1] * X[1] + P.m[2][2] * X[2] + P.m[2][3];
    if (behind) {
        *behind = (w <= 0.0);
    }
    if (std::fabs(w) < 1e-12) {
        w = (w < 0.0) ? -1e-12 : 1e-12;
    }
    return {x / w, y / w};
}

/// Eigen-decomposition result for a 4x4 symmetric matrix.
struct Eigen4 {
    double values[4];       ///< Eigenvalues (not sorted).
    double vectors[4][4];   ///< Column j is the eigenvector for values[j].
};

/**
 * @brief Jacobi eigen-decomposition of a 4x4 real symmetric matrix.
 *
 * Uses cyclic two-sided Givens rotations. Convergence is quadratic for
 * symmetric input; the iteration cap is generous enough that well-formed
 * inputs always converge before it is hit.
 */
inline Eigen4 symmetric_eig4(const double Ain[4][4]) {
    constexpr int n = 4;
    double a[n][n];
    double v[n][n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            a[i][j] = Ain[i][j];
            v[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                off += a[p][q] * a[p][q];
            }
        }
        if (off < 1e-30) {
            break;
        }

        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                if (std::fabs(a[p][q]) < 1e-300) {
                    continue;
                }
                // Rotation angle that annihilates a[p][q].
                // Zeroing (J^T A J)[p][q] requires tan(2*phi) = 2*a_pq / (a_qq - a_pp).
                double phi = 0.5 * std::atan2(2.0 * a[p][q], a[q][q] - a[p][p]);
                double c = std::cos(phi);
                double s = std::sin(phi);

                // A <- J^T A J, applied as columns then rows.
                for (int k = 0; k < n; ++k) {
                    double akp = a[k][p];
                    double akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {
                    double apk = a[p][k];
                    double aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                // Accumulate eigenvectors V <- V J.
                for (int k = 0; k < n; ++k) {
                    double vkp = v[k][p];
                    double vkq = v[k][q];
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }

    Eigen4 out;
    for (int i = 0; i < n; ++i) {
        out.values[i] = a[i][i];
        for (int j = 0; j < n; ++j) {
            out.vectors[i][j] = v[i][j];
        }
    }
    return out;
}

/// Result of triangulating a single correspondence.
struct TriangulationResult {
    Vec3 point{};       ///< 3D point in the world frame of P1.
    bool valid = false; ///< False if the homogeneous point is at/near infinity.
};

/**
 * @brief Linear (DLT) triangulation of one point from two calibrated views.
 *
 * Builds the 4x4 system A X = 0 from the two projection matrices and the two
 * image observations (Hartley & Zisserman eq. 12.1), row-normalises it for
 * conditioning, and solves for the null space as the eigenvector of the
 * smallest eigenvalue of A^T A.
 *
 * @param P1,P2  3x4 projection matrices of the two views.
 * @param u1,v1  Pixel observation in view 1.
 * @param u2,v2  Pixel observation in view 2.
 */
inline TriangulationResult triangulate(const Mat34& P1, const Mat34& P2,
                                       double u1, double v1, double u2, double v2) {
    double A[4][4];
    for (int j = 0; j < 4; ++j) {
        A[0][j] = u1 * P1.m[2][j] - P1.m[0][j];
        A[1][j] = v1 * P1.m[2][j] - P1.m[1][j];
        A[2][j] = u2 * P2.m[2][j] - P2.m[0][j];
        A[3][j] = v2 * P2.m[2][j] - P2.m[1][j];
    }

    // Row-normalise for numerical conditioning.
    for (int i = 0; i < 4; ++i) {
        double norm = 0.0;
        for (int j = 0; j < 4; ++j) {
            norm += A[i][j] * A[i][j];
        }
        norm = std::sqrt(norm);
        if (norm > 1e-12) {
            for (int j = 0; j < 4; ++j) {
                A[i][j] /= norm;
            }
        }
    }

    // M = A^T A (4x4 symmetric PSD); null space is its smallest eigenvector.
    double M[4][4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 4; ++k) {
                sum += A[k][i] * A[k][j];
            }
            M[i][j] = sum;
        }
    }

    Eigen4 eig = symmetric_eig4(M);
    int smallest = 0;
    for (int i = 1; i < 4; ++i) {
        if (eig.values[i] < eig.values[smallest]) {
            smallest = i;
        }
    }

    double X[4];
    for (int i = 0; i < 4; ++i) {
        X[i] = eig.vectors[i][smallest];
    }

    TriangulationResult result;
    if (std::fabs(X[3]) < 1e-9) {
        result.valid = false;  // Point at infinity.
        return result;
    }
    result.point = {X[0] / X[3], X[1] / X[3], X[2] / X[3]};
    result.valid = true;
    return result;
}

}  // namespace ar_slam::geometry
