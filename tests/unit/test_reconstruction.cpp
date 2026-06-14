// Integration test for the two-view reconstruction back-end.
// Builds a synthetic calibrated scene, projects it into two cameras with a
// known relative pose, and verifies that TwoViewReconstruction recovers both
// the motion (up to scale) and the 3D structure (up to the baseline scale).
// Headless and deterministic — no camera or image files required.

#include <cmath>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include "core/reconstruction.h"
#include "test_util.h"

namespace {

    cv::Point2f project(const cv::Matx33d& K,
                        const cv::Matx33d& R,
                        const cv::Vec3d& t,
                        const cv::Point3f& X) {
        cv::Vec3d Xc = R * cv::Vec3d(X.x, X.y, X.z) + t;
        cv::Vec3d px = K * Xc;
        return cv::Point2f(static_cast<float>(px[0] / px[2]), static_cast<float>(px[1] / px[2]));
    }

    cv::Matx33d rot_y(double deg) {
        double r = deg * CV_PI / 180.0;
        double c = std::cos(r), s = std::sin(r);
        return cv::Matx33d(c, 0, s, 0, 1, 0, -s, 0, c);
    }

}  // namespace

int main() {
    cv::Matx33d K(525, 0, 320, 0, 525, 240, 0, 0, 1);

    // A grid of points spread in depth, all in front of both cameras.
    std::vector<cv::Point3f> world;
    for (int i = -4; i <= 4; ++i) {
        for (int j = -3; j <= 3; ++j) {
            world.emplace_back(i * 0.15f, j * 0.15f, 4.0f + 0.1f * (i + j));
        }
    }

    // Ground-truth relative pose: sideways translation plus a small rotation.
    cv::Matx33d R_gt = rot_y(6.0);
    cv::Vec3d t_gt(-0.5, 0.02, 0.05);

    std::vector<cv::Point2f> pts1, pts2;
    cv::Matx33d I = cv::Matx33d::eye();
    cv::Vec3d zero(0, 0, 0);
    for (const auto& X : world) {
        pts1.push_back(project(K, I, zero, X));
        pts2.push_back(project(K, R_gt, t_gt, X));
    }

    ar_slam::TwoViewReconstruction recon(K);
    ar_slam::ReconstructionResult result = recon.reconstruct(pts1, pts2);

    CHECK(result.success);
    CHECK(result.num_inliers >= 30);

    // Translation direction should match ground truth (both unit, sign resolved
    // by cheirality).
    cv::Vec3d t_gt_unit = cv::normalize(t_gt);
    cv::Vec3d t_rec = cv::normalize(result.t);
    double cos_angle = t_gt_unit.dot(t_rec);
    CHECK(cos_angle > 0.99);  // within ~8 degrees

    // Rotation should match ground truth: R_gt * R_rec^T ~= I (trace ~= 3).
    cv::Matx33d dR = R_gt * result.R.t();
    double trace = dR(0, 0) + dR(1, 1) + dR(2, 2);
    CHECK_NEAR(trace, 3.0, 0.02);

    // Structure is recovered up to the baseline scale: rescaling by the true
    // baseline length should reproduce the world points.
    double baseline = cv::norm(t_gt);
    double max_err = 0.0;
    for (size_t k = 0; k < result.points.size(); ++k) {
        const cv::Point3f& p = result.points[k];
        const cv::Point3f& X = world[result.point_indices[k]];
        cv::Point3f scaled(p.x * static_cast<float>(baseline), p.y * static_cast<float>(baseline),
                           p.z * static_cast<float>(baseline));
        double dx = scaled.x - X.x, dy = scaled.y - X.y, dz = scaled.z - X.z;
        double e = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (e > max_err)
            max_err = e;
    }
    CHECK(max_err < 0.05);  // < 5cm on a scene a few metres deep

    // A degenerate input (too few correspondences) must fail cleanly, not crash.
    std::vector<cv::Point2f> few1(pts1.begin(), pts1.begin() + 5);
    std::vector<cv::Point2f> few2(pts2.begin(), pts2.begin() + 5);
    ar_slam::ReconstructionResult degenerate = recon.reconstruct(few1, few2);
    CHECK(!degenerate.success);

    return artest::report("test_reconstruction");
}
