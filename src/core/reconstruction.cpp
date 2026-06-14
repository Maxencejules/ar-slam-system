#include "core/reconstruction.h"

#include <opencv2/calib3d.hpp>

#include "core/geometry.h"

namespace ar_slam {

namespace {

geometry::Mat3 to_geom_mat3(const cv::Matx33d& m) {
    geometry::Mat3 out;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            out.m[i][j] = m(i, j);
        }
    }
    return out;
}

}  // namespace

TwoViewReconstruction::TwoViewReconstruction(const cv::Matx33d& K)
    : TwoViewReconstruction(K, Config{}) {}

TwoViewReconstruction::TwoViewReconstruction(const cv::Matx33d& K, const Config& config)
    : K_(K), config_(config) {}

ReconstructionResult TwoViewReconstruction::reconstruct(
    const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2) const {
    ReconstructionResult result;

    if (pts1.size() != pts2.size() ||
        static_cast<int>(pts1.size()) < config_.min_correspondences) {
        return result;
    }

    const cv::Mat K = cv::Mat(K_);

    // 1. Robustly estimate the essential matrix.
    cv::Mat inlier_mask;
    cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, config_.ransac_prob,
                                     config_.ransac_threshold, inlier_mask);
    if (E.rows != 3 || E.cols != 3) {
        return result;  // Degenerate configuration (e.g. pure rotation, no parallax).
    }

    // 2. Decompose into relative pose using the cheirality constraint.
    cv::Mat R, t;
    int pose_inliers = cv::recoverPose(E, pts1, pts2, K, R, t, inlier_mask);
    if (pose_inliers <= 0) {
        return result;
    }

    cv::Matx33d Rx;
    cv::Vec3d tx;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Rx(i, j) = R.at<double>(i, j);
        }
        tx(i) = t.at<double>(i, 0);
    }

    // 3. Build projection matrices and triangulate the surviving inliers.
    const geometry::Mat3 Kg = to_geom_mat3(K_);
    const geometry::Mat3 Rg = to_geom_mat3(Rx);
    const geometry::Vec3 tg{tx(0), tx(1), tx(2)};
    const geometry::Mat34 P1 =
        geometry::make_projection(Kg, geometry::Mat3::identity(), {0, 0, 0});
    const geometry::Mat34 P2 = geometry::make_projection(Kg, Rg, tg);

    result.points.reserve(pose_inliers);
    result.point_indices.reserve(pose_inliers);

    for (size_t i = 0; i < pts1.size(); ++i) {
        if (!inlier_mask.empty() && inlier_mask.at<uchar>(static_cast<int>(i)) == 0) {
            continue;
        }

        auto tri = geometry::triangulate(P1, P2, pts1[i].x, pts1[i].y, pts2[i].x, pts2[i].y);
        if (!tri.valid) {
            continue;
        }

        // Cheirality: the point must lie in front of both cameras.
        const double z1 = tri.point[2];
        const double z2 = Rx(2, 0) * tri.point[0] + Rx(2, 1) * tri.point[1] +
                          Rx(2, 2) * tri.point[2] + tx(2);
        if (z1 <= 0.0 || z2 <= 0.0 || z1 > config_.max_depth) {
            continue;
        }

        result.points.emplace_back(static_cast<float>(tri.point[0]),
                                   static_cast<float>(tri.point[1]),
                                   static_cast<float>(tri.point[2]));
        result.point_indices.push_back(static_cast<int>(i));
    }

    result.R = Rx;
    result.t = tx;
    result.num_inliers = static_cast<int>(result.points.size());
    result.inlier_ratio =
        pts1.empty() ? 0.0 : static_cast<double>(result.num_inliers) / pts1.size();
    result.success = result.num_inliers >= config_.min_correspondences / 2;
    return result;
}

}  // namespace ar_slam
