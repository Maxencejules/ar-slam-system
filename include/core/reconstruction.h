#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace ar_slam {

    /**
     * @brief Output of a two-view reconstruction attempt.
     *
     * Poses follow the standard convention: view 1 is the world origin
     * (R = I, t = 0) and a world point X maps into view 2 by X2 = R * X + t.
     * Because monocular reconstruction is only defined up to a global scale,
     * @ref t is a unit vector and @ref points are expressed in that same
     * (unknown) metric scale.
     */
    struct ReconstructionResult {
        bool success = false;                ///< True if a valid pose was recovered.
        cv::Matx33d R = cv::Matx33d::eye();  ///< Rotation of view 2 w.r.t. view 1.
        cv::Vec3d t{0, 0, 0};                ///< Unit translation (up to scale).
        std::vector<cv::Point3f> points;     ///< Triangulated inlier points (view-1 frame).
        std::vector<int> point_indices;      ///< Index of each point in the input arrays.
        int num_inliers = 0;                 ///< Correspondences passing the cheirality check.
        double inlier_ratio = 0.0;           ///< num_inliers / input correspondences.
    };

    /**
     * @brief Recovers relative camera motion and sparse 3D structure from two views.
     *
     * Given pixel correspondences between two frames of a calibrated camera, this
     * class estimates the essential matrix with RANSAC, decomposes it into a
     * relative pose using the cheirality (positive-depth) constraint, and
     * triangulates the surviving inliers. Triangulation uses the dependency-free
     * DLT solver in geometry.h, which is unit-tested in isolation.
     *
     * This is the geometric back-end that turns the tracking front-end's 2D
     * correspondences into real 3D structure.
     */
    class TwoViewReconstruction {
    public:
        /// Tunable thresholds for the reconstruction.
        struct Config {
            double ransac_prob = 0.999;     ///< RANSAC confidence for findEssentialMat.
            double ransac_threshold = 1.0;  ///< Max epipolar error in pixels for inliers.
            int min_correspondences = 30;   ///< Minimum matches required to attempt.
            double max_depth = 100.0;       ///< Reject points farther than this (scale units).
        };

        /// Construct with default thresholds.
        explicit TwoViewReconstruction(const cv::Matx33d& K);

        /// Construct with explicit thresholds.
        TwoViewReconstruction(const cv::Matx33d& K, const Config& config);

        /**
         * @brief Reconstruct structure and motion from matched correspondences.
         * @param pts1 Pixel observations in view 1.
         * @param pts2 Pixel observations in view 2 (pts2[i] matches pts1[i]).
         * @return A ReconstructionResult; check .success before using its fields.
         */
        ReconstructionResult reconstruct(const std::vector<cv::Point2f>& pts1,
                                         const std::vector<cv::Point2f>& pts2) const;

        const cv::Matx33d& intrinsics() const { return K_; }

    private:
        cv::Matx33d K_;
        Config config_;
    };

}  // namespace ar_slam
