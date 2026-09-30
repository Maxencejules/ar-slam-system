#pragma once
#include <opencv2/core.hpp>
#include <vector>

namespace ar_slam {
    // X2 = R*X1 + t. Translation has unit norm; points are in camera-1
    // coordinates in baseline units, not metres. Only use a successful result.
    struct ReconstructionResult {
        bool success = false;
        cv::Matx33d R = cv::Matx33d::eye();
        cv::Vec3d t{0, 0, 0};
        std::vector<cv::Point3f> points;
        std::vector<int> point_indices;
        int num_model_inliers = 0;  // Initial RANSAC consensus, before quality gates.
        int num_inliers = 0;        // Accepted after all geometric quality gates.
        double inlier_ratio = 0.0;
        double median_reprojection_error_px = 0.0;
        double median_triangulation_angle_deg = 0.0;
    };

    class TwoViewReconstruction {
    public:
        struct Config {
            double ransac_prob = 0.999;
            double ransac_threshold = 1.0;  // Pixel epipolar threshold.
            int min_correspondences = 30;
            int min_inliers = 15;
            double min_inlier_ratio = 0.5;
            double max_depth = 100.0;  // Positive z limit in BOTH cameras, baseline units.
            double max_reprojection_error_px = 2.0;
            double min_triangulation_angle_deg = 1.0;
        };
        explicit TwoViewReconstruction(const cv::Matx33d& K);
        TwoViewReconstruction(const cv::Matx33d& K, const Config& config);
        // Inputs are matched, already-undistorted pinhole pixels. Invalid
        // correspondence arrays or failed geometry return success=false.
        // Invalid calibration/configuration throws invalid_argument at construction.
        ReconstructionResult reconstruct(const std::vector<cv::Point2f>& pts1,
                                         const std::vector<cv::Point2f>& pts2) const;
        const cv::Matx33d& intrinsics() const { return K_; }

    private:
        cv::Matx33d K_;
        Config config_;
    };
}  // namespace ar_slam
