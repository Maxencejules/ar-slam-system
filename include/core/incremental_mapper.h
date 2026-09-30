#pragma once
#include <cstddef>
#include <opencv2/core.hpp>
#include <unordered_map>
#include <vector>
#include "core/reconstruction.h"

namespace ar_slam {
    // Selects camera pairs from ordered observations and replaces a pair-local
    // cloud. It does not compose poses, align scales, or accumulate a global map.
    class IncrementalMapper {
    public:
        struct Config {
            double min_parallax_px = 20.0;  // Median pixel displacement; rotation also causes this.
            int min_correspondences = 40;
            int min_shared_to_keep = 12;
            double force_keyframe_px = 80.0;
        };
        explicit IncrementalMapper(const cv::Matx33d& K);
        IncrementalMapper(const cv::Matx33d& K, const Config& config);
        // IDs are unique nonnegative observations from ONE tracker instance.
        // Sizes, duplicates and nonfinite pixels throw invalid_argument without
        // changing mapper state. Reset if replacing the tracker or calibration.
        bool update(const std::vector<int>& track_ids, const std::vector<cv::Point2f>& points);
        bool has_cloud() const { return has_cloud_; }
        bool cloud_is_stale() const { return has_cloud_ && cloud_stale_; }
        const std::vector<cv::Point3f>& cloud() const { return cloud_; }
        // Accepted update indices, NOT timestamps or a world coordinate frame.
        std::size_t cloud_reference_index() const { return cloud_reference_index_; }
        std::size_t cloud_current_index() const { return cloud_current_index_; }
        std::size_t reference_index() const { return reference_index_; }
        double last_parallax() const { return last_parallax_; }
        const ReconstructionResult& last_result() const { return last_result_; }
        void reset();

    private:
        Config config_;
        TwoViewReconstruction reconstructor_;
        std::unordered_map<int, cv::Point2f> reference_;
        std::vector<cv::Point3f> cloud_;
        bool has_cloud_ = false;
        bool cloud_stale_ = true;
        double last_parallax_ = 0.0;
        ReconstructionResult last_result_;
        std::size_t update_index_ = 0;
        std::size_t reference_index_ = 0;
        std::size_t cloud_reference_index_ = 0;
        std::size_t cloud_current_index_ = 0;
        void set_reference(const std::vector<int>& ids, const std::vector<cv::Point2f>& points);
        void clear_cloud();
    };
}  // namespace ar_slam
