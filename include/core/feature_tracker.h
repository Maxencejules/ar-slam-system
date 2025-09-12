#pragma once
#include "core/frame.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace ar_slam {

    struct TrackingResult {
        std::vector<cv::Point2f> prev_points;
        std::vector<cv::Point2f> curr_points;
        std::vector<int> track_ids;
        std::vector<bool> inliers;
        int num_tracked = 0;
        int num_inliers = 0;
        float tracking_quality = 0.0f;
    };

    class FeatureTracker {
    private:
        Frame::Ptr prev_frame_;
        std::vector<cv::Point2f> prev_points_;
        std::vector<int> track_ids_;
        int next_track_id_ = 0;

        // Optical flow parameters
        cv::Size win_size_{21, 21};
        int max_level_{3};

    public:
        FeatureTracker() = default;

        // Main tracking function
        TrackingResult track_features(Frame::Ptr current_frame);

        // Reset tracker
        void reset();
    };

} // namespace ar_slam