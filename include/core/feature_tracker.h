#pragma once
#include "core/frame.h"
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <vector>

namespace ar_slam {
    struct TrackingResult {
        // All four observation arrays have num_tracked entries. New features
        // have prev_points[i] == curr_points[i] as a placeholder and inliers[i]
        // == false: they have no previous-frame correspondence.
        std::vector<cv::Point2f> prev_points;
        std::vector<cv::Point2f> curr_points;
        std::vector<int> track_ids;
        std::vector<bool> inliers;
        int num_tracked = 0;  // Active observations, including newly detected ones.
        int num_inliers = 0;  // Surviving temporal correspondences, excluding new ones.
        int num_new = 0;
        float tracking_quality = 0.0f;  // Retained previous observations / previous count.
        bool reinitialized = false;
    };

    // Single-threaded frontend. Frame timestamps must increase strictly and
    // resolution must remain fixed between reset() calls.
    class FeatureTracker {
    private:
        Frame::Ptr prev_frame_;
        std::vector<cv::Point2f> prev_points_;
        std::vector<int> track_ids_;
        std::int64_t next_track_id_ = 0;
        cv::Size win_size_{21, 21};
        int max_level_{3};
        int next_id();

    public:
        FeatureTracker() = default;
        TrackingResult track_features(Frame::Ptr current_frame);
        // Clears frame state; IDs are never reused within this tracker instance.
        void reset();
    };
}  // namespace ar_slam
