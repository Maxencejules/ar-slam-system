#include "core/feature_tracker.h"
#include <iostream>

namespace ar_slam {

TrackingResult FeatureTracker::track_features(Frame::Ptr current_frame) {
    TrackingResult result;
    result.tracking_quality = 0.0f;
    
    if (!prev_frame_) {
        // First frame - just extract features
        current_frame->extract_features();
        prev_frame_ = current_frame;
        
        // Initialize tracking points
        prev_points_.clear();
        track_ids_.clear();
        
        for (const auto& feat : current_frame->get_features()) {
            prev_points_.push_back(feat.pixel);
            track_ids_.push_back(next_track_id_++);
        }
        
        result.num_tracked = prev_points_.size();
        result.tracking_quality = 1.0f;
        std::cout << "Initialized tracker with " << result.num_tracked << " features" << std::endl;
    } else {
        // Track using optical flow
        if (prev_points_.empty()) {
            std::cout << "No previous points to track" << std::endl;
            return result;
        }
        
        std::vector<cv::Point2f> curr_points;
        std::vector<uchar> status;
        std::vector<float> err;
        
        // Optical flow
        cv::calcOpticalFlowPyrLK(
            prev_frame_->get_image(), 
            current_frame->get_image(),
            prev_points_, 
            curr_points,
            status, 
            err,
            win_size_, 
            max_level_
        );
        
        // Collect valid tracks
        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i]) {
                result.prev_points.push_back(prev_points_[i]);
                result.curr_points.push_back(curr_points[i]);
                result.track_ids.push_back(track_ids_[i]);
                result.inliers.push_back(true);
            }
        }
        
        result.num_tracked = result.curr_points.size();
        result.num_inliers = result.num_tracked;
        result.tracking_quality = static_cast<float>(result.num_tracked) / prev_points_.size();
        
        std::cout << "Tracked " << result.num_tracked << "/" << prev_points_.size() 
                  << " features (quality: " << result.tracking_quality << ")" << std::endl;
        
        // Update for next frame
        prev_frame_ = current_frame;
        prev_points_ = result.curr_points;
    }
    
    return result;
}

void FeatureTracker::reset() {
    prev_frame_.reset();
    prev_points_.clear();
    track_ids_.clear();
    next_track_id_ = 0;
}

} // namespace ar_slam