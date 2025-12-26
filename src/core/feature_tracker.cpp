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

        // Set result points for consistency
        result.curr_points = prev_points_;
        result.track_ids = track_ids_;

    } else {
        // Track using optical flow
        if (prev_points_.empty()) {
            std::cout << "No previous points to track, re-initializing..." << std::endl;
            prev_frame_.reset();
            return track_features(current_frame);  // Recursive call to re-initialize
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
        std::vector<cv::Point2f> good_prev_points;
        std::vector<cv::Point2f> good_curr_points;
        std::vector<int> good_track_ids;

        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i] && err[i] < 30.0f) {  // Add error threshold
                // Check if point is within image bounds
                const cv::Point2f& pt = curr_points[i];
                if (pt.x >= 0 && pt.x < current_frame->get_image().cols &&
                    pt.y >= 0 && pt.y < current_frame->get_image().rows) {
                    good_prev_points.push_back(prev_points_[i]);
                    good_curr_points.push_back(curr_points[i]);
                    if (i < track_ids_.size()) {
                        good_track_ids.push_back(track_ids_[i]);
                    }
                }
            }
        }

        // Apply RANSAC with Fundamental Matrix to remove outliers
        if (good_curr_points.size() >= 8) {
            std::vector<uchar> mask;
            cv::findFundamentalMat(good_prev_points, good_curr_points, cv::FM_RANSAC, 3.0, 0.99, mask);

            std::vector<cv::Point2f> ransac_prev_points;
            std::vector<cv::Point2f> ransac_curr_points;
            std::vector<int> ransac_track_ids;

            for (size_t i = 0; i < mask.size(); ++i) {
                if (mask[i]) {
                    ransac_prev_points.push_back(good_prev_points[i]);
                    ransac_curr_points.push_back(good_curr_points[i]);
                    ransac_track_ids.push_back(good_track_ids[i]);
                }
            }

            // Only update if we didn't lose too many points (sanity check)
            if (ransac_curr_points.size() > good_curr_points.size() * 0.5) {
                good_prev_points = ransac_prev_points;
                good_curr_points = ransac_curr_points;
                good_track_ids = ransac_track_ids;
            }
        }

        // Calculate tracking quality
        result.tracking_quality = prev_points_.empty() ? 0.0f :
                                 static_cast<float>(good_curr_points.size()) / prev_points_.size();

        std::cout << "Tracked " << good_curr_points.size() << "/" << prev_points_.size()
                  << " features (quality: " << result.tracking_quality << ")" << std::endl;

        // Check if we need to re-detect features
        const float MIN_QUALITY = 0.5f;
        const size_t MIN_FEATURES = 100;
        const size_t TARGET_FEATURES = 500;

        if (result.tracking_quality < MIN_QUALITY || good_curr_points.size() < MIN_FEATURES) {
            std::cout << "Tracking quality too low, re-detecting features..." << std::endl;

            // Re-extract features completely
            current_frame->extract_features();

            // Reset tracking
            prev_points_.clear();
            track_ids_.clear();

            for (const auto& feat : current_frame->get_features()) {
                prev_points_.push_back(feat.pixel);
                track_ids_.push_back(next_track_id_++);
            }

            prev_frame_ = current_frame;

            // Set result
            result.prev_points.clear();
            result.curr_points = prev_points_;
            result.track_ids = track_ids_;
            result.num_tracked = prev_points_.size();
            result.num_inliers = result.num_tracked;
            result.tracking_quality = 1.0f;

            for (int i = 0; i < result.num_tracked; ++i) {
                result.inliers.push_back(true);
            }

            std::cout << "Re-initialized with " << result.num_tracked << " features" << std::endl;

        } else {
            // Normal tracking result
            result.prev_points = good_prev_points;
            result.curr_points = good_curr_points;
            result.track_ids = good_track_ids;
            result.num_tracked = good_curr_points.size();
            result.num_inliers = result.num_tracked;

            for (size_t i = 0; i < result.num_tracked; ++i) {
                result.inliers.push_back(true);
            }

            // Check if we need to add more features
            if (good_curr_points.size() < TARGET_FEATURES) {
                // Create mask to avoid detecting features near existing ones
                cv::Mat mask = cv::Mat::ones(current_frame->get_image().rows,
                                            current_frame->get_image().cols, CV_8UC1) * 255;

                // Mask out areas around existing features
                for (const auto& pt : good_curr_points) {
                    cv::circle(mask, cv::Point(pt.x, pt.y), 20, cv::Scalar(0), -1);
                }

                // Detect additional features
                std::vector<cv::KeyPoint> new_keypoints;
                cv::Ptr<cv::ORB> detector = cv::ORB::create(TARGET_FEATURES - good_curr_points.size());
                detector->detect(current_frame->get_image(), new_keypoints, mask);

                // Add new features to tracking
                int added = 0;
                for (const auto& kp : new_keypoints) {
                    good_curr_points.push_back(kp.pt);
                    good_track_ids.push_back(next_track_id_++);
                    added++;

                    if (good_curr_points.size() >= TARGET_FEATURES) break;
                }

                if (added > 0) {
                    std::cout << "Added " << added << " new features (total: "
                              << good_curr_points.size() << ")" << std::endl;
                }

                // Update result with new features
                result.curr_points = good_curr_points;
                result.track_ids = good_track_ids;
                result.num_tracked = good_curr_points.size();
            }

            // Update for next frame
            prev_frame_ = current_frame;
            prev_points_ = result.curr_points;
            track_ids_ = result.track_ids;
        }
    }

    return result;
}

void FeatureTracker::reset() {
    prev_frame_.reset();
    prev_points_.clear();
    track_ids_.clear();
    next_track_id_ = 0;
    std::cout << "Tracker reset" << std::endl;
}

} // namespace ar_slam