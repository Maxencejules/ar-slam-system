#include <utility>
#include "core/feature_tracker.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ar_slam {
    int FeatureTracker::next_id() {
        if (next_track_id_ > std::numeric_limits<int>::max()) {
            throw std::overflow_error("Track identifier space exhausted");
        }
        return static_cast<int>(next_track_id_++);
    }

    TrackingResult FeatureTracker::track_features(Frame::Ptr current_frame) {
        if (!current_frame)
            throw std::invalid_argument("Cannot track a null frame");
        if (prev_frame_) {
            if (current_frame->get_timestamp() <= prev_frame_->get_timestamp()) {
                throw std::invalid_argument("Frame timestamps must increase strictly");
            }
            if (current_frame->get_image().size() != prev_frame_->get_image().size()) {
                throw std::invalid_argument(
                    "Reset tracker and calibration after resolution changes");
            }
        }
        TrackingResult result;
        if (!prev_frame_ || prev_points_.empty()) {
            current_frame->extract_features(500);
            for (const auto& feature : current_frame->get_features()) {
                result.prev_points.push_back(feature.pixel);
                result.curr_points.push_back(feature.pixel);
                result.track_ids.push_back(next_id());
                result.inliers.push_back(false);
            }
            result.reinitialized = true;
        } else {
            std::vector<cv::Point2f> current, backward;
            std::vector<uchar> status, backward_status;
            std::vector<float> error, backward_error;
            cv::calcOpticalFlowPyrLK(prev_frame_->get_image(), current_frame->get_image(),
                                     prev_points_, current, status, error, win_size_, max_level_);
            cv::calcOpticalFlowPyrLK(current_frame->get_image(), prev_frame_->get_image(), current,
                                     backward, backward_status, backward_error, win_size_,
                                     max_level_);
            for (std::size_t i = 0; i < status.size(); ++i) {
                const auto& point = current[i];
                const double roundtrip =
                    std::hypot(static_cast<double>(backward[i].x) - prev_points_[i].x,
                               static_cast<double>(backward[i].y) - prev_points_[i].y);
                if (status[i] && backward_status[i] && error[i] < 30.0f && std::isfinite(point.x) &&
                    std::isfinite(point.y) && roundtrip <= 1.0 && point.x >= 0 && point.y >= 0 &&
                    point.x < current_frame->get_image().cols &&
                    point.y < current_frame->get_image().rows) {
                    result.prev_points.push_back(prev_points_[i]);
                    result.curr_points.push_back(point);
                    result.track_ids.push_back(track_ids_[i]);
                    result.inliers.push_back(true);
                }
            }
            if (result.curr_points.size() >= 8) {
                std::vector<uchar> mask;
                const auto fundamental = cv::findFundamentalMat(
                    result.prev_points, result.curr_points, cv::FM_RANSAC, 3.0, 0.99, mask);
                if (!fundamental.empty() && mask.size() == result.curr_points.size()) {
                    TrackingResult filtered;
                    for (std::size_t i = 0; i < mask.size(); ++i) {
                        if (!mask[i])
                            continue;
                        filtered.prev_points.push_back(result.prev_points[i]);
                        filtered.curr_points.push_back(result.curr_points[i]);
                        filtered.track_ids.push_back(result.track_ids[i]);
                        filtered.inliers.push_back(true);
                    }
                    result = std::move(filtered);
                }
            }
            result.num_inliers = static_cast<int>(result.curr_points.size());
            result.tracking_quality =
                static_cast<float>(result.num_inliers) / static_cast<float>(prev_points_.size());
            result.reinitialized = result.num_inliers == 0;

            constexpr std::size_t target = 500;
            if (result.curr_points.size() < target) {
                cv::Mat mask(current_frame->get_image().size(), CV_8UC1, cv::Scalar(255));
                for (const auto& point : result.curr_points) {
                    cv::circle(mask, cv::Point(cvRound(point.x), cvRound(point.y)), 20,
                               cv::Scalar(0), -1);
                }
                std::vector<cv::KeyPoint> keypoints;
                auto detector =
                    cv::ORB::create(static_cast<int>(target - result.curr_points.size()));
                detector->detect(current_frame->get_image(), keypoints, mask);
                for (const auto& keypoint : keypoints) {
                    if (result.curr_points.size() >= target)
                        break;
                    result.prev_points.push_back(keypoint.pt);
                    result.curr_points.push_back(keypoint.pt);
                    result.track_ids.push_back(next_id());
                    result.inliers.push_back(false);
                }
            }
        }
        result.num_tracked = static_cast<int>(result.curr_points.size());
        result.num_new = result.num_tracked - result.num_inliers;
        prev_frame_ = std::move(current_frame);
        prev_points_ = result.curr_points;
        track_ids_ = result.track_ids;
        return result;
    }

    void FeatureTracker::reset() {
        prev_frame_.reset();
        prev_points_.clear();
        track_ids_.clear();
    }
}  // namespace ar_slam
