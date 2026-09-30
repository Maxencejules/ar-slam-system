#include "core/incremental_mapper.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace ar_slam {
    namespace {
        TwoViewReconstruction::Config reconstruction_config(
            const IncrementalMapper::Config& config) {
            TwoViewReconstruction::Config result;
            result.min_correspondences = config.min_correspondences;
            result.min_inliers = std::min(result.min_inliers, config.min_correspondences);
            return result;
        }
    }  // namespace
    IncrementalMapper::IncrementalMapper(const cv::Matx33d& K) : IncrementalMapper(K, Config{}) {}
    IncrementalMapper::IncrementalMapper(const cv::Matx33d& K, const Config& config)
        : config_(config), reconstructor_(K, reconstruction_config(config)) {
        if (!std::isfinite(config.min_parallax_px) || config.min_parallax_px <= 0 ||
            config.min_correspondences < 5 || config.min_shared_to_keep < 1 ||
            config.min_shared_to_keep > config.min_correspondences ||
            !std::isfinite(config.force_keyframe_px) ||
            config.force_keyframe_px < config.min_parallax_px) {
            throw std::invalid_argument("Invalid mapper thresholds");
        }
    }
    void IncrementalMapper::set_reference(const std::vector<int>& ids,
                                          const std::vector<cv::Point2f>& points) {
        reference_.clear();
        for (std::size_t i = 0; i < ids.size(); ++i)
            reference_.emplace(ids[i], points[i]);
        reference_index_ = update_index_;
    }
    void IncrementalMapper::clear_cloud() {
        cloud_.clear();
        has_cloud_ = false;
        cloud_stale_ = true;
        cloud_reference_index_ = cloud_current_index_ = 0;
        last_result_ = {};
    }
    bool IncrementalMapper::update(const std::vector<int>& ids,
                                   const std::vector<cv::Point2f>& points) {
        if (ids.size() != points.size())
            throw std::invalid_argument("Observation arrays differ");
        std::unordered_set<int> unique;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] < 0 || !unique.insert(ids[i]).second || !std::isfinite(points[i].x) ||
                !std::isfinite(points[i].y)) {
                throw std::invalid_argument("IDs must be unique and pixels finite");
            }
        }
        ++update_index_;
        last_parallax_ = 0;
        cloud_stale_ = true;
        if (reference_.empty()) {
            set_reference(ids, points);
            clear_cloud();
            return false;
        }
        std::vector<cv::Point2f> previous, current;
        std::vector<double> displacement;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            const auto found = reference_.find(ids[i]);
            if (found == reference_.end())
                continue;
            previous.push_back(found->second);
            current.push_back(points[i]);
            displacement.push_back(std::hypot(static_cast<double>(points[i].x) - found->second.x,
                                              static_cast<double>(points[i].y) - found->second.y));
        }
        if (previous.size() < static_cast<std::size_t>(config_.min_shared_to_keep)) {
            set_reference(ids, points);
            clear_cloud();  // Do not display an old cloud after tracking loss.
            return false;
        }
        std::sort(displacement.begin(), displacement.end());
        const auto middle = displacement.size() / 2;
        last_parallax_ = displacement.size() % 2
                             ? displacement[middle]
                             : (displacement[middle - 1] + displacement[middle]) * 0.5;
        if (previous.size() < static_cast<std::size_t>(config_.min_correspondences) ||
            last_parallax_ < config_.min_parallax_px)
            return false;
        last_result_ = reconstructor_.reconstruct(previous, current);
        if (last_result_.success) {
            cloud_ = last_result_.points;
            has_cloud_ = true;
            cloud_stale_ = false;
            cloud_reference_index_ = reference_index_;
            cloud_current_index_ = update_index_;
            set_reference(ids, points);
            return true;
        }
        if (last_parallax_ > config_.force_keyframe_px)
            set_reference(ids, points);
        return false;
    }
    void IncrementalMapper::reset() {
        reference_.clear();
        last_parallax_ = 0;
        update_index_ = reference_index_ = 0;
        clear_cloud();
    }
}  // namespace ar_slam
