#include "core/incremental_mapper.h"

#include <algorithm>
#include <cmath>

namespace ar_slam {

    IncrementalMapper::IncrementalMapper(const cv::Matx33d& K) : IncrementalMapper(K, Config{}) {}

    IncrementalMapper::IncrementalMapper(const cv::Matx33d& K, const Config& config)
        : K_(K), config_(config), reconstructor_(K) {}

    void IncrementalMapper::set_reference(const std::vector<int>& ids,
                                          const std::vector<cv::Point2f>& pts) {
        reference_.clear();
        reference_.reserve(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            reference_[ids[i]] = pts[i];
        }
        has_reference_ = !reference_.empty();
    }

    bool IncrementalMapper::update(const std::vector<int>& track_ids,
                                   const std::vector<cv::Point2f>& points) {
        last_parallax_ = 0.0;
        if (track_ids.size() != points.size()) {
            return false;
        }

        if (!has_reference_) {
            set_reference(track_ids, points);
            return false;
        }

        // Match current observations to the reference keyframe by track id.
        std::vector<cv::Point2f> ref_pts;
        std::vector<cv::Point2f> cur_pts;
        std::vector<double> displacements;
        ref_pts.reserve(track_ids.size());
        cur_pts.reserve(track_ids.size());
        displacements.reserve(track_ids.size());

        for (size_t i = 0; i < track_ids.size(); ++i) {
            auto it = reference_.find(track_ids[i]);
            if (it == reference_.end()) {
                continue;
            }
            ref_pts.push_back(it->second);
            cur_pts.push_back(points[i]);
            cv::Point2f d = points[i] - it->second;
            displacements.push_back(std::sqrt(static_cast<double>(d.x * d.x + d.y * d.y)));
        }

        // Too little overlap with the reference (e.g. after a re-detection): the
        // reference is stale, so anchor a fresh one on the current frame.
        if (static_cast<int>(ref_pts.size()) < config_.min_shared_to_keep) {
            set_reference(track_ids, points);
            return false;
        }

        // Median parallax against the reference.
        std::vector<double> sorted = displacements;
        std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
        last_parallax_ = sorted[sorted.size() / 2];

        if (static_cast<int>(ref_pts.size()) < config_.min_correspondences ||
            last_parallax_ < config_.min_parallax_px) {
            return false;  // Keep accumulating baseline.
        }

        ReconstructionResult result = reconstructor_.reconstruct(ref_pts, cur_pts);
        last_result_ = result;

        if (result.success) {
            cloud_ = result.points;
            has_cloud_ = true;
            set_reference(track_ids, points);  // Promote current frame to keyframe.
            return true;
        }

        // Reconstruction failed despite parallax (degenerate motion). If the
        // baseline is already very wide, advance the keyframe anyway to recover.
        if (last_parallax_ > config_.force_keyframe_px) {
            set_reference(track_ids, points);
        }
        return false;
    }

    void IncrementalMapper::reset() {
        reference_.clear();
        has_reference_ = false;
        cloud_.clear();
        has_cloud_ = false;
        last_parallax_ = 0.0;
        last_result_ = ReconstructionResult{};
    }

}  // namespace ar_slam
