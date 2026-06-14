#pragma once

#include <opencv2/core.hpp>
#include <unordered_map>
#include <vector>

#include "core/reconstruction.h"

namespace ar_slam {

/**
 * @brief Turns a stream of feature tracks into sparse 3D structure.
 *
 * The mapper holds a reference keyframe (a snapshot of track id -> pixel
 * observations). On each update it matches the current tracks against the
 * reference by track id, measures the parallax (median pixel displacement),
 * and once the baseline is wide enough it runs two-view reconstruction on the
 * matched correspondences. A successful reconstruction promotes the current
 * frame to the new reference keyframe, so the structure is always triangulated
 * from a pair with genuine parallax rather than from noise on a near-static
 * pair.
 *
 * The recovered cloud is expressed in the reference camera frame, up to the
 * usual monocular scale ambiguity.
 */
class IncrementalMapper {
public:
    struct Config {
        double min_parallax_px = 20.0;   ///< Min median displacement to triangulate.
        int min_correspondences = 40;    ///< Min shared tracks to attempt reconstruction.
        int min_shared_to_keep = 12;     ///< Below this, drop the stale reference keyframe.
        double force_keyframe_px = 80.0;  ///< Parallax beyond which we advance the keyframe
                                          ///< even if reconstruction failed (e.g. pure rotation).
    };

    /// Construct with default thresholds.
    explicit IncrementalMapper(const cv::Matx33d& K);

    /// Construct with explicit thresholds.
    IncrementalMapper(const cv::Matx33d& K, const Config& config);

    /**
     * @brief Feed the current frame's tracks.
     * @param track_ids  Stable identifier per tracked feature.
     * @param points     Pixel location of each tracked feature (same size as ids).
     * @return true if a new 3D cloud was produced on this update.
     */
    bool update(const std::vector<int>& track_ids, const std::vector<cv::Point2f>& points);

    /// True once at least one successful reconstruction has been produced.
    bool has_cloud() const { return has_cloud_; }

    /// The most recent triangulated cloud (reference-camera frame, up to scale).
    const std::vector<cv::Point3f>& cloud() const { return cloud_; }

    /// Median parallax (px) measured against the reference on the last update.
    double last_parallax() const { return last_parallax_; }

    /// Result of the most recent reconstruction attempt.
    const ReconstructionResult& last_result() const { return last_result_; }

    /// Reset all state (drops the reference keyframe and the cloud).
    void reset();

private:
    cv::Matx33d K_;
    Config config_;
    TwoViewReconstruction reconstructor_;

    std::unordered_map<int, cv::Point2f> reference_;
    bool has_reference_ = false;

    std::vector<cv::Point3f> cloud_;
    bool has_cloud_ = false;
    double last_parallax_ = 0.0;
    ReconstructionResult last_result_;

    void set_reference(const std::vector<int>& ids, const std::vector<cv::Point2f>& pts);
};

}  // namespace ar_slam
