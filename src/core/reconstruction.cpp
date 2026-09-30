#include "core/reconstruction.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <opencv2/calib3d.hpp>
#include "core/geometry.h"

namespace ar_slam {
    namespace {
        geometry::Mat3 to_geom_mat3(const cv::Matx33d& value) {
            geometry::Mat3 result;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    result.m[i][j] = value(i, j);
            return result;
        }
        bool positive(double value) {
            return std::isfinite(value) && value > 0;
        }
        double median(std::vector<double> values) {
            std::sort(values.begin(), values.end());
            const auto middle = values.size() / 2;
            return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) * 0.5;
        }
        cv::Point2d project(const cv::Matx33d& K, const cv::Vec3d& point) {
            const auto pixel = K * point;
            return {pixel[0] / pixel[2], pixel[1] / pixel[2]};
        }
    }  // namespace

    TwoViewReconstruction::TwoViewReconstruction(const cv::Matx33d& K)
        : TwoViewReconstruction(K, Config{}) {}
    TwoViewReconstruction::TwoViewReconstruction(const cv::Matx33d& K, const Config& config)
        : K_(K), config_(config) {
        for (double value : K.val) {
            if (!std::isfinite(value))
                throw std::invalid_argument("Calibration must be finite");
        }
        if (!positive(K(0, 0)) || !positive(K(1, 1)) || std::abs(K(0, 1)) > 1e-12 ||
            std::abs(K(1, 0)) > 1e-12 || std::abs(K(2, 0)) > 1e-12 || std::abs(K(2, 1)) > 1e-12 ||
            std::abs(K(2, 2) - 1) > 1e-12) {
            throw std::invalid_argument(
                "Calibration must be a positive-focal, zero-skew pinhole K");
        }
        if (!positive(config.ransac_prob) || config.ransac_prob >= 1 ||
            !positive(config.ransac_threshold) || config.min_correspondences < 5 ||
            config.min_inliers < 5 || !positive(config.min_inlier_ratio) ||
            config.min_inlier_ratio > 1 || !positive(config.max_depth) ||
            !positive(config.max_reprojection_error_px) ||
            !positive(config.min_triangulation_angle_deg) ||
            config.min_triangulation_angle_deg >= 90) {
            throw std::invalid_argument("Invalid reconstruction thresholds");
        }
    }

    ReconstructionResult TwoViewReconstruction::reconstruct(
        const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2) const {
        if (pts1.size() != pts2.size() ||
            pts1.size() < static_cast<std::size_t>(config_.min_correspondences))
            return {};
        for (std::size_t i = 0; i < pts1.size(); ++i) {
            if (!std::isfinite(pts1[i].x) || !std::isfinite(pts1[i].y) ||
                !std::isfinite(pts2[i].x) || !std::isfinite(pts2[i].y))
                return {};
        }
        cv::Mat R, t, mask;
        int model_inliers = 0;
        try {
            const cv::Mat K(K_);
            cv::Mat essential = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, config_.ransac_prob,
                                                     config_.ransac_threshold, mask);
            if (essential.rows != 3 || essential.cols != 3 || mask.empty())
                return {};
            model_inliers = cv::countNonZero(mask);
            // Refine the minimal RANSAC hypothesis using its consensus. The
            // normalized eight-point fit uses all accepted correspondences;
            // equal nonzero singular values enforce the essential constraint.
            std::vector<cv::Point2d> normalized1, normalized2;
            for (std::size_t i = 0; i < pts1.size(); ++i) {
                if (mask.at<uchar>(static_cast<int>(i)) == 0)
                    continue;
                normalized1.emplace_back((pts1[i].x - K_(0, 2)) / K_(0, 0),
                                         (pts1[i].y - K_(1, 2)) / K_(1, 1));
                normalized2.emplace_back((pts2[i].x - K_(0, 2)) / K_(0, 0),
                                         (pts2[i].y - K_(1, 2)) / K_(1, 1));
            }
            if (normalized1.size() >= 8) {
                const cv::Mat fitted =
                    cv::findFundamentalMat(normalized1, normalized2, cv::FM_8POINT);
                if (fitted.rows == 3 && fitted.cols == 3 && cv::checkRange(fitted)) {
                    cv::SVD decomposition(fitted, cv::SVD::FULL_UV);
                    const double sigma =
                        (decomposition.w.at<double>(0) + decomposition.w.at<double>(1)) * 0.5;
                    const cv::Matx33d constrained(sigma, 0, 0, 0, sigma, 0, 0, 0, 0);
                    essential = decomposition.u * cv::Mat(constrained) * decomposition.vt;
                }
            }
            if (cv::recoverPose(essential, pts1, pts2, K, R, t, config_.max_depth, mask) <= 0)
                return {};
        } catch (const cv::Exception&) {
            return {};  // Model estimation can fail on degenerate observations.
        }
        cv::Matx33d rotation;
        cv::Vec3d translation;
        for (int i = 0; i < 3; ++i) {
            translation[i] = t.at<double>(i, 0);
            if (!std::isfinite(translation[i]))
                return {};
            for (int j = 0; j < 3; ++j) {
                rotation(i, j) = R.at<double>(i, j);
                if (!std::isfinite(rotation(i, j)))
                    return {};
            }
        }
        const auto Kg = to_geom_mat3(K_);
        const auto P1 = geometry::make_projection(Kg, geometry::Mat3::identity(), {0, 0, 0});
        const auto P2 = geometry::make_projection(Kg, to_geom_mat3(rotation),
                                                  {translation[0], translation[1], translation[2]});
        const auto inverse = K_.inv();
        ReconstructionResult result;
        std::vector<double> errors, angles;
        for (std::size_t i = 0; i < pts1.size(); ++i) {
            if (mask.empty() || mask.at<uchar>(static_cast<int>(i)) == 0)
                continue;
            const auto tri =
                geometry::triangulate(P1, P2, pts1[i].x, pts1[i].y, pts2[i].x, pts2[i].y);
            if (!tri.valid)
                continue;
            const cv::Vec3d first(tri.point[0], tri.point[1], tri.point[2]);
            const auto second = rotation * first + translation;
            if (first[2] <= 0 || second[2] <= 0 || first[2] > config_.max_depth ||
                second[2] > config_.max_depth)
                continue;
            bool representable = true;
            for (double value : first.val) {
                representable = representable && std::isfinite(value) &&
                                std::abs(value) <= std::numeric_limits<float>::max();
            }
            if (!representable)
                continue;
            const auto ray1 = inverse * cv::Vec3d(pts1[i].x, pts1[i].y, 1);
            const auto ray2 = rotation.t() * inverse * cv::Vec3d(pts2[i].x, pts2[i].y, 1);
            const double cosine = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));
            const double angle = std::acos(std::clamp(cosine, -1.0, 1.0)) * 180 / CV_PI;
            const auto pixel1 = project(K_, first), pixel2 = project(K_, second);
            const double error = std::max(std::hypot(pixel1.x - pts1[i].x, pixel1.y - pts1[i].y),
                                          std::hypot(pixel2.x - pts2[i].x, pixel2.y - pts2[i].y));
            if (!std::isfinite(angle) || !std::isfinite(error) ||
                angle < config_.min_triangulation_angle_deg ||
                error > config_.max_reprojection_error_px)
                continue;
            result.points.emplace_back(static_cast<float>(first[0]), static_cast<float>(first[1]),
                                       static_cast<float>(first[2]));
            result.point_indices.push_back(static_cast<int>(i));
            errors.push_back(error);
            angles.push_back(angle);
        }
        result.num_model_inliers = model_inliers;
        result.num_inliers = static_cast<int>(result.points.size());
        result.inlier_ratio =
            static_cast<double>(result.num_inliers) / static_cast<double>(pts1.size());
        if (result.num_inliers < config_.min_inliers ||
            result.inlier_ratio < config_.min_inlier_ratio)
            return {};
        result.R = rotation;
        result.t = translation;
        result.median_reprojection_error_px = median(errors);
        result.median_triangulation_angle_deg = median(angles);
        result.success = true;
        return result;
    }
}  // namespace ar_slam
