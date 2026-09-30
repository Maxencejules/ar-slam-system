#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include "core/reconstruction.h"

namespace synthetic {
    struct Scene {
        cv::Matx33d K{525, 0, 320, 0, 510, 240, 0, 0, 1};
        cv::Matx33d R;
        cv::Vec3d t{-0.6, 0.02, 0.04};
        std::vector<cv::Point3d> world;
        std::vector<cv::Point2f> first, second;
        std::vector<bool> outlier;
    };
    inline cv::Matx33d rotation_y(double degrees) {
        const double angle = degrees * CV_PI / 180;
        const double c = std::cos(angle), s = std::sin(angle);
        return {c, 0, s, 0, 1, 0, -s, 0, c};
    }
    // Direct pinhole projection of known scene coordinates: does not invoke
    // the production geometry::project or triangulation/eigensolver.
    inline cv::Point2f pixel(const cv::Matx33d& K,
                             const cv::Matx33d& R,
                             const cv::Vec3d& t,
                             const cv::Point3d& point) {
        const auto camera = R * cv::Vec3d(point.x, point.y, point.z) + t;
        return {static_cast<float>(K(0, 0) * camera[0] / camera[2] + K(0, 2)),
                static_cast<float>(K(1, 1) * camera[1] / camera[2] + K(1, 2))};
    }
    inline Scene scene(std::uint32_t seed = 2026, double noise_px = 0, int outlier_stride = 0) {
        Scene result;
        result.R = rotation_y(5);
        auto uniform = [&seed] {
            seed = seed * 1664525u + 1013904223u;
            return static_cast<double>(seed >> 8) / 16777216.0;
        };
        for (int i = 0; i < 240; ++i) {
            const double x = 2.4 * uniform() - 1.2;
            const double y = 1.6 * uniform() - 0.8;
            const double z = 4 + 4 * uniform();
            result.world.emplace_back(x, y, z);
        }
        for (const auto& point : result.world) {
            result.first.push_back(pixel(result.K, cv::Matx33d::eye(), {0, 0, 0}, point));
            result.second.push_back(pixel(result.K, result.R, result.t, point));
        }
        const auto original_second = result.second;
        for (std::size_t i = 0; i < result.world.size(); ++i) {
            const bool corrupt =
                outlier_stride > 0 && i % static_cast<std::size_t>(outlier_stride) == 0;
            result.outlier.push_back(corrupt);
            if (corrupt)
                result.second[i] = original_second[(i + 37) % result.world.size()];
            for (auto* observation : {&result.first[i], &result.second[i]}) {
                observation->x += static_cast<float>((2 * uniform() - 1) * noise_px);
                observation->y += static_cast<float>((2 * uniform() - 1) * noise_px);
            }
        }
        return result;
    }
    inline double angle_degrees(double cosine) {
        return std::acos(std::clamp(cosine, -1.0, 1.0)) * 180 / CV_PI;
    }
    struct Metrics {
        double rotation_error_deg = 0;
        double translation_direction_error_deg = 0;
        double structure_rmse = 0;
        double max_reprojection_px = 0;
        double min_angle_deg = 180;
        int true_points = 0, retained_outliers = 0;
    };
    inline Metrics metrics(const Scene& scene, const ar_slam::ReconstructionResult& result) {
        Metrics measured;
        const auto difference = scene.R * result.R.t();
        measured.rotation_error_deg =
            angle_degrees((difference(0, 0) + difference(1, 1) + difference(2, 2) - 1) * 0.5);
        measured.translation_direction_error_deg =
            angle_degrees(scene.t.dot(result.t) / (cv::norm(scene.t) * cv::norm(result.t)));
        const double baseline = cv::norm(scene.t);
        const auto center = -result.R.t() * result.t;
        double square_sum = 0;
        for (std::size_t i = 0; i < result.points.size(); ++i) {
            const auto index = static_cast<std::size_t>(result.point_indices[i]);
            const auto& point = result.points[i];
            const cv::Vec3d first(point.x, point.y, point.z);
            const auto second = result.R * first + result.t;
            const auto& truth = scene.world[index];
            if (scene.outlier[index]) {
                ++measured.retained_outliers;
            } else {
                square_sum += std::pow(first[0] * baseline - truth.x, 2) +
                              std::pow(first[1] * baseline - truth.y, 2) +
                              std::pow(first[2] * baseline - truth.z, 2);
                ++measured.true_points;
            }
            const auto projected1 =
                pixel(scene.K, cv::Matx33d::eye(), {0, 0, 0}, {first[0], first[1], first[2]});
            const auto projected2 =
                pixel(scene.K, cv::Matx33d::eye(), {0, 0, 0}, {second[0], second[1], second[2]});
            measured.max_reprojection_px =
                std::max<double>(measured.max_reprojection_px,
                                 std::max(std::hypot(projected1.x - scene.first[index].x,
                                                     projected1.y - scene.first[index].y),
                                          std::hypot(projected2.x - scene.second[index].x,
                                                     projected2.y - scene.second[index].y)));
            measured.min_angle_deg =
                std::min(measured.min_angle_deg,
                         angle_degrees(first.dot(first - center) /
                                       (cv::norm(first) * cv::norm(first - center))));
        }
        measured.structure_rmse =
            measured.true_points ? std::sqrt(square_sum / measured.true_points) : INFINITY;
        return measured;
    }
    inline cv::Mat texture(int seed) {
        cv::RNG rng(seed);
        cv::Mat image(480, 640, CV_8UC1);
        rng.fill(image, cv::RNG::UNIFORM, 40, 120);
        for (int i = 0; i < 200; ++i) {
            const int x = rng.uniform(10, 610), y = rng.uniform(10, 450);
            const int width = rng.uniform(8, 28), height = rng.uniform(8, 28);
            const int color = rng.uniform(150, 255);
            cv::rectangle(image, {x, y}, {x + width, y + height}, cv::Scalar(color), -1);
        }
        return image;
    }
    inline cv::Mat translated(const cv::Mat& image, double dx, double dy) {
        cv::Mat result;
        const cv::Matx23d transform(1, 0, dx, 0, 1, dy);
        cv::warpAffine(image, result, transform, image.size());
        return result;
    }
}  // namespace synthetic
