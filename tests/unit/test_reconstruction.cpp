#include <cmath>
#include <limits>
#include <stdexcept>
#include <opencv2/calib3d.hpp>
#include "core/geometry.h"
#include "core/reconstruction.h"
#include "synthetic_scene.h"
#include "test_util.h"

namespace {
    void check_scene(std::uint32_t seed, double noise, int stride) {
        const auto scene = synthetic::scene(seed, noise, stride);
        cv::setRNGSeed(2026);
        const auto result =
            ar_slam::TwoViewReconstruction(scene.K).reconstruct(scene.first, scene.second);
        CHECK(result.success);
        if (!result.success)
            return;
        CHECK(result.points.size() == result.point_indices.size());
        CHECK(result.num_inliers == static_cast<int>(result.points.size()));
        CHECK(result.num_model_inliers >= result.num_inliers);
        CHECK(result.num_model_inliers <= static_cast<int>(scene.world.size()));
        CHECK(result.num_inliers >= (stride ? 150 : 220));
        const auto measured = synthetic::metrics(scene, result);
        std::cout << "seed=" << seed << " noise_px=" << noise << " inliers=" << result.num_inliers
                  << " rotation_deg=" << measured.rotation_error_deg
                  << " translation_deg=" << measured.translation_direction_error_deg
                  << " structure_rmse=" << measured.structure_rmse << "\n";
        CHECK(measured.rotation_error_deg < (noise ? 0.75 : 0.01));
        CHECK(measured.translation_direction_error_deg < (noise ? 6 : 0.1));
        CHECK(measured.structure_rmse < (noise ? 0.35 : 0.002));
        CHECK(measured.max_reprojection_px <= 2.001);
        CHECK(measured.min_angle_deg > 1);
        CHECK(measured.retained_outliers <= 2);
        CHECK(result.median_triangulation_angle_deg >= 1);
        CHECK(result.median_reprojection_error_px <= 2);
        CHECK_NEAR(cv::norm(result.t), 1, 1e-9);
        std::vector<bool> seen(scene.world.size(), false);
        for (std::size_t i = 0; i < result.points.size(); ++i) {
            const auto index = static_cast<std::size_t>(result.point_indices[i]);
            CHECK(index < scene.world.size());
            CHECK(!seen[index]);
            seen[index] = true;
            const auto& point = result.points[i];
            const auto second = result.R * cv::Vec3d(point.x, point.y, point.z) + result.t;
            CHECK(std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z));
            CHECK(point.z > 0 && point.z <= 100);
            CHECK(second[2] > 0 && second[2] <= 100);
        }
    }
    void compare_independent_svd() {
        const auto scene = synthetic::scene();
        const auto& K = scene.K;
        const cv::Matx34d extrinsic1(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
        const cv::Matx34d extrinsic2(scene.R(0, 0), scene.R(0, 1), scene.R(0, 2), scene.t[0],
                                     scene.R(1, 0), scene.R(1, 1), scene.R(1, 2), scene.t[1],
                                     scene.R(2, 0), scene.R(2, 1), scene.R(2, 2), scene.t[2]);
        const auto first = K * extrinsic1, second = K * extrinsic2;
        std::vector<cv::Point2d> pixels1, pixels2;
        for (std::size_t i = 0; i < scene.first.size(); ++i) {
            pixels1.emplace_back(scene.first[i].x, scene.first[i].y);
            pixels2.emplace_back(scene.second[i].x, scene.second[i].y);
        }
        cv::Mat homogeneous;
        cv::triangulatePoints(first, second, pixels1, pixels2, homogeneous);
        ar_slam::geometry::Mat34 P1, P2;
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 4; ++column) {
                P1.m[row][column] = first(row, column);
                P2.m[row][column] = second(row, column);
            }
        for (std::size_t i = 0; i < scene.first.size(); ++i) {
            const auto actual = ar_slam::geometry::triangulate(P1, P2, pixels1[i].x, pixels1[i].y,
                                                               pixels2[i].x, pixels2[i].y);
            CHECK(actual.valid);
            const int column = static_cast<int>(i);
            for (int axis = 0; axis < 3; ++axis) {
                const double independent =
                    homogeneous.at<double>(axis, column) / homogeneous.at<double>(3, column);
                CHECK_NEAR(actual.point[axis], independent, 1e-5);
            }
        }
    }
    void test_failure_inputs() {
        const auto scene = synthetic::scene();
        ar_slam::TwoViewReconstruction recon(scene.K);
        CHECK(!recon.reconstruct({}, {}).success);
        CHECK(!recon.reconstruct(scene.first, {}).success);
        auto first = scene.first;
        first[3].x = std::numeric_limits<float>::quiet_NaN();
        CHECK(!recon.reconstruct(first, scene.second).success);
        first[3].x = std::numeric_limits<float>::infinity();
        CHECK(!recon.reconstruct(first, scene.second).success);
        CHECK(!recon.reconstruct(scene.first, scene.first).success);
        std::vector<cv::Point2f> identical(100, {320, 240});
        CHECK(!recon.reconstruct(identical, identical).success);
        auto config = ar_slam::TwoViewReconstruction::Config{};
        config.max_depth = 1e6;  // Isolate the angle gate from the depth threshold.
        ar_slam::TwoViewReconstruction low_parallax(scene.K, config);
        std::vector<cv::Point2f> rotation_only, tiny_translation;
        for (const auto& point : scene.world) {
            rotation_only.push_back(
                synthetic::pixel(scene.K, synthetic::rotation_y(8), {0, 0, 0}, point));
            tiny_translation.push_back(
                synthetic::pixel(scene.K, cv::Matx33d::eye(), {-0.001, 0, 0}, point));
        }
        CHECK(!low_parallax.reconstruct(scene.first, rotation_only).success);
        CHECK(!low_parallax.reconstruct(scene.first, tiny_translation).success);
        for (int mode = 0; mode < 4; ++mode) {
            auto invalid = scene.K;
            if (mode == 0)
                invalid(0, 0) = 0;
            if (mode == 1)
                invalid(0, 1) = 1;
            if (mode == 2)
                invalid(2, 2) = 0;
            if (mode == 3)
                invalid(1, 1) = std::numeric_limits<double>::quiet_NaN();
            CHECK(artest::throws<std::invalid_argument>(
                [&] { ar_slam::TwoViewReconstruction bad(invalid); }));
        }
        for (int mode = 0; mode < 7; ++mode) {
            auto invalid = ar_slam::TwoViewReconstruction::Config{};
            if (mode == 0)
                invalid.min_correspondences = 4;
            if (mode == 1)
                invalid.min_inliers = 0;
            if (mode == 2)
                invalid.ransac_prob = 1;
            if (mode == 3)
                invalid.min_inlier_ratio = -1;
            if (mode == 4)
                invalid.max_depth = INFINITY;
            if (mode == 5)
                invalid.max_reprojection_error_px = 0;
            if (mode == 6)
                invalid.min_triangulation_angle_deg = 90;
            CHECK(artest::throws<std::invalid_argument>(
                [&] { ar_slam::TwoViewReconstruction bad(scene.K, invalid); }));
        }
    }
}  // namespace
int main() {
    cv::setNumThreads(1);
    for (auto seed : {2026u, 7u, 55u}) {
        check_scene(seed, 0, 0);
        check_scene(seed, 0.15, 7);
    }
    compare_independent_svd();
    test_failure_inputs();
    return artest::report("test_reconstruction");
}
