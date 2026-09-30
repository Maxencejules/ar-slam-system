// Deterministic synthetic experiment, not a recording of a real SLAM run.
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <numeric>
#include <stdexcept>
#include <opencv2/core/ocl.hpp>
#include "core/feature_tracker.h"
#include "core/incremental_mapper.h"
#include "synthetic_scene.h"

namespace {
    void require(bool condition, const char* message) {
        if (!condition)
            throw std::runtime_error(message);
    }
    std::ofstream output(const std::filesystem::path& path) {
        std::ofstream stream(path, std::ios::binary);
        stream.exceptions(std::ios::failbit | std::ios::badbit);
        stream.imbue(std::locale::classic());
        stream << std::setprecision(17);
        return stream;
    }
    ar_slam::ReconstructionResult reconstruct(const synthetic::Scene& scene) {
        cv::setRNGSeed(2026);
        auto result =
            ar_slam::TwoViewReconstruction(scene.K).reconstruct(scene.first, scene.second);
        require(result.success, "Synthetic reconstruction failed");
        return result;
    }
    void verify(const synthetic::Metrics& measured, bool noisy) {
        require(measured.rotation_error_deg < (noisy ? 0.75 : 0.01),
                "Rotation ground truth mismatch");
        require(measured.translation_direction_error_deg < (noisy ? 6 : 0.1),
                "Translation ground truth mismatch");
        require(measured.structure_rmse < (noisy ? 0.35 : 0.002),
                "Structure ground truth mismatch");
        require(measured.max_reprojection_px <= 2.001, "Reprojection error exceeded gate");
        require(measured.min_angle_deg > 1, "Insufficient triangulation angle");
        require(measured.retained_outliers <= 2, "Too many known mismatches retained");
    }
}  // namespace
int main(int argc, char** argv) {
    try {
        std::filesystem::path directory = "artifacts/offline";
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout
                << "offline_demo [--output directory]: fixed seed2026, synthetic offline proof\n";
            return 0;
        }
        if (argc != 1) {
            if (argc != 3 || std::string(argv[1]) != "--output")
                throw std::invalid_argument("Usage: offline_demo [--output directory]");
            directory = argv[2];
        }
        cv::setNumThreads(1);
        cv::ocl::setUseOpenCL(false);
        const auto clean = synthetic::scene();
        const auto noisy = synthetic::scene(2026, 0.15, 7);
        const auto clean_result = reconstruct(clean), noisy_result = reconstruct(noisy);
        const auto clean_metrics = synthetic::metrics(clean, clean_result);
        const auto noisy_metrics = synthetic::metrics(noisy, noisy_result);
        verify(clean_metrics, false);
        verify(noisy_metrics, true);
        ar_slam::IncrementalMapper::Config config;
        config.min_parallax_px = 3;
        ar_slam::IncrementalMapper mapper(clean.K, config);
        std::vector<int> ids(clean.world.size());
        std::iota(ids.begin(), ids.end(), 0);
        require(!mapper.update(ids, clean.first), "Unexpected initial cloud");
        cv::setRNGSeed(2026);
        require(mapper.update(ids, clean.second), "Mapper did not reconstruct known pair");
        require(mapper.cloud_reference_index() == 1 && mapper.cloud_current_index() == 2,
                "Incorrect pair-local frame metadata");

        const auto image = synthetic::texture(2026);
        const auto shifted = synthetic::translated(image, 4, 3);
        ar_slam::FeatureTracker tracker;
        using Frame = ar_slam::Frame;
        const auto first = tracker.track_features(
            std::make_shared<Frame>(image, Frame::Timestamp{} + std::chrono::milliseconds(0)));
        const auto second = tracker.track_features(
            std::make_shared<Frame>(shifted, Frame::Timestamp{} + std::chrono::milliseconds(20)));
        std::vector<double> tracking_errors;
        for (std::size_t i = 0; i < second.curr_points.size(); ++i) {
            if (second.inliers[i])
                tracking_errors.push_back(
                    std::hypot(second.curr_points[i].x - second.prev_points[i].x - 4,
                               second.curr_points[i].y - second.prev_points[i].y - 3));
        }
        std::sort(tracking_errors.begin(), tracking_errors.end());
        require(tracking_errors.size() > 150 && second.tracking_quality > 0.5,
                "Insufficient tracked synthetic correspondences");
        const double tracking_median = tracking_errors[tracking_errors.size() / 2];
        require(tracking_median < 0.2, "Tracked displacement differs from analytic warp");

        std::filesystem::create_directories(directory);
        auto scene_csv = output(directory / "scene.csv");
        scene_csv << "index,x_scene,y_scene,z_scene,u1_px,v1_px,u2_px,v2_px,known_mismatch\n";
        for (std::size_t i = 0; i < noisy.world.size(); ++i) {
            const auto& point = noisy.world[i];
            scene_csv << i << ',' << point.x << ',' << point.y << ',' << point.z << ','
                      << noisy.first[i].x << ',' << noisy.first[i].y << ',' << noisy.second[i].x
                      << ',' << noisy.second[i].y << ',' << noisy.outlier[i] << '\n';
        }
        auto cloud = output(directory / "cloud.ply");
        cloud << "ply\nformat ascii 1.0\ncomment SYNTHETIC estimated camera1 cloud in baseline "
                 "units\n"
              << "element vertex " << noisy_result.points.size()
              << "\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
        for (const auto& point : noisy_result.points)
            cloud << point.x << ' ' << point.y << ' ' << point.z << '\n';
        auto points = output(directory / "estimated_points.csv");
        points << "input_index,x_baseline,y_baseline,z_baseline\n";
        for (std::size_t i = 0; i < noisy_result.points.size(); ++i) {
            const auto& point = noisy_result.points[i];
            points << noisy_result.point_indices[i] << ',' << point.x << ',' << point.y << ','
                   << point.z << '\n';
        }
        auto tracks = output(directory / "tracking.csv");
        tracks << "id,has_previous,u_previous,v_previous,u_current,v_current\n";
        for (std::size_t i = 0; i < second.curr_points.size(); ++i)
            tracks << second.track_ids[i] << ',' << second.inliers[i] << ','
                   << second.prev_points[i].x << ',' << second.prev_points[i].y << ','
                   << second.curr_points[i].x << ',' << second.curr_points[i].y << '\n';
        auto measured = output(directory / "metrics.csv");
        measured << "scenario,model_inliers,accepted_points,true_points,retained_mismatches,"
                    "rotation_error_deg,translation_error_deg,structure_rmse_scene,max_"
                    "reprojection_px,min_angle_deg\n";
        auto row = [&](const char* name, const ar_slam::ReconstructionResult& result,
                       const synthetic::Metrics& metric) {
            measured << name << ',' << result.num_model_inliers << ',' << result.num_inliers << ','
                     << metric.true_points << ',' << metric.retained_outliers << ','
                     << metric.rotation_error_deg << ',' << metric.translation_direction_error_deg
                     << ',' << metric.structure_rmse << ',' << metric.max_reprojection_px << ','
                     << metric.min_angle_deg << '\n';
        };
        row("clean", clean_result, clean_metrics);
        row("noise_and_mismatches", noisy_result, noisy_metrics);
        auto report = output(directory / "report.json");
        report
            << "{\n  \"provenance\": \"Generated synthetic experiments; no real camera "
               "measurements\",\n"
            << "  \"seed\": 2026,\n  \"points\": 240,\n  \"noise_uniform_half_width_px\": 0.15,\n"
            << "  \"mismatch_stride\": 7,\n  \"known_baseline_scene_units\": " << cv::norm(clean.t)
            << ",\n"
            << "  \"opencv\": \"" << CV_VERSION << "\",\n"
#ifdef _MSC_VER
            << "  \"compiler_msvc_full_version\": " << _MSC_FULL_VER << ",\n"
#else
            << "  \"compiler\": \"" << __VERSION__ << "\",\n"
#endif
            << "  \"opencv_threads\": 1,\n  \"opencl\": false,\n"
            << "  \"tracker_initial_observations\": " << first.num_tracked << ",\n"
            << "  \"tracker_temporal_correspondences\": " << second.num_inliers << ",\n"
            << "  \"tracker_retention\": " << second.tracking_quality << ",\n"
            << "  \"tracker_median_warp_error_px\": " << tracking_median << ",\n"
            << "  \"cloud_reference_update\": " << mapper.cloud_reference_index() << ",\n"
            << "  \"cloud_current_update\": " << mapper.cloud_current_index() << ",\n"
            << "  \"calibration\": [" << clean.K(0, 0) << ',' << clean.K(1, 1) << ','
            << clean.K(0, 2) << ',' << clean.K(1, 2) << "],\n"
            << "  \"known_rotation_y_deg\": 5,\n"
            << "  \"known_translation_scene_units\": [" << clean.t[0] << ',' << clean.t[1] << ','
            << clean.t[2] << "],\n  \"estimated_rotation_row_major\": [";
        for (int i = 0; i < 9; ++i)
            report << (i ? "," : "") << noisy_result.R.val[i];
        report << "],\n  \"estimated_translation_baseline_units\": [" << noisy_result.t[0] << ','
               << noisy_result.t[1] << ',' << noisy_result.t[2] << "]\n}\n";
        std::cout << "Verified synthetic geometry: clean " << clean_result.num_inliers
                  << ", noisy/mismatched " << noisy_result.num_inliers << " accepted points.\n"
                  << "Verified frontend warp: " << second.num_inliers
                  << " temporal tracks, median error " << tracking_median
                  << " px.\nArtifacts: " << directory.string() << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
