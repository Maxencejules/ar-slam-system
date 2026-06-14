#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <map>
#include <memory>
#include <set>
#include <vector>
#include "core/frame.h"
#include "core/feature_tracker.h"
#include "core/incremental_mapper.h"
#include "rendering/gl_viewer.h"

namespace {

// Pinhole intrinsics from frame size, using focal ~ max(width, height). This is
// the standard uncalibrated default; for metric results, calibrate the camera
// and supply a real K.
cv::Matx33d default_intrinsics(const cv::Size& size) {
    double f = static_cast<double>(std::max(size.width, size.height));
    return cv::Matx33d(f, 0, size.width / 2.0, 0, f, size.height / 2.0, 0, 0, 1);
}

// Centre and scale a reconstructed cloud for the orbit viewer (Y flipped to
// OpenGL's up axis). Relative geometry is preserved; only the global pose and
// scale are normalised for display.
std::vector<cv::Point3f> to_display_cloud(const std::vector<cv::Point3f>& pts) {
    std::vector<cv::Point3f> out;
    if (pts.empty()) return out;

    double cx = 0, cy = 0, cz = 0;
    for (const auto& p : pts) {
        cx += p.x;
        cy += p.y;
        cz += p.z;
    }
    cx /= pts.size();
    cy /= pts.size();
    cz /= pts.size();

    std::vector<float> radii;
    radii.reserve(pts.size());
    for (const auto& p : pts) {
        double dx = p.x - cx, dy = p.y - cy, dz = p.z - cz;
        radii.push_back(static_cast<float>(std::sqrt(dx * dx + dy * dy + dz * dz)));
    }
    std::nth_element(radii.begin(), radii.begin() + radii.size() / 2, radii.end());
    float median_radius = radii[radii.size() / 2];
    float scale = (median_radius > 1e-3f) ? 2.5f / median_radius : 1.0f;

    out.reserve(pts.size());
    for (const auto& p : pts) {
        out.push_back(cv::Point3f(static_cast<float>(p.x - cx) * scale,
                                  static_cast<float>(-(p.y - cy)) * scale,
                                  static_cast<float>(p.z - cz) * scale));
    }
    return out;
}

// Until enough parallax accrues, show the live features on a frontal plane.
// This is an honest 2D projection (constant depth) rather than invented depth.
std::vector<cv::Point3f> features_on_plane(const std::vector<cv::Point2f>& pts,
                                           const cv::Matx33d& K, float depth) {
    std::vector<cv::Point3f> out;
    out.reserve(pts.size());
    double fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);
    for (const auto& p : pts) {
        float X = static_cast<float>((p.x - cx) / fx) * depth;
        float Y = static_cast<float>((p.y - cy) / fy) * depth;
        out.push_back(cv::Point3f(X, -Y, 0.0f));
    }
    return out;
}

}  // namespace

int main() {
    std::cout << "=== 3D Camera Test ===" << std::endl;

    // Initialize camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera" << std::endl;
        return -1;
    }

    // Initialize 3D viewer
    ar_slam::GLViewer viewer("AR SLAM - 3D Point Cloud");
    if (!viewer.init()) {
        std::cerr << "Cannot initialize 3D viewer" << std::endl;
        return -1;
    }

    ar_slam::FeatureTracker tracker;
    std::unique_ptr<ar_slam::IncrementalMapper> mapper;  // created once frame size is known
    cv::Mat frame;

    // Trail history for 2D visualization
    const int TRAIL_LENGTH = 10;
    std::map<int, std::deque<cv::Point2f>> feature_trails;

    // FPS tracking
    int frame_count = 0;
    auto last_time = std::chrono::high_resolution_clock::now();
    float fps = 0.0f;

    std::cout << "Controls:" << std::endl;
    std::cout << "  Arrow Keys: Rotate/Zoom camera" << std::endl;
    std::cout << "  Q: Quit" << std::endl;
    std::cout << "  Space: Print stats" << std::endl;

    while (!viewer.should_close()) {
        cap >> frame;
        if (frame.empty()) break;

        auto slam_frame = std::make_shared<ar_slam::Frame>(frame);
        auto result = tracker.track_features(slam_frame);

        // Lazily build the intrinsics + mapper once we know the frame size.
        if (!mapper) {
            mapper = std::make_unique<ar_slam::IncrementalMapper>(default_intrinsics(frame.size()));
        }

        // Feed the tracks to the mapper. Once it has triangulated real structure
        // from a wide-enough baseline, show that; until then show the live
        // features on a frontal plane (an honest 2D projection, not fake depth).
        mapper->update(result.track_ids, result.curr_points);

        std::vector<cv::Point3f> points_3d;
        if (mapper->has_cloud()) {
            points_3d = to_display_cloud(mapper->cloud());
        } else {
            points_3d = features_on_plane(result.curr_points, default_intrinsics(frame.size()), 1.0f);
        }

        // Update 3D viewer
        viewer.update_points(points_3d);

        // Calculate FPS
        frame_count++;
        auto now = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration<float>(now - last_time).count();

        if (elapsed > 1.0f) {
            fps = frame_count / elapsed;
            frame_count = 0;
            last_time = now;

            // Update window title with stats
            std::string title = "AR SLAM 3D - FPS: " + std::to_string((int)fps) +
                               " Points: " + std::to_string(points_3d.size());
            viewer.set_title(title);
        }

        // Render 3D view
        if (!viewer.render()) break;

        // Show 2D view with overlays
        cv::Mat display = frame.clone();

        // Update trails
        std::set<int> current_ids;
        for(size_t i = 0; i < result.track_ids.size(); ++i) {
            int id = result.track_ids[i];
            current_ids.insert(id);
            feature_trails[id].push_back(result.curr_points[i]);
            if(feature_trails[id].size() > TRAIL_LENGTH) {
                feature_trails[id].pop_front();
            }
        }

        // Clean up old trails
        for(auto it = feature_trails.begin(); it != feature_trails.end(); ) {
            if(current_ids.find(it->first) == current_ids.end()) {
                it = feature_trails.erase(it);
            } else {
                ++it;
            }
        }

        // Draw trails
        for (const auto& [id, trail] : feature_trails) {
            if (trail.size() < 2) continue;
            for (size_t i = 0; i < trail.size() - 1; ++i) {
                float opacity = (float)(i + 1) / trail.size();
                cv::line(display, trail[i], trail[i+1],
                        cv::Scalar(0, 255 * opacity, 0), 1, cv::LINE_AA);
            }
        }

        // Draw tracked points
        for (size_t i = 0; i < result.curr_points.size(); ++i) {
            cv::circle(display, result.curr_points[i], 3, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
        }

        // Draw UI overlay
        int overlay_height = 170;
        int overlay_width = 330;
        cv::Mat overlay = display(cv::Rect(0, 0, overlay_width, overlay_height));
        cv::Mat dimmed;
        double alpha = 0.5;
        overlay.copyTo(dimmed);
        cv::rectangle(dimmed, cv::Rect(0, 0, overlay_width, overlay_height), cv::Scalar(0, 0, 0), -1);
        cv::addWeighted(dimmed, alpha, overlay, 1.0 - alpha, 0, overlay);

        // Add text overlays
        cv::putText(display, "FPS: " + std::to_string((int)fps),
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                   0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        cv::putText(display, "Tracked: " + std::to_string(result.num_tracked),
                   cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX,
                   0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        cv::putText(display, "Quality: " + std::to_string((int)(result.tracking_quality * 100)) + "%",
                   cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX,
                   0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        cv::putText(display, "3D Points: " + std::to_string(points_3d.size()),
                   cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX,
                   0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        // Mapping status: shows whether we are triangulating real structure or
        // still gathering baseline.
        std::string map_status =
            mapper->has_cloud()
                ? "Map: " + std::to_string(mapper->cloud().size()) + " pts (triangulated)"
                : "Map: gathering baseline (" + std::to_string((int)mapper->last_parallax()) + "px)";
        cv::putText(display, map_status, cv::Point(10, 150), cv::FONT_HERSHEY_SIMPLEX,
                   0.55, cv::Scalar(0, 220, 255), 1, cv::LINE_AA);

        cv::imshow("2D View", display);

        char key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
        if (key == ' ') {
            std::cout << "\n=== Frame Stats ===" << std::endl;
            std::cout << "FPS: " << fps << std::endl;
            std::cout << "Tracked Features: " << result.num_tracked << std::endl;
            std::cout << "Tracking Quality: " << result.tracking_quality << std::endl;
            std::cout << "3D Points: " << points_3d.size() << std::endl;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    std::cout << "\nShutting down..." << std::endl;
    return 0;
}