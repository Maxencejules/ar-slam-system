#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
#include <chrono>
#include "core/frame.h"
#include "core/feature_tracker.h"
#include "rendering/gl_viewer.h"

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

        // Convert 2D points to 3D with better depth estimation
        std::vector<cv::Point3f> points_3d;
        for (size_t i = 0; i < result.curr_points.size(); ++i) {
            const auto& pt = result.curr_points[i];

            // Normalize coordinates
            float x = (pt.x - frame.cols/2) / 200.0f;
            float y = -(pt.y - frame.rows/2) / 200.0f;  // Flip Y for OpenGL

            // Estimate depth based on feature position and movement
            float base_depth = 2.0f;

            // Features higher in frame are typically farther away
            float depth_from_position = base_depth + (frame.rows - pt.y) / frame.rows * 1.5f;

            // If we have previous points, use motion for depth hint
            float depth_variation = 0.0f;
            if (i < result.prev_points.size()) {
                float motion = cv::norm(pt - result.prev_points[i]);
                // Less motion often means farther away
                depth_variation = (5.0f - std::min(motion, 5.0f)) * 0.05f;
            }

            // Add some random variation for visual effect
            float random_depth = ((rand() % 100) - 50) * 0.001f;

            float z = depth_from_position + depth_variation + random_depth;
            points_3d.push_back(cv::Point3f(x, y, z));
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
        int overlay_height = 140;
        int overlay_width = 250;
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