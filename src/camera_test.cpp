#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
#include <chrono>
#include <deque>
#include <numeric>
#include <iomanip>
#include "core/frame.h"
#include "core/feature_tracker.h"
#include "rendering/gl_viewer.h"

// Class to track performance metrics
class PerformanceMonitor {
private:
    std::deque<double> fps_history_;
    std::deque<double> quality_history_;
    std::deque<int> tracked_history_;
    size_t max_history_ = 30;

public:
    void add_frame(double fps, double quality, int tracked) {
        fps_history_.push_back(fps);
        quality_history_.push_back(quality);
        tracked_history_.push_back(tracked);

        while (fps_history_.size() > max_history_) {
            fps_history_.pop_front();
            quality_history_.pop_front();
            tracked_history_.pop_front();
        }
    }

    double avg_fps() const {
        if (fps_history_.empty()) return 0;
        return std::accumulate(fps_history_.begin(), fps_history_.end(), 0.0) / fps_history_.size();
    }

    double avg_quality() const {
        if (quality_history_.empty()) return 0;
        return std::accumulate(quality_history_.begin(), quality_history_.end(), 0.0) / quality_history_.size();
    }

    double avg_tracked() const {
        if (tracked_history_.empty()) return 0;
        return std::accumulate(tracked_history_.begin(), tracked_history_.end(), 0.0) / tracked_history_.size();
    }

    double min_quality() const {
        if (quality_history_.empty()) return 0;
        return *std::min_element(quality_history_.begin(), quality_history_.end());
    }

    double max_quality() const {
        if (quality_history_.empty()) return 0;
        return *std::max_element(quality_history_.begin(), quality_history_.end());
    }
};

int main() {
    std::cout << "=== AR SLAM 3D Camera Test ===" << std::endl;
    std::cout << "Real-world performance monitoring enabled\n" << std::endl;

    // Initialize camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera" << std::endl;
        std::cerr << "Trying to use video file instead..." << std::endl;

        // Fallback to video file if available
        cap.open("test_video.mp4");
        if (!cap.isOpened()) {
            std::cerr << "No video source available" << std::endl;
            return -1;
        }
    }

    // Set camera properties for better performance
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);

    // Initialize 3D viewer
    ar_slam::GLViewer viewer("AR SLAM - 3D Point Cloud");
    if (!viewer.init()) {
        std::cerr << "Cannot initialize 3D viewer" << std::endl;
        return -1;
    }

    ar_slam::FeatureTracker tracker;
    cv::Mat frame;

    // Performance monitoring
    PerformanceMonitor monitor;
    int frame_count = 0;
    int low_quality_frames = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_frame_time = start_time;

    std::cout << "Controls:" << std::endl;
    std::cout << "  Arrow Keys: Rotate/zoom 3D view" << std::endl;
    std::cout << "  Space: Print detailed statistics" << std::endl;
    std::cout << "  R: Reset tracker" << std::endl;
    std::cout << "  Q/ESC: Quit" << std::endl;
    std::cout << "\nStarting real-time tracking...\n" << std::endl;

    while (!viewer.should_close()) {
        auto frame_start = std::chrono::high_resolution_clock::now();

        cap >> frame;
        if (frame.empty()) {
            std::cout << "End of video or camera disconnected" << std::endl;
            break;
        }

        // Process frame
        auto slam_frame = std::make_shared<ar_slam::Frame>(frame);
        auto result = tracker.track_features(slam_frame);

        // Calculate frame timing
        auto frame_end = std::chrono::high_resolution_clock::now();
        double frame_time = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
        double fps = 1000.0 / frame_time;

        // Update performance monitor
        monitor.add_frame(fps, result.tracking_quality, result.num_tracked);
        frame_count++;

        // Track low quality frames
        if (result.tracking_quality < 0.7) {
            low_quality_frames++;
        }

        // Convert 2D points to 3D with better depth estimation
        std::vector<cv::Point3f> points_3d;
        for (size_t i = 0; i < result.curr_points.size(); ++i) {
            const auto& pt = result.curr_points[i];

            // Normalize coordinates
            float x = (pt.x - frame.cols/2) / 200.0f;
            float y = -(pt.y - frame.rows/2) / 200.0f;

            // Depth estimation based on position and motion
            float base_depth = 2.0f;
            float depth_from_position = base_depth + (frame.rows - pt.y) / frame.rows * 1.5f;

            // Motion-based depth hint
            float depth_variation = 0.0f;
            if (i < result.prev_points.size()) {
                float motion = cv::norm(pt - result.prev_points[i]);
                depth_variation = (5.0f - std::min(motion, 5.0f)) * 0.05f;
            }

            float z = depth_from_position + depth_variation;
            points_3d.push_back(cv::Point3f(x, y, z));
        }

        // Update 3D viewer
        viewer.update_points(points_3d);

        // Update window title with real-time stats
        std::stringstream title;
        title << "AR SLAM 3D - FPS: " << std::fixed << std::setprecision(1) << monitor.avg_fps()
              << " | Points: " << result.num_tracked
              << " | Quality: " << std::setprecision(1) << (monitor.avg_quality() * 100) << "%";
        viewer.set_title(title.str());

        // Print periodic status updates
        if (frame_count % 30 == 0) {  // Every second at 30 FPS
            std::cout << "Frame " << frame_count
                      << " | FPS: " << std::fixed << std::setprecision(1) << monitor.avg_fps()
                      << " | Tracked: " << result.num_tracked
                      << " | Quality: " << std::setprecision(1) << (result.tracking_quality * 100) << "%"
                      << " | Avg: " << (monitor.avg_quality() * 100) << "%"
                      << std::endl;

            // Warn about quality drops
            if (result.tracking_quality < 0.7) {
                std::cout << "  ⚠ WARNING: Low tracking quality detected!" << std::endl;
            }
            if (result.tracking_quality < 0.5) {
                std::cout << "  ⚠ CRITICAL: Very poor tracking - consider resetting" << std::endl;
            }
        }

        // Render 3D view
        if (!viewer.render()) break;

        // Show 2D view with detailed overlays
        cv::Mat display = frame.clone();

        // Draw tracked points with motion vectors
        for (size_t i = 0; i < result.curr_points.size(); ++i) {
            // Color based on tracking quality
            cv::Scalar color;
            if (result.tracking_quality > 0.8) {
                color = cv::Scalar(0, 255, 0);  // Green - good
            } else if (result.tracking_quality > 0.6) {
                color = cv::Scalar(0, 255, 255);  // Yellow - ok
            } else {
                color = cv::Scalar(0, 0, 255);  // Red - poor
            }

            cv::circle(display, result.curr_points[i], 3, color, -1);

            // Draw motion vectors
            if (i < result.prev_points.size()) {
                cv::line(display, result.prev_points[i], result.curr_points[i],
                        cv::Scalar(0, 100, 0), 1);
            }
        }

        // Add text overlays
        int y_offset = 30;
        cv::putText(display, "Real-Time Metrics:",
                   cv::Point(10, y_offset), cv::FONT_HERSHEY_SIMPLEX,
                   0.7, cv::Scalar(255, 255, 255), 2);
        y_offset += 30;

        cv::putText(display, "FPS: " + std::to_string((int)monitor.avg_fps()),
                   cv::Point(10, y_offset), cv::FONT_HERSHEY_SIMPLEX,
                   0.6, cv::Scalar(0, 255, 0), 2);
        y_offset += 25;

        cv::putText(display, "Tracked: " + std::to_string(result.num_tracked),
                   cv::Point(10, y_offset), cv::FONT_HERSHEY_SIMPLEX,
                   0.6, cv::Scalar(0, 255, 0), 2);
        y_offset += 25;

        cv::putText(display, "Quality: " + std::to_string((int)(result.tracking_quality * 100)) + "%",
                   cv::Point(10, y_offset), cv::FONT_HERSHEY_SIMPLEX,
                   0.6, cv::Scalar(0, 255, 0), 2);
        y_offset += 25;

        cv::putText(display, "Avg Quality: " + std::to_string((int)(monitor.avg_quality() * 100)) + "%",
                   cv::Point(10, y_offset), cv::FONT_HERSHEY_SIMPLEX,
                   0.6, cv::Scalar(0, 255, 255), 2);

        // Quality indicator bar
        int bar_width = 200;
        int bar_height = 20;
        cv::rectangle(display, cv::Point(10, frame.rows - 40),
                     cv::Point(10 + bar_width, frame.rows - 40 + bar_height),
                     cv::Scalar(100, 100, 100), -1);

        int quality_width = (int)(result.tracking_quality * bar_width);
        cv::Scalar bar_color = result.tracking_quality > 0.7 ?
                              cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::rectangle(display, cv::Point(10, frame.rows - 40),
                     cv::Point(10 + quality_width, frame.rows - 40 + bar_height),
                     bar_color, -1);

        cv::imshow("2D Camera View", display);

        // Handle keyboard input
        char key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;  // Q or ESC
        if (key == ' ') {  // Space - print detailed stats
            auto total_time = std::chrono::high_resolution_clock::now() - start_time;
            double seconds = std::chrono::duration<double>(total_time).count();

            std::cout << "\n=== Detailed Statistics ===" << std::endl;
            std::cout << "Total frames: " << frame_count << std::endl;
            std::cout << "Runtime: " << std::fixed << std::setprecision(1) << seconds << " seconds" << std::endl;
            std::cout << "Average FPS: " << monitor.avg_fps() << std::endl;
            std::cout << "Average tracked: " << std::setprecision(0) << monitor.avg_tracked() << std::endl;
            std::cout << "Average quality: " << std::setprecision(1) << (monitor.avg_quality() * 100) << "%" << std::endl;
            std::cout << "Min quality: " << (monitor.min_quality() * 100) << "%" << std::endl;
            std::cout << "Max quality: " << (monitor.max_quality() * 100) << "%" << std::endl;
            std::cout << "Low quality frames: " << low_quality_frames
                      << " (" << (100.0 * low_quality_frames / frame_count) << "%)" << std::endl;
            std::cout << "===========================\n" << std::endl;
        }
        if (key == 'r' || key == 'R') {  // R - reset tracker
            tracker.reset();
            std::cout << "Tracker reset!" << std::endl;
        }
    }

    // Final statistics
    auto total_time = std::chrono::high_resolution_clock::now() - start_time;
    double seconds = std::chrono::duration<double>(total_time).count();

    std::cout << "\n=== Final Real-World Performance ===" << std::endl;
    std::cout << "Total frames: " << frame_count << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(2) << seconds << " seconds" << std::endl;
    std::cout << "Overall FPS: " << std::setprecision(1) << (frame_count / seconds) << std::endl;
    std::cout << "Average tracking quality: " << (monitor.avg_quality() * 100) << "%" << std::endl;
    std::cout << "Min/Max quality: " << (monitor.min_quality() * 100) << "% / "
              << (monitor.max_quality() * 100) << "%" << std::endl;
    std::cout << "Low quality frames (<70%): " << low_quality_frames
              << " (" << (100.0 * low_quality_frames / frame_count) << "%)" << std::endl;

    if (monitor.avg_quality() > 0.85) {
        std::cout << "\nExcellent tracking performance!" << std::endl;
    } else if (monitor.avg_quality() > 0.70) {
        std::cout << "\nGood tracking performance." << std::endl;
    } else {
        std::cout << "\nTracking quality could be improved." << std::endl;
        std::cout << "Consider better lighting or slower camera motion." << std::endl;
    }

    cap.release();
    cv::destroyAllWindows();

    std::cout << "\nShutting down..." << std::endl;
    return 0;
}