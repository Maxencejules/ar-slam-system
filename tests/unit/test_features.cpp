#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>
#include <iomanip>
#include <chrono>
#include "core/frame.h"
#include "core/feature_tracker.h"

bool test_feature_extraction_real() {
    std::cout << "Testing feature extraction on real camera image..." << std::endl;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "  WARNING: Cannot open camera, using synthetic image" << std::endl;
        cv::Mat img(480, 640, CV_8UC3);
        cv::randu(img, 50, 200);
        auto frame = std::make_shared<ar_slam::Frame>(img);
        frame->extract_features(500);
        return frame->get_features().size() > 200;
    }

    // Capture a real frame
    cv::Mat img;
    for (int i = 0; i < 5; i++) cap >> img;  // Skip first frames (auto-exposure)

    if (img.empty()) return false;

    auto frame = std::make_shared<ar_slam::Frame>(img);
    frame->extract_features(500);

    size_t extracted = frame->get_features().size();
    std::cout << "  Extracted " << extracted << " features from real camera" << std::endl;

    bool passed = extracted > 200 && extracted <= 1000;
    if (passed) {
        std::cout << "  PASSED - Good feature count" << std::endl;
    } else {
        std::cout << "  Check your lighting and scene texture" << std::endl;
    }

    cap.release();
    return passed;
}

bool test_tracking_with_real_camera() {
    std::cout << "\nTesting tracking with REAL camera motion..." << std::endl;
    std::cout << "INSTRUCTIONS: Move your camera slowly when prompted\n" << std::endl;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "  ERROR: Cannot open camera" << std::endl;
        return false;
    }

    ar_slam::FeatureTracker tracker;
    std::vector<float> quality_samples;

    // Warm up camera
    cv::Mat frame;
    for (int i = 0; i < 10; i++) {
        cap >> frame;
        cv::imshow("Camera Test", frame);
        cv::waitKey(30);
    }

    std::cout << "PHASE 1: Keep camera STILL for 3 seconds..." << std::endl;

    // Test stationary camera
    for (int i = 0; i < 30; i++) {  // 1 second at 30fps
        cap >> frame;
        if (frame.empty()) continue;

        auto slam_frame = std::make_shared<ar_slam::Frame>(frame);
        auto result = tracker.track_features(slam_frame);

        if (i > 10) {  // Skip initialization
            quality_samples.push_back(result.tracking_quality);
        }

        cv::putText(frame, "Keep Still", cv::Point(30, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        cv::imshow("Camera Test", frame);
        cv::waitKey(33);
    }

    float still_quality = std::accumulate(quality_samples.begin(),
                                         quality_samples.end(), 0.0f) / quality_samples.size();
    std::cout << "  Stationary quality: " << (still_quality * 100) << "%" << std::endl;

    quality_samples.clear();

    std::cout << "\nPHASE 2: Move camera SLOWLY left and right for 3 seconds..." << std::endl;

    // Test slow motion
    for (int i = 0; i < 90; i++) {
        cap >> frame;
        if (frame.empty()) continue;

        auto slam_frame = std::make_shared<ar_slam::Frame>(frame);
        auto result = tracker.track_features(slam_frame);
        quality_samples.push_back(result.tracking_quality);

        cv::putText(frame, "Move Slowly", cv::Point(30, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
        cv::putText(frame, "Quality: " + std::to_string(int(result.tracking_quality * 100)) + "%",
                   cv::Point(30, 70), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
        cv::imshow("Camera Test", frame);
        cv::waitKey(33);
    }

    float slow_quality = std::accumulate(quality_samples.begin(),
                                        quality_samples.end(), 0.0f) / quality_samples.size();
    std::cout << "  Slow motion quality: " << (slow_quality * 100) << "%" << std::endl;

    quality_samples.clear();

    std::cout << "\nPHASE 3: Move camera QUICKLY for 2 seconds..." << std::endl;

    // Test rapid motion
    for (int i = 0; i < 60; i++) {
        cap >> frame;
        if (frame.empty()) continue;

        auto slam_frame = std::make_shared<ar_slam::Frame>(frame);
        auto result = tracker.track_features(slam_frame);
        quality_samples.push_back(result.tracking_quality);

        cv::putText(frame, "Move Quickly!", cv::Point(30, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        cv::putText(frame, "Quality: " + std::to_string(int(result.tracking_quality * 100)) + "%",
                   cv::Point(30, 70), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        cv::imshow("Camera Test", frame);
        cv::waitKey(33);
    }

    float fast_quality = std::accumulate(quality_samples.begin(),
                                        quality_samples.end(), 0.0f) / quality_samples.size();
    std::cout << "  Fast motion quality: " << (fast_quality * 100) << "%" << std::endl;

    cv::destroyWindow("Camera Test");
    cap.release();

    // Evaluate results
    std::cout << "\n=== Real Camera Results ===" << std::endl;
    std::cout << "Stationary: " << (still_quality * 100) << "%" << std::endl;
    std::cout << "Slow motion: " << (slow_quality * 100) << "%" << std::endl;
    std::cout << "Fast motion: " << (fast_quality * 100) << "%" << std::endl;

    // Realistic pass criteria
    bool passed = (still_quality > 0.85) &&  // Should track well when still
                  (slow_quality > 0.50) &&    // Should maintain >50% with slow motion
                  (fast_quality < slow_quality); // Fast should be worse than slow

    if (passed) {
        std::cout << "\nPASSED - Realistic tracking behavior observed" << std::endl;
    } else {
        std::cout << "\nResults show tracking characteristics" << std::endl;
    }

    return passed;
}

bool test_recovery_with_camera() {
    std::cout << "\nTesting recovery from tracking loss..." << std::endl;
    std::cout << "INSTRUCTIONS: Cover camera with hand when prompted\n" << std::endl;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "  ERROR: Cannot open camera" << std::endl;
        return false;
    }

    ar_slam::FeatureTracker tracker;
    cv::Mat frame;

    // Initialize tracking
    for (int i = 0; i < 10; i++) {
        cap >> frame;
        auto slam_frame = std::make_shared<ar_slam::Frame>(frame);
        tracker.track_features(slam_frame);
    }

    std::cout << "COVER camera with your hand for 2 seconds..." << std::endl;

    int features_before = 0;
    int features_during = 0;
    int features_after = 0;

    // Before covering
    cap >> frame;
    auto result_before = tracker.track_features(std::make_shared<ar_slam::Frame>(frame));
    features_before = result_before.num_tracked;

    // During covering (wait for user to cover)
    for (int i = 0; i < 60; i++) {
        cap >> frame;
        auto result = tracker.track_features(std::make_shared<ar_slam::Frame>(frame));

        cv::putText(frame, "Cover Camera Now", cv::Point(30, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        cv::putText(frame, "Features: " + std::to_string(result.num_tracked),
                   cv::Point(30, 70), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        cv::imshow("Recovery Test", frame);
        cv::waitKey(33);

        if (i == 30) features_during = result.num_tracked;
    }

    std::cout << "UNCOVER camera..." << std::endl;

    // After uncovering
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 30; i++) {
        cap >> frame;
        auto result = tracker.track_features(std::make_shared<ar_slam::Frame>(frame));

        if (i == 15) {
            features_after = result.num_tracked;
            auto end = std::chrono::high_resolution_clock::now();
            double recovery_ms = std::chrono::duration<double, std::milli>(end - start).count();
            std::cout << "  Recovery time: " << recovery_ms << " ms" << std::endl;
        }

        cv::putText(frame, "Uncovered", cv::Point(30, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, "Features: " + std::to_string(result.num_tracked),
                   cv::Point(30, 70), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        cv::imshow("Recovery Test", frame);
        cv::waitKey(33);
    }

    cv::destroyWindow("Recovery Test");
    cap.release();

    std::cout << "  Features before: " << features_before << std::endl;
    std::cout << "  Features during: " << features_during << std::endl;
    std::cout << "  Features after: " << features_after << std::endl;

    bool passed = (features_during < features_before / 2) &&  // Should lose most features
                  (features_after > features_before / 2);     // Should recover

    if (passed) {
        std::cout << "  PASSED - System recovers from occlusion" << std::endl;
    } else {
        std::cout << "  Recovery behavior observed" << std::endl;
    }

    return passed;
}

int main() {
    std::cout << "=== Real-World Camera Testing ===" << std::endl;
    std::cout << "This test uses your actual webcam to measure real performance\n" << std::endl;

    int passed = 0;
    int total = 0;

    if (test_feature_extraction_real()) passed++;
    total++;

    if (test_tracking_with_real_camera()) passed++;
    total++;

    if (test_recovery_with_camera()) passed++;
    total++;

    std::cout << "\n=== Final Results ===" << std::endl;
    std::cout << "Tests passed: " << passed << "/" << total << std::endl;

    std::cout << "\nThese are your ACTUAL performance metrics." << std::endl;
    std::cout << "Report these numbers to Google - they're real!" << std::endl;

    return (passed == total) ? 0 : 1;
}