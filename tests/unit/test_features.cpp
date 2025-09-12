#include <iostream>
#include <opencv2/opencv.hpp>
#include "core/frame.h"
#include "core/feature_tracker.h"

bool test_feature_extraction() {
    std::cout << "Testing feature extraction..." << std::endl;
    
    // Create test image
    cv::Mat img(480, 640, CV_8UC3);
    cv::randu(img, 0, 255);
    
    // Add some rectangles for features
    for (int i = 0; i < 10; i++) {
        cv::rectangle(img, 
                     cv::Point(rand() % 600, rand() % 440),
                     cv::Point(rand() % 600 + 20, rand() % 440 + 20),
                     cv::Scalar(255, 255, 255), -1);
    }
    
    auto frame = std::make_shared<ar_slam::Frame>(img);
    frame->extract_features(500);
    
    bool passed = frame->get_features().size() > 0;
    std::cout << "  Extracted " << frame->get_features().size() << " features - "
              << (passed ? "PASSED" : "FAILED") << std::endl;
    
    return passed;
}

bool test_tracking() {
    std::cout << "Testing feature tracking..." << std::endl;
    
    cv::Mat img1(480, 640, CV_8UC3);
    cv::randu(img1, 0, 200);
    
    // Add trackable features
    for (int i = 0; i < 20; i++) {
        cv::circle(img1, cv::Point(rand() % 640, rand() % 480), 
                  10, cv::Scalar(255, 255, 255), -1);
    }
    
    cv::Mat img2;
    cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, 5, 0, 1, 3);
    cv::warpAffine(img1, img2, M, img1.size());
    
    ar_slam::FeatureTracker tracker;
    
    auto frame1 = std::make_shared<ar_slam::Frame>(img1);
    auto frame2 = std::make_shared<ar_slam::Frame>(img2);
    
    auto result1 = tracker.track_features(frame1);
    auto result2 = tracker.track_features(frame2);
    
    bool passed = result2.num_tracked > 0 && result2.tracking_quality > 0.5;
    std::cout << "  Tracked " << result2.num_tracked << " features with quality " 
              << result2.tracking_quality << " - "
              << (passed ? "PASSED" : "FAILED") << std::endl;
    
    return passed;
}

int main() {
    std::cout << "=== AR SLAM Feature Tests ===" << std::endl;
    
    int passed = 0;
    int total = 0;
    
    if (test_feature_extraction()) passed++;
    total++;
    
    if (test_tracking()) passed++;
    total++;
    
    std::cout << "\n=== Test Results ===" << std::endl;
    std::cout << "Passed: " << passed << "/" << total << std::endl;
    
    return (passed == total) ? 0 : 1;
}