#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "core/frame.h"
#include "core/feature_tracker.h"

int main() {
    std::cout << "=== Performance Stress Test ===" << std::endl;
    
    // System info
    std::cout << "Hardware threads: " << std::thread::hardware_concurrency() << std::endl;
    
    // Create high-resolution test image
    cv::Mat test_img(1080, 1920, CV_8UC3);
    cv::randu(test_img, 0, 255);
    
    // Test maximum sustainable FPS
    ar_slam::FeatureTracker tracker;
    auto start = std::chrono::high_resolution_clock::now();
    int frames = 0;
    
    while (frames < 1000) {
        auto frame = std::make_shared<ar_slam::Frame>(test_img);
        tracker.track_features(frame);
        frames++;
        
        if (frames % 100 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - start).count();
            double fps = frames / elapsed;
            std::cout << "Frames: " << frames << " | FPS: " << fps << std::endl;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end - start).count();
    
    std::cout << "\nResults:" << std::endl;
    std::cout << "Total frames: " << frames << std::endl;
    std::cout << "Total time: " << total_time << " seconds" << std::endl;
    std::cout << "Average FPS: " << frames / total_time << std::endl;
    
    return 0;
}