// Test with actual video files or live camera
#include <opencv2/opencv.hpp>
#include "core/feature_tracker.h"

void test_handheld_motion() {
    cv::VideoCapture cap(0);  // Or load a test video
    ar_slam::FeatureTracker tracker;

    std::vector<float> qualities;
    int total_losses = 0;

    for (int i = 0; i < 300; i++) {  // 10 seconds at 30fps
        cv::Mat frame;
        cap >> frame;

        auto result = tracker.track_features(frame);
        qualities.push_back(result.tracking_quality);

        if (result.tracking_quality < 0.3) {
            total_losses++;
        }
    }

    // Report REAL metrics
    float avg_quality = std::accumulate(qualities.begin(), qualities.end(), 0.0f) / qualities.size();
    std::cout << "Real handheld: " << avg_quality * 100 << "% average retention" << std::endl;
    std::cout << "Total tracking losses: " << total_losses << std::endl;
}