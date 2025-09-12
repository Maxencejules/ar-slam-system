#include <iostream>
#include <opencv2/opencv.hpp>
#include "core/frame.h"
#include "core/feature_tracker.h"

int main() {
    std::cout << "=== Camera Test ===" << std::endl;
    
    // Try to open camera with OpenCV first (simpler)
    cv::VideoCapture cap(0);
    
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open camera" << std::endl;
        std::cerr << "Try: sudo apt-get install v4l-utils" << std::endl;
        std::cerr << "Check camera: ls /dev/video*" << std::endl;
        return -1;
    }
    
    std::cout << "Camera opened successfully!" << std::endl;
    std::cout << "Press 'q' to quit, 'space' to process frame" << std::endl;
    
    ar_slam::FeatureTracker tracker;
    cv::Mat frame;
    
    while (true) {
        cap >> frame;
        
        if (frame.empty()) {
            std::cerr << "Empty frame!" << std::endl;
            break;
        }
        
        // Convert to grayscale for display
        cv::Mat display = frame.clone();
        
        // Create Frame object
        auto slam_frame = std::make_shared<ar_slam::Frame>(frame);
        
        // Track features
        auto result = tracker.track_features(slam_frame);
        
        // Draw tracked points
        for (size_t i = 0; i < result.curr_points.size(); ++i) {
            cv::circle(display, result.curr_points[i], 3, cv::Scalar(0, 255, 0), -1);
            if (i < result.prev_points.size()) {
                cv::line(display, result.prev_points[i], result.curr_points[i], 
                        cv::Scalar(0, 0, 255), 1);
            }
        }
        
        // Add text overlay
        cv::putText(display, "Tracked: " + std::to_string(result.num_tracked), 
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 
                   1, cv::Scalar(0, 255, 0), 2);
        
        cv::putText(display, "Quality: " + std::to_string(result.tracking_quality), 
                   cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 
                   1, cv::Scalar(0, 255, 0), 2);
        
        // Show frame
        cv::imshow("AR SLAM - Camera Test", display);
        
        // Handle key press
        char key = cv::waitKey(1);
        if (key == 'q' || key == 27) {  // 'q' or ESC
            break;
        } else if (key == ' ') {  // Space
            std::cout << "Frame " << slam_frame->get_id() 
                     << ": " << result.num_tracked << " features tracked" << std::endl;
        }
    }
    
    cap.release();
    cv::destroyAllWindows();
    
    std::cout << "Camera test completed!" << std::endl;
    return 0;
}