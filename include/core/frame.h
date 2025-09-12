#pragma once
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <chrono>
#include <iostream>

namespace ar_slam {

    struct Feature {
        cv::Point2f pixel;           // 2D pixel coordinates
        cv::Point2f undistorted;     // Undistorted coordinates
        cv::Mat descriptor;          // Feature descriptor
        float response;              // Feature response strength
        int octave;                  // Scale space octave
        bool is_outlier = false;

        Feature() = default;
        Feature(const cv::KeyPoint& kp, const cv::Mat& desc);
    };

    class Frame {
    public:
        using Ptr = std::shared_ptr<Frame>;
        using Timestamp = std::chrono::steady_clock::time_point;

        // Public members for simplicity (in production, use getters)
        std::vector<cv::KeyPoint> keypoints_;
        cv::Mat descriptors_;

    private:
        static uint64_t next_id_;

        uint64_t id_;
        Timestamp timestamp_;
        cv::Mat image_gray_;
        cv::Mat image_rgb_;

        // Camera parameters
        cv::Mat K_;                  // Intrinsic matrix
        cv::Mat dist_coeffs_;        // Distortion coefficients

        // Features
        std::vector<Feature> features_;

        // Performance metrics
        double extraction_time_ms_ = 0;

    public:
        explicit Frame(const cv::Mat& image,
                      const Timestamp& timestamp = std::chrono::steady_clock::now());

        // Getters
        uint64_t get_id() const { return id_; }
        const cv::Mat& get_image() const { return image_gray_; }
        const std::vector<Feature>& get_features() const { return features_; }

        // Feature extraction
        void extract_features(int max_features = 1000);

        // Memory info
        size_t get_memory_usage() const;
    };

} // namespace ar_slam