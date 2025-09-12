#pragma once
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <atomic>

namespace ar_slam {

    struct CameraConfig {
        int width = 640;
        int height = 480;
        int fps = 30;
        int device_id = 0;
        std::string device_path = "/dev/video0";
    };

    class CameraInterface {
    public:
        using Ptr = std::shared_ptr<CameraInterface>;

        virtual ~CameraInterface() = default;

        virtual bool open(const CameraConfig& config) = 0;
        virtual void close() = 0;
        virtual bool is_open() const = 0;

        virtual bool grab_frame(cv::Mat& frame) = 0;
        virtual double get_fps() const = 0;

    protected:
        CameraConfig config_;
        std::atomic<bool> is_running_{false};
    };

} // namespace ar_slam