#pragma once
#include "camera/camera_interface.h"
#include <linux/videodev2.h>
#include <vector>

namespace ar_slam {

    class V4L2Camera : public CameraInterface {
    private:
        int fd_ = -1;

        struct Buffer {
            void* start;
            size_t length;
        };

        std::vector<Buffer> buffers_;
        v4l2_format format_;

        // Performance tracking
        std::chrono::steady_clock::time_point last_frame_time_;
        double measured_fps_ = 0;

    public:
        V4L2Camera() = default;
        ~V4L2Camera() override;

        bool open(const CameraConfig& config) override;
        void close() override;
        bool is_open() const override { return fd_ >= 0; }

        bool grab_frame(cv::Mat& frame) override;
        double get_fps() const override { return measured_fps_; }

    private:
        bool init_device();
        bool init_mmap();
        bool start_capture();
        bool stop_capture();
        void uninit_device();

        bool xioctl(int request, void* arg);
    };

} // namespace ar_slam