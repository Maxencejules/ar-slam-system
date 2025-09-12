#include <iostream>
#include "core/memory_pool.h"
#include "core/frame.h"

int main() {
    std::cout << "=== Memory Pool Test ===" << std::endl;
    
    const size_t POOL_SIZE = 256 * 1024 * 1024;  // 256MB
    ar_slam::MemoryPool<ar_slam::Frame> pool(POOL_SIZE);
    
    cv::Mat dummy(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<ar_slam::Frame*> frames;

    // Test allocation until limit
    std::cout << "Allocating frames until memory limit..." << std::endl;

    int max_frames = 100;  // Limit for testing

    for (int i = 0; i < max_frames; ++i) {
        // Allocate memory
        ar_slam::Frame* frame_mem = pool.allocate();

        // Construct object using placement new
        ar_slam::Frame* frame = new (frame_mem) ar_slam::Frame(dummy);

        frames.push_back(frame);

        if (frames.size() % 10 == 0) {
            std::cout << "Allocated " << frames.size() << " frames, "
                      << "Memory usage: " << pool.get_usage() / (1024*1024) << " MB" << std::endl;
        }
    }

    std::cout << "Successfully allocated " << frames.size() << " frames" << std::endl;

    // Clean up
    for (auto* frame : frames) {
        frame->~Frame();  // Call destructor explicitly
        pool.deallocate(frame);
    }
    
    std::cout << "Final memory usage: " << pool.get_usage() << " bytes" << std::endl;
    std::cout << "Test passed!" << std::endl;
    
    return 0;
}