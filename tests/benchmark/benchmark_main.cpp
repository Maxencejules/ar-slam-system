#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <iomanip>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "core/frame.h"
#include "core/feature_tracker.h"
#include "core/memory_pool.h"

using namespace std::chrono;

class BenchmarkTimer {
private:
    std::string name_;
    high_resolution_clock::time_point start_;
    std::vector<double>& results_;
    
public:
    BenchmarkTimer(const std::string& name, std::vector<double>& results) 
        : name_(name), results_(results) {
        start_ = high_resolution_clock::now();
    }
    
    ~BenchmarkTimer() {
        auto end = high_resolution_clock::now();
        double ms = duration<double, std::milli>(end - start_).count();
        results_.push_back(ms);
    }
};

void print_statistics(const std::string& name, const std::vector<double>& times) {
    if (times.empty()) return;
    
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / times.size();
    
    double sq_sum = 0;
    for (double t : times) {
        sq_sum += (t - mean) * (t - mean);
    }
    double stddev = std::sqrt(sq_sum / times.size());
    
    auto minmax = std::minmax_element(times.begin(), times.end());
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << name << ":" << std::endl;
    std::cout << "  Mean:   " << mean << " ms" << std::endl;
    std::cout << "  StdDev: " << stddev << " ms" << std::endl;
    std::cout << "  Min:    " << *minmax.first << " ms" << std::endl;
    std::cout << "  Max:    " << *minmax.second << " ms" << std::endl;
    std::cout << "  FPS:    " << 1000.0 / mean << std::endl;
    std::cout << std::endl;
}

void benchmark_feature_extraction() {
    std::cout << "=== Feature Extraction Benchmark ===" << std::endl;
    
    // Create test image
    cv::Mat test_img(480, 640, CV_8UC3);
    cv::randu(test_img, 0, 255);
    
    // Add some features
    for (int i = 0; i < 20; i++) {
        cv::rectangle(test_img, 
                     cv::Point(rand() % 600, rand() % 440),
                     cv::Point(rand() % 600 + 20, rand() % 440 + 20),
                     cv::Scalar(255, 255, 255), -1);
    }
    
    std::vector<int> feature_counts = {100, 500, 1000};
    
    for (int features : feature_counts) {
        std::vector<double> times;
        
        for (int i = 0; i < 50; i++) {
            auto frame = std::make_shared<ar_slam::Frame>(test_img);
            
            {
                BenchmarkTimer timer("extraction", times);
                frame->extract_features(features);
            }
        }
        
        std::cout << "Target features: " << features << std::endl;
        print_statistics("  Extraction time", times);
    }
}

void benchmark_tracking() {
    std::cout << "=== Feature Tracking Benchmark ===" << std::endl;
    
    cv::Mat img1(480, 640, CV_8UC3);
    cv::Mat img2(480, 640, CV_8UC3);
    
    // Create images with trackable features
    cv::randu(img1, 0, 100);
    for (int i = 0; i < 30; i++) {
        int x = rand() % 600;
        int y = rand() % 440;
        cv::rectangle(img1, cv::Point(x, y), cv::Point(x+20, y+20), 
                     cv::Scalar(255, 255, 255), -1);
    }
    
    // Slightly shifted version
    cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, 3, 0, 1, 2);
    cv::warpAffine(img1, img2, M, img2.size());
    
    ar_slam::FeatureTracker tracker;
    
    std::vector<double> init_times;
    std::vector<double> track_times;
    
    for (int i = 0; i < 50; i++) {
        tracker.reset();
        
        auto frame1 = std::make_shared<ar_slam::Frame>(img1);
        auto frame2 = std::make_shared<ar_slam::Frame>(img2);
        
        // Initial frame
        {
            BenchmarkTimer timer("init", init_times);
            tracker.track_features(frame1);
        }
        
        // Track to second frame
        {
            BenchmarkTimer timer("track", track_times);
            auto result = tracker.track_features(frame2);
        }
    }
    
    print_statistics("Initialization", init_times);
    print_statistics("Tracking", track_times);
}

void benchmark_memory() {
    std::cout << "=== Memory Pool Benchmark ===" << std::endl;
    
    struct TestObject {
        int id;
        char data[1024];  // 1KB object
        TestObject(int i) : id(i) {
            // Initialize data
            for (int j = 0; j < 1024; j++) {
                data[j] = static_cast<char>(i % 256);
            }
        }
    };
    
    ar_slam::MemoryPool<TestObject> pool(10 * 1024 * 1024);  // 10MB
    
    std::vector<double> alloc_times;
    std::vector<double> dealloc_times;
    std::vector<TestObject*> allocated;
    
    // Allocation benchmark
    for (int i = 0; i < 100; i++) {
        {
            BenchmarkTimer timer("alloc", alloc_times);
            TestObject* mem = pool.allocate();
            TestObject* obj = new (mem) TestObject(i);
            allocated.push_back(obj);
        }
    }
    
    // Deallocation benchmark
    for (auto* obj : allocated) {
        {
            BenchmarkTimer timer("dealloc", dealloc_times);
            obj->~TestObject();
            pool.deallocate(obj);
        }
    }
    
    print_statistics("Allocation", alloc_times);
    print_statistics("Deallocation", dealloc_times);
    
    std::cout << "Memory pool final usage: " << pool.get_usage() << " bytes" << std::endl;
}

int main() {
    std::cout << "=====================================" << std::endl;
    std::cout << "    AR SLAM System Benchmarks" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << std::endl;
    
    // Seed random number generator
    srand(static_cast<unsigned>(time(nullptr)));
    
    try {
        benchmark_feature_extraction();
        benchmark_tracking();
        benchmark_memory();
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "=====================================" << std::endl;
    std::cout << "        Benchmark Complete" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    return 0;
}