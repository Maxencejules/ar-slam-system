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

cv::Mat create_realistic_test_image(int complexity_level) {
    cv::Mat img(480, 640, CV_8UC3);

    // Base noise
    cv::randu(img, 50, 150);

    // Add features based on complexity
    int num_features = complexity_level * 10;
    for (int i = 0; i < num_features; i++) {
        int x = rand() % 600 + 20;
        int y = rand() % 440 + 20;

        if (rand() % 2) {
            // Rectangle
            cv::rectangle(img, cv::Point(x, y),
                         cv::Point(x + rand() % 30 + 10, y + rand() % 30 + 10),
                         cv::Scalar(rand() % 100 + 155, rand() % 100 + 155, rand() % 100 + 155),
                         -1);
        } else {
            // Circle
            cv::circle(img, cv::Point(x, y), rand() % 15 + 5,
                      cv::Scalar(rand() % 100 + 155, rand() % 100 + 155, rand() % 100 + 155),
                      -1);
        }
    }

    // Add realistic noise
    cv::Mat noise(img.size(), CV_8UC3);
    cv::randn(noise, 0, 10);
    img += noise;

    // Slight blur (lens imperfection)
    if (complexity_level > 1) {
        cv::GaussianBlur(img, img, cv::Size(3, 3), 0.5);
    }

    return img;
}

void benchmark_feature_extraction() {
    std::cout << "=== Feature Extraction Benchmark (Realistic) ===" << std::endl;

    std::vector<int> feature_counts = {100, 500, 1000};

    for (int features : feature_counts) {
        std::vector<double> times;

        for (int complexity = 1; complexity <= 3; complexity++) {
            cv::Mat test_img = create_realistic_test_image(complexity);

            for (int i = 0; i < 20; i++) {
                // Add frame-to-frame variations
                cv::Mat varied_img = test_img.clone();
                varied_img.convertTo(varied_img, -1,
                                    0.95 + (rand() % 10) / 100.0,  // 0.95-1.05 contrast
                                    -5 + rand() % 10);              // -5 to +5 brightness

                auto frame = std::make_shared<ar_slam::Frame>(varied_img);

                {
                    BenchmarkTimer timer("extraction", times);
                    frame->extract_features(features);
                }
            }
        }

        std::cout << "Target features: " << features << std::endl;
        print_statistics("  Extraction time", times);
    }
}

void benchmark_tracking() {
    std::cout << "=== Feature Tracking Benchmark (Realistic Motion) ===" << std::endl;

    cv::Mat img1 = create_realistic_test_image(3);

    std::vector<double> init_times;
    std::vector<double> track_times;
    std::vector<double> quality_values;

    for (int i = 0; i < 50; i++) {
        ar_slam::FeatureTracker tracker;

        // Simulate realistic camera motion
        cv::Mat img2;
        cv::Point2f center(320, 240);

        // Random small rotation (-3 to +3 degrees)
        double angle = -3.0 + (rand() % 60) / 10.0;

        // Random small scale (0.98 to 1.02)
        double scale = 0.98 + (rand() % 40) / 1000.0;

        // Random translation
        cv::Mat M = cv::getRotationMatrix2D(center, angle, scale);
        M.at<double>(0, 2) += -10.0 + rand() % 20;  // -10 to +10 pixels
        M.at<double>(1, 2) += -10.0 + rand() % 20;

        cv::warpAffine(img1, img2, M, img1.size());

        // Add realistic frame-to-frame changes
        cv::Mat blur_kernel = cv::Mat::ones(3, 3, CV_32F) / 9.0f;
        cv::filter2D(img2, img2, -1, blur_kernel);

        // Lighting variation
        img2.convertTo(img2, -1, 0.9 + (rand() % 20) / 100.0, -10 + rand() % 20);

        // Noise
        cv::Mat noise(img2.size(), CV_8UC3);
        cv::randn(noise, 0, 8);
        img2 += noise;

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
            quality_values.push_back(result.tracking_quality);
        }
    }

    print_statistics("Initialization", init_times);
    print_statistics("Tracking", track_times);

    // Print tracking quality statistics
    double avg_quality = std::accumulate(quality_values.begin(), quality_values.end(), 0.0)
                        / quality_values.size();
    auto minmax_quality = std::minmax_element(quality_values.begin(), quality_values.end());

    std::cout << "Tracking Quality Statistics:" << std::endl;
    std::cout << "  Average: " << (avg_quality * 100) << "%" << std::endl;
    std::cout << "  Min:     " << (*minmax_quality.first * 100) << "%" << std::endl;
    std::cout << "  Max:     " << (*minmax_quality.second * 100) << "%" << std::endl;
    std::cout << "  Note: Realistic tracking quality is typically 60-85%" << std::endl;
    std::cout << std::endl;
}

void benchmark_memory() {
    std::cout << "=== Memory Pool Benchmark ===" << std::endl;

    struct TestObject {
        int id;
        char data[1024];  // 1KB object
        TestObject(int i) : id(i) {
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

void benchmark_full_pipeline() {
    std::cout << "=== Full Pipeline Benchmark (Realistic Conditions) ===" << std::endl;

    ar_slam::FeatureTracker tracker;
    std::vector<double> pipeline_times;
    std::vector<double> tracking_qualities;

    // Create initial frame
    cv::Mat prev_img = create_realistic_test_image(3);
    auto prev_frame = std::make_shared<ar_slam::Frame>(prev_img);
    tracker.track_features(prev_frame);

    std::cout << "Running 100 frame benchmark with realistic motion..." << std::endl;

    for (int frame_num = 0; frame_num < 100; frame_num++) {
        // Simulate continuous camera motion
        cv::Mat curr_img;
        cv::Point2f center(320, 240);

        // Accumulating small motions (more realistic than random jumps)
        double angle = sin(frame_num * 0.1) * 2.0;  // Smooth rotation
        double scale = 1.0 + sin(frame_num * 0.05) * 0.02;  // Smooth scale

        cv::Mat M = cv::getRotationMatrix2D(center, angle, scale);
        M.at<double>(0, 2) += sin(frame_num * 0.15) * 5;  // Smooth translation
        M.at<double>(1, 2) += cos(frame_num * 0.15) * 5;

        cv::warpAffine(prev_img, curr_img, M, prev_img.size());

        // Frame-to-frame degradation
        cv::GaussianBlur(curr_img, curr_img, cv::Size(3, 3), 0.3);
        curr_img.convertTo(curr_img, -1, 0.98, 2);

        cv::Mat noise(curr_img.size(), CV_8UC3);
        cv::randn(noise, 0, 5);
        curr_img += noise;

        auto pipeline_start = high_resolution_clock::now();

        auto curr_frame = std::make_shared<ar_slam::Frame>(curr_img);
        auto result = tracker.track_features(curr_frame);

        auto pipeline_end = high_resolution_clock::now();

        double time_ms = duration<double, std::milli>(pipeline_end - pipeline_start).count();
        pipeline_times.push_back(time_ms);
        tracking_qualities.push_back(result.tracking_quality);

        prev_img = curr_img;

        if ((frame_num + 1) % 25 == 0) {
            std::cout << "  Processed " << (frame_num + 1) << " frames... "
                      << "Avg quality: "
                      << (std::accumulate(tracking_qualities.begin(),
                                         tracking_qualities.end(), 0.0)
                         / tracking_qualities.size() * 100) << "%" << std::endl;
        }
    }

    print_statistics("Full Pipeline", pipeline_times);

    double avg_quality = std::accumulate(tracking_qualities.begin(),
                                        tracking_qualities.end(), 0.0) / tracking_qualities.size();
    std::cout << "Average tracking quality over 100 frames: "
              << (avg_quality * 100) << "%" << std::endl;
    std::cout << "Note: 70-80% is excellent for continuous tracking" << std::endl;
}

int main() {
    std::cout << "=====================================" << std::endl;
    std::cout << "    AR SLAM System Benchmarks" << std::endl;
    std::cout << "    (Realistic Test Conditions)" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << std::endl;

    // Seed random number generator
    srand(static_cast<unsigned>(time(nullptr)));

    try {
        benchmark_feature_extraction();
        benchmark_tracking();
        benchmark_memory();
        benchmark_full_pipeline();
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "=====================================" << std::endl;
    std::cout << "        Benchmark Complete" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << "\nNote: These benchmarks use realistic conditions including:" << std::endl;
    std::cout << "- Gaussian noise and motion blur" << std::endl;
    std::cout << "- Lighting variations" << std::endl;
    std::cout << "- Rotation and scale changes" << std::endl;
    std::cout << "- Continuous motion simulation" << std::endl;
    
    return 0;
}