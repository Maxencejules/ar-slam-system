// Headless tests for feature extraction and optical-flow tracking.
// Uses synthetic, deterministic images instead of a live camera so the suite
// runs unattended in CI.

#include <opencv2/opencv.hpp>

#include "core/feature_tracker.h"
#include "core/frame.h"
#include "test_util.h"

namespace {

// A richly textured synthetic image so ORB has plenty of corners to find.
cv::Mat make_textured_image(int seed) {
    cv::RNG rng(seed);
    cv::Mat img(480, 640, CV_8UC3);
    rng.fill(img, cv::RNG::UNIFORM, 40, 120);
    for (int i = 0; i < 200; ++i) {
        cv::Point p(rng.uniform(10, 630), rng.uniform(10, 470));
        cv::Scalar color(rng.uniform(150, 255), rng.uniform(150, 255), rng.uniform(150, 255));
        if (rng.uniform(0, 2)) {
            cv::rectangle(img, p, p + cv::Point(rng.uniform(8, 30), rng.uniform(8, 30)), color, -1);
        } else {
            cv::circle(img, p, rng.uniform(4, 14), color, -1);
        }
    }
    return img;
}

void test_feature_extraction() {
    cv::Mat img = make_textured_image(7);
    auto frame = std::make_shared<ar_slam::Frame>(img);
    frame->extract_features(500);
    // A textured 640x480 image should yield a healthy number of features.
    CHECK(frame->get_features().size() > 200);
    CHECK(frame->get_features().size() <= 500);
    CHECK(frame->get_memory_usage() > 0);
}

void test_tracking_small_motion() {
    cv::Mat img1 = make_textured_image(11);

    // Translate by a few pixels: optical flow should keep almost everything.
    cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, 4, 0, 1, 3);
    cv::Mat img2;
    cv::warpAffine(img1, img2, M, img1.size());

    ar_slam::FeatureTracker tracker;
    auto f1 = std::make_shared<ar_slam::Frame>(img1);
    auto r1 = tracker.track_features(f1);
    CHECK(r1.num_tracked > 100);  // Initialisation seeds the track set.

    auto f2 = std::make_shared<ar_slam::Frame>(img2);
    auto r2 = tracker.track_features(f2);
    CHECK(r2.tracking_quality > 0.5f);
    CHECK(r2.num_tracked > 100);
    CHECK(r2.curr_points.size() == r2.track_ids.size());
}

void test_reset() {
    cv::Mat img = make_textured_image(3);
    ar_slam::FeatureTracker tracker;
    tracker.track_features(std::make_shared<ar_slam::Frame>(img));
    tracker.reset();
    // After reset, the next frame re-initialises as if it were the first.
    auto r = tracker.track_features(std::make_shared<ar_slam::Frame>(img));
    CHECK(r.tracking_quality == 1.0f);
    CHECK(r.num_tracked > 100);
}

}  // namespace

int main() {
    test_feature_extraction();
    test_tracking_small_motion();
    test_reset();
    return artest::report("test_tracking");
}
