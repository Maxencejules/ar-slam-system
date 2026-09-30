#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include "core/feature_tracker.h"
#include "synthetic_scene.h"
#include "test_util.h"

namespace {
    using Frame = ar_slam::Frame;
    Frame::Ptr frame(const cv::Mat& image, int index) {
        return std::make_shared<Frame>(image,
                                       Frame::Timestamp{} + std::chrono::milliseconds(index));
    }
    void aligned(const ar_slam::TrackingResult& result) {
        const auto count = result.curr_points.size();
        CHECK(count == result.prev_points.size());
        CHECK(count == result.track_ids.size());
        CHECK(count == result.inliers.size());
        CHECK(result.num_tracked == static_cast<int>(count));
        CHECK(result.num_inliers ==
              static_cast<int>(std::count(result.inliers.begin(), result.inliers.end(), true)));
        CHECK(result.num_tracked == result.num_inliers + result.num_new);
        CHECK(std::unordered_set<int>(result.track_ids.begin(), result.track_ids.end()).size() ==
              count);
    }
    void test_translation_and_replenishment() {
        const auto image = synthetic::texture(11);
        ar_slam::FeatureTracker tracker;
        const auto first = tracker.track_features(frame(image, 1));
        aligned(first);
        CHECK(first.num_tracked > 200);
        CHECK(first.num_inliers == 0);
        CHECK(first.tracking_quality == 0);
        const auto second = tracker.track_features(frame(synthetic::translated(image, 4, 3), 2));
        aligned(second);
        CHECK(second.tracking_quality > 0.5);
        CHECK(second.num_inliers > 150);
        std::unordered_map<int, cv::Point2f> original;
        for (std::size_t i = 0; i < first.curr_points.size(); ++i)
            original.emplace(first.track_ids[i], first.curr_points[i]);
        std::vector<double> errors;
        for (std::size_t i = 0; i < second.curr_points.size(); ++i) {
            if (!second.inliers[i]) {
                CHECK(original.count(second.track_ids[i]) == 0);
                CHECK(second.prev_points[i] == second.curr_points[i]);
                continue;
            }
            CHECK(original.at(second.track_ids[i]) == second.prev_points[i]);
            errors.push_back(std::hypot(second.curr_points[i].x - second.prev_points[i].x - 4,
                                        second.curr_points[i].y - second.prev_points[i].y - 3));
        }
        std::sort(errors.begin(), errors.end());
        CHECK(!errors.empty());
        if (!errors.empty())
            CHECK(errors[errors.size() / 2] < 0.2);
        CHECK_NEAR(second.tracking_quality,
                   static_cast<double>(second.num_inliers) / first.num_tracked, 1e-6);
        // Replace most of the image with unrelated texture: replenish new IDs,
        // but preserve the low measured retention instead of reporting quality=1.
        auto changed = synthetic::translated(image, 8, 6);
        auto unrelated = synthetic::texture(77);
        unrelated(cv::Rect(160, 0, 480, 480)).copyTo(changed(cv::Rect(160, 0, 480, 480)));
        const auto third = tracker.track_features(frame(changed, 3));
        aligned(third);
        CHECK(third.num_new > 0);
        CHECK(third.tracking_quality < 0.5);
        CHECK_NEAR(third.tracking_quality,
                   static_cast<double>(third.num_inliers) / second.num_tracked, 1e-6);
        tracker.reset();
        const auto restarted = tracker.track_features(frame(image, 4));
        aligned(restarted);
        CHECK(*std::min_element(restarted.track_ids.begin(), restarted.track_ids.end()) >
              *std::max_element(third.track_ids.begin(), third.track_ids.end()));
        CHECK(restarted.tracking_quality == 0);
    }
    void test_input_contracts() {
        const auto image = synthetic::texture(3);
        ar_slam::FeatureTracker tracker;
        CHECK(artest::throws<std::invalid_argument>([&] { tracker.track_features(nullptr); }));
        tracker.track_features(frame(image, 10));
        for (int timestamp : {9, 10}) {
            CHECK(artest::throws<std::invalid_argument>(
                [&] { tracker.track_features(frame(image, timestamp)); }));
        }
        CHECK(artest::throws<std::invalid_argument>(
            [&] { tracker.track_features(frame(cv::Mat(200, 300, CV_8UC1, cv::Scalar(0)), 11)); }));
        const auto valid = tracker.track_features(frame(image, 11));
        CHECK(valid.num_inliers > 200);  // Bad calls did not advance timestamp/image state.
        for (const auto& invalid :
             {cv::Mat{}, cv::Mat(10, 10, CV_16UC1), cv::Mat(10, 10, CV_8UC2)}) {
            CHECK(artest::throws<std::invalid_argument>([&] { Frame bad(invalid); }));
        }
        Frame bgra(cv::Mat(10, 10, CV_8UC4, cv::Scalar(20, 30, 40, 255)));
        CHECK(bgra.get_image().type() == CV_8UC1);
        CHECK(artest::throws<std::invalid_argument>([&] { bgra.extract_features(0); }));
        cv::Mat owned(10, 10, CV_8UC1, cv::Scalar(77));
        Frame copied(owned);
        owned.setTo(0);
        CHECK(copied.get_image().at<uchar>(0, 0) == 77);
    }
}  // namespace
int main() {
    cv::setNumThreads(1);
    cv::setRNGSeed(2026);
    test_translation_and_replenishment();
    test_input_contracts();
    return artest::report("test_tracking");
}
