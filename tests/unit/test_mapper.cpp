#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include "core/incremental_mapper.h"
#include "synthetic_scene.h"
#include "test_util.h"

namespace {
    void test_ordered_pairs_and_loss() {
        const auto scene = synthetic::scene();
        ar_slam::IncrementalMapper::Config config;
        config.min_parallax_px = 3;
        ar_slam::IncrementalMapper mapper(scene.K, config);
        std::vector<int> ids(scene.world.size());
        std::iota(ids.begin(), ids.end(), 0);
        CHECK(!mapper.update(ids, scene.first));
        CHECK(mapper.reference_index() == 1);
        CHECK(!mapper.update(ids, scene.first));
        CHECK(!mapper.has_cloud());
        CHECK(mapper.last_parallax() == 0);
        // Observation order is independent of identity.
        std::reverse(ids.begin(), ids.end());
        auto current = scene.second;
        std::reverse(current.begin(), current.end());
        cv::setRNGSeed(2026);
        CHECK(mapper.update(ids, current));
        CHECK(mapper.has_cloud());
        CHECK(!mapper.cloud_is_stale());
        CHECK(mapper.cloud_reference_index() == 1);
        CHECK(mapper.cloud_current_index() == 3);
        CHECK(mapper.reference_index() == 3);
        CHECK(mapper.cloud().size() >= 220);
        CHECK(!mapper.update(ids, current));
        CHECK(mapper.cloud_is_stale());
        CHECK(mapper.cloud_reference_index() == 1);
        const auto rotation3 = synthetic::rotation_y(8);
        const cv::Vec3d translation3{-1.2, 0.01, 0.02};
        std::vector<cv::Point2f> third;
        for (int id : ids)
            third.push_back(synthetic::pixel(scene.K, rotation3, translation3,
                                             scene.world[static_cast<std::size_t>(id)]));
        cv::setRNGSeed(2026);
        CHECK(mapper.update(ids, third));
        CHECK(mapper.cloud_reference_index() == 3);
        CHECK(mapper.cloud_current_index() == 5);
        CHECK(mapper.reference_index() == 5);
        // The second cloud lives in camera 2, with its own baseline scale.
        const auto relative_rotation = rotation3 * scene.R.t();
        const auto relative_translation = translation3 - relative_rotation * scene.t;
        const double baseline = cv::norm(relative_translation);
        double max_error = 0;
        const auto& result = mapper.last_result();
        for (std::size_t i = 0; i < result.points.size(); ++i) {
            const auto id = ids[static_cast<std::size_t>(result.point_indices[i])];
            const auto& world = scene.world[static_cast<std::size_t>(id)];
            const auto truth = scene.R * cv::Vec3d(world.x, world.y, world.z) + scene.t;
            const auto& point = result.points[i];
            max_error = std::max(max_error,
                                 cv::norm(cv::Vec3d(point.x, point.y, point.z) * baseline - truth));
        }
        CHECK(max_error < 0.005);
        for (auto& id : ids)
            id += 1000;
        CHECK(!mapper.update(ids, third));
        CHECK(!mapper.has_cloud());
        CHECK(mapper.cloud().empty());
        CHECK(!mapper.last_result().success);
        mapper.reset();
        CHECK(mapper.reference_index() == 0);
        CHECK(!mapper.has_cloud());
        CHECK(!mapper.update({}, {}));
    }
    void test_invalid_and_rotation() {
        const auto scene = synthetic::scene();
        ar_slam::IncrementalMapper::Config config;
        config.min_parallax_px = 3;
        config.force_keyframe_px = 20;
        ar_slam::IncrementalMapper mapper(scene.K, config);
        std::vector<int> ids(scene.world.size());
        std::iota(ids.begin(), ids.end(), 0);
        mapper.update(ids, scene.first);
        CHECK(artest::throws<std::invalid_argument>([&] { mapper.update(ids, {}); }));
        auto duplicate = ids;
        duplicate[1] = duplicate[0];
        CHECK(
            artest::throws<std::invalid_argument>([&] { mapper.update(duplicate, scene.second); }));
        auto invalid = scene.second;
        invalid[1].y = std::numeric_limits<float>::quiet_NaN();
        CHECK(artest::throws<std::invalid_argument>([&] { mapper.update(ids, invalid); }));
        CHECK(mapper.reference_index() == 1);
        std::vector<cv::Point2f> rotated;
        for (const auto& point : scene.world)
            rotated.push_back(
                synthetic::pixel(scene.K, synthetic::rotation_y(12), {0, 0, 0}, point));
        CHECK(!mapper.update(ids, rotated));
        CHECK(!mapper.has_cloud());
        CHECK(mapper.last_parallax() > 20);
        CHECK(mapper.reference_index() == 2);  // Wide rotation advances the reference only.
        for (int mode = 0; mode < 4; ++mode) {
            auto bad = config;
            if (mode == 0)
                bad.min_shared_to_keep = 0;
            if (mode == 1)
                bad.min_correspondences = 4;
            if (mode == 2)
                bad.min_parallax_px = INFINITY;
            if (mode == 3)
                bad.force_keyframe_px = 1;
            CHECK(artest::throws<std::invalid_argument>(
                [&] { ar_slam::IncrementalMapper invalid_mapper(scene.K, bad); }));
        }
    }
}  // namespace
int main() {
    cv::setNumThreads(1);
    test_ordered_pairs_and_loss();
    test_invalid_and_rotation();
    return artest::report("test_mapper");
}
