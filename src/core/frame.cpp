#include "core/frame.h"
#include <opencv2/features2d.hpp>

namespace ar_slam {

uint64_t Frame::next_id_ = 0;

Feature::Feature(const cv::KeyPoint& kp, const cv::Mat& desc) 
    : pixel(kp.pt)
    , descriptor(desc.clone())
    , response(kp.response)
    , octave(kp.octave) {
}

Frame::Frame(const cv::Mat& image, const Timestamp& timestamp)
    : id_(next_id_++)
    , timestamp_(timestamp) {
    
    if (image.channels() == 3) {
        cv::cvtColor(image, image_gray_, cv::COLOR_BGR2GRAY);
        image_rgb_ = image.clone();
    } else {
        image_gray_ = image.clone();
        cv::cvtColor(image, image_rgb_, cv::COLOR_GRAY2BGR);
    }
}

void Frame::extract_features(int max_features) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Use ORB for efficiency
    auto orb = cv::ORB::create(
        max_features,    // nfeatures
        1.2f,            // scaleFactor
        8,               // nlevels
        31,              // edgeThreshold
        0,               // firstLevel
        2,               // WTA_K
        cv::ORB::HARRIS_SCORE,
        31,              // patchSize
        20               // fastThreshold
    );
    
    orb->detectAndCompute(image_gray_, cv::noArray(), keypoints_, descriptors_);
    
    // Convert to Feature objects
    features_.clear();
    features_.reserve(keypoints_.size());
    
    for (size_t i = 0; i < keypoints_.size(); ++i) {
        features_.emplace_back(keypoints_[i], descriptors_.row(i));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    extraction_time_ms_ = std::chrono::duration<double, std::milli>(end - start).count();
    
    std::cout << "Extracted " << features_.size() << " features in " 
              << extraction_time_ms_ << " ms" << std::endl;
}

size_t Frame::get_memory_usage() const {
    size_t total = sizeof(*this);
    total += image_gray_.total() * image_gray_.elemSize();
    total += image_rgb_.total() * image_rgb_.elemSize();
    total += descriptors_.total() * descriptors_.elemSize();
    total += features_.size() * sizeof(Feature);
    return total;
}

} // namespace ar_slam