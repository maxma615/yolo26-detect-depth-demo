#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace tracking {

struct Detection {
  cv::Rect2f box;
  float score = 0.0f;
};

struct SourceDetection {
  cv::Rect2f box;
  float score = 0.0f;
  int class_id = -1;
};

enum class TrackState { kNew, kTracked, kLost, kRemoved };

struct TrackView {
  int id = 0;
  cv::Rect2f box;
  float score = 0.0f;
  TrackState state = TrackState::kNew;
};

struct Config {
  float high_thresh = 0.30f;
  float low_thresh = 0.10f;
  float match_thresh = 0.80f;
  int track_buffer = 60;
  int frame_rate = 30;
};

struct TrackerStats {
  int active = 0;
  int lost = 0;
  int total_created = 0;
};

std::string ValidateConfig(const Config& config);
std::vector<Detection> SelectPersonDetections(
    const std::vector<SourceDetection>& detections, float low_thresh);
bool ShouldRenderRawDetection(int class_id);
int CountDisplayedObjects(int active_tracks,
                          const std::vector<SourceDetection>& detections,
                          float visible_thresh);

class ByteTracker {
 public:
  explicit ByteTracker(Config config);
  ~ByteTracker();

  ByteTracker(ByteTracker&&) noexcept;
  ByteTracker& operator=(ByteTracker&&) noexcept;
  ByteTracker(const ByteTracker&) = delete;
  ByteTracker& operator=(const ByteTracker&) = delete;

  std::vector<TrackView> Update(const std::vector<Detection>& detections);
  TrackerStats Snapshot() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tracking
