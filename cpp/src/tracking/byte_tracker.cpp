#include "tracking/byte_tracker.h"
#include "tracking/byte_tracker_internal.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include <opencv2/video/tracking.hpp>

namespace tracking {
namespace {

constexpr float kMinDimension = 1e-3f;

cv::Mat DetectionMeasurement(const Detection& detection) {
  cv::Mat measurement(4, 1, CV_32F);
  const float height = std::max(detection.box.height, kMinDimension);
  measurement.at<float>(0) = detection.box.x + detection.box.width * 0.5f;
  measurement.at<float>(1) = detection.box.y + detection.box.height * 0.5f;
  measurement.at<float>(2) = detection.box.width / height;
  measurement.at<float>(3) = height;
  return measurement;
}

void SetDiagonal(cv::Mat& matrix, const std::vector<float>& values) {
  matrix.setTo(0);
  for (std::size_t i = 0; i < values.size(); ++i) {
    matrix.at<float>(static_cast<int>(i), static_cast<int>(i)) =
        values[i] * values[i];
  }
}

struct Track {
  int id = 0;
  float score = 0.0f;
  TrackState state = TrackState::kNew;
  bool activated = false;
  int frame_id = 0;
  int start_frame = 0;
  int tracklet_length = 0;
  cv::KalmanFilter kalman;

  explicit Track(const Detection& detection) : score(detection.score) {
    kalman.init(8, 4, 0, CV_32F);
    cv::setIdentity(kalman.transitionMatrix);
    for (int i = 0; i < 4; ++i) {
      kalman.transitionMatrix.at<float>(i, i + 4) = 1.0f;
    }
    kalman.measurementMatrix.setTo(0);
    for (int i = 0; i < 4; ++i) {
      kalman.measurementMatrix.at<float>(i, i) = 1.0f;
    }

    const cv::Mat measurement = DetectionMeasurement(detection);
    kalman.statePost.setTo(0);
    measurement.copyTo(kalman.statePost.rowRange(0, 4));
    const float height = measurement.at<float>(3);
    SetDiagonal(kalman.errorCovPost,
                {2.0f * height / 20.0f, 2.0f * height / 20.0f, 1e-2f,
                 2.0f * height / 20.0f, 10.0f * height / 160.0f,
                 10.0f * height / 160.0f, 1e-5f,
                 10.0f * height / 160.0f});
  }

  void Activate(int new_id, int current_frame) {
    id = new_id;
    state = TrackState::kTracked;
    activated = current_frame == 1;
    frame_id = current_frame;
    start_frame = current_frame;
    tracklet_length = 0;
  }

  void Predict() {
    if (state != TrackState::kTracked) {
      kalman.statePost.at<float>(7) = 0.0f;
    }
    const float height =
        std::max(std::abs(kalman.statePost.at<float>(3)), kMinDimension);
    SetDiagonal(kalman.processNoiseCov,
                {height / 20.0f, height / 20.0f, 1e-2f, height / 20.0f,
                 height / 160.0f, height / 160.0f, 1e-5f,
                 height / 160.0f});
    const cv::Mat predicted = kalman.predict();
    predicted.copyTo(kalman.statePost);
    kalman.errorCovPre.copyTo(kalman.errorCovPost);
  }

  void Update(const Detection& detection, int current_frame) {
    const cv::Mat measurement = DetectionMeasurement(detection);
    kalman.statePost.copyTo(kalman.statePre);
    kalman.errorCovPost.copyTo(kalman.errorCovPre);
    const float height =
        std::max(std::abs(kalman.statePost.at<float>(3)), kMinDimension);
    SetDiagonal(kalman.measurementNoiseCov,
                {height / 20.0f, height / 20.0f, 1e-1f,
                 height / 20.0f});
    kalman.correct(measurement);
    score = detection.score;
    state = TrackState::kTracked;
    activated = true;
    frame_id = current_frame;
    ++tracklet_length;
  }

  void Reactivate(const Detection& detection, int current_frame) {
    Update(detection, current_frame);
    tracklet_length = 0;
  }

  cv::Rect2f Box() const {
    const float center_x = kalman.statePost.at<float>(0);
    const float center_y = kalman.statePost.at<float>(1);
    const float aspect = kalman.statePost.at<float>(2);
    const float height =
        std::max(kalman.statePost.at<float>(3), kMinDimension);
    const float width = std::max(aspect * height, 0.0f);
    return {center_x - width * 0.5f, center_y - height * 0.5f, width,
            height};
  }
};

using TrackPtr = std::shared_ptr<Track>;

float IntersectionOverUnion(const cv::Rect2f& a, const cv::Rect2f& b) {
  const float left = std::max(a.x, b.x);
  const float top = std::max(a.y, b.y);
  const float right = std::min(a.x + a.width, b.x + b.width);
  const float bottom = std::min(a.y + a.height, b.y + b.height);
  const float intersection =
      std::max(0.0f, right - left) * std::max(0.0f, bottom - top);
  const float area_a = std::max(0.0f, a.width) * std::max(0.0f, a.height);
  const float area_b = std::max(0.0f, b.width) * std::max(0.0f, b.height);
  const float union_area = area_a + area_b - intersection;
  return union_area > 0.0f ? intersection / union_area : 0.0f;
}

struct Assignment {
  std::vector<std::pair<std::size_t, std::size_t>> matches;
  std::vector<std::size_t> unmatched_tracks;
  std::vector<std::size_t> unmatched_detections;
};

struct HungarianSolution {
  std::vector<int> row_to_column;
  std::vector<long double> row_potential;
  std::vector<long double> column_potential;
};

HungarianSolution FindOptimalAssignment(
    const std::vector<std::vector<double>>& costs) {
  const int size = static_cast<int>(costs.size());
  if (size == 0) {
    return {};
  }

  std::vector<long double> row_potential(size + 1, 0.0L);
  std::vector<long double> column_potential(size + 1, 0.0L);
  std::vector<int> column_row(size + 1, 0);
  std::vector<int> previous_column(size + 1, 0);

  for (int row = 1; row <= size; ++row) {
    column_row[0] = row;
    int current_column = 0;
    std::vector<long double> minimum(
        size + 1, std::numeric_limits<long double>::infinity());
    std::vector<bool> used(size + 1, false);
    do {
      used[current_column] = true;
      const int current_row = column_row[current_column];
      long double delta = std::numeric_limits<long double>::infinity();
      int next_column = 0;
      for (int column = 1; column <= size; ++column) {
        if (used[column]) {
          continue;
        }
        const long double reduced =
            static_cast<long double>(costs[current_row - 1][column - 1]) -
            row_potential[current_row] - column_potential[column];
        if (reduced < minimum[column]) {
          minimum[column] = reduced;
          previous_column[column] = current_column;
        }
        if (minimum[column] < delta ||
            (minimum[column] == delta &&
             (next_column == 0 || column < next_column))) {
          delta = minimum[column];
          next_column = column;
        }
      }
      for (int column = 0; column <= size; ++column) {
        if (used[column]) {
          row_potential[column_row[column]] += delta;
          column_potential[column] -= delta;
        } else {
          minimum[column] -= delta;
        }
      }
      current_column = next_column;
    } while (column_row[current_column] != 0);

    do {
      const int next_column = previous_column[current_column];
      column_row[current_column] = column_row[next_column];
      current_column = next_column;
    } while (current_column != 0);
  }

  std::vector<int> row_column(size, -1);
  for (int column = 1; column <= size; ++column) {
    if (column_row[column] != 0) {
      row_column[column_row[column] - 1] = column - 1;
    }
  }
  row_potential.erase(row_potential.begin());
  column_potential.erase(column_potential.begin());
  return {std::move(row_column), std::move(row_potential),
          std::move(column_potential)};
}

bool IsTightEdge(double cost, long double row_potential,
                 long double column_potential) {
  const long double reduced = static_cast<long double>(cost) - row_potential -
                              column_potential;
  const long double scale =
      std::max({1.0L, std::abs(static_cast<long double>(cost)),
                std::abs(row_potential), std::abs(column_potential)});
  const long double tolerance =
      128.0L * std::numeric_limits<long double>::epsilon() * scale;
  return std::abs(reduced) <= tolerance;
}

std::vector<int> CanonicalizeOptimalAssignment(
    const std::vector<std::vector<double>>& costs,
    HungarianSolution solution) {
  const int size = static_cast<int>(costs.size());
  auto& row_to_column = solution.row_to_column;
  std::vector<int> column_to_row(size, -1);
  for (int row = 0; row < size; ++row) {
    column_to_row[row_to_column[row]] = row;
  }

  std::vector<std::vector<bool>> tight(
      size, std::vector<bool>(size, false));
  for (int row = 0; row < size; ++row) {
    for (int column = 0; column < size; ++column) {
      tight[row][column] =
          IsTightEdge(costs[row][column], solution.row_potential[row],
                      solution.column_potential[column]);
    }
    // The primary Hungarian result is authoritative for its matched edge even
    // if accumulated floating-point error is just outside the tight tolerance.
    tight[row][row_to_column[row]] = true;
  }

  std::vector<bool> fixed_column(size, false);
  for (int fixed_row = 0; fixed_row < size; ++fixed_row) {
    // Mark every alternating-graph node that can reach fixed_row. An unmatched
    // tight edge fixed_row->column can be forced precisely when that column can
    // return to fixed_row along an alternating path, forming an optimal cycle.
    std::vector<bool> reachable_row(size, false);
    std::vector<bool> reachable_column(size, false);
    std::queue<std::pair<bool, int>> reverse_search;
    reachable_row[fixed_row] = true;
    reverse_search.emplace(true, fixed_row);
    while (!reverse_search.empty()) {
      const auto [is_row, index] = reverse_search.front();
      reverse_search.pop();
      if (is_row) {
        const int matched_column = row_to_column[index];
        if (!fixed_column[matched_column] &&
            !reachable_column[matched_column]) {
          reachable_column[matched_column] = true;
          reverse_search.emplace(false, matched_column);
        }
        continue;
      }
      for (int row = fixed_row; row < size; ++row) {
        if (row_to_column[row] != index && tight[row][index] &&
            !reachable_row[row]) {
          reachable_row[row] = true;
          reverse_search.emplace(true, row);
        }
      }
    }

    const int current_column = row_to_column[fixed_row];
    int chosen_column = current_column;
    for (int column = 0; column < size; ++column) {
      if (!fixed_column[column] && tight[fixed_row][column] &&
          (column == current_column || reachable_column[column])) {
        chosen_column = column;
        break;
      }
    }

    if (chosen_column != current_column) {
      // Recover an alternating path chosen_column -> fixed_row and rotate the
      // current perfect matching around the resulting zero-reduced-cost cycle.
      const int node_count = 2 * size;
      const int start = size + chosen_column;
      const int target = fixed_row;
      std::vector<int> parent(node_count, -1);
      std::vector<bool> seen(node_count, false);
      std::queue<int> forward_search;
      seen[start] = true;
      forward_search.push(start);
      while (!forward_search.empty() && !seen[target]) {
        const int node = forward_search.front();
        forward_search.pop();
        if (node >= size) {
          const int column = node - size;
          const int row = column_to_row[column];
          if (row >= fixed_row && !seen[row]) {
            seen[row] = true;
            parent[row] = node;
            forward_search.push(row);
          }
          continue;
        }
        for (int column = 0; column < size; ++column) {
          const int column_node = size + column;
          if (!fixed_column[column] &&
              column != row_to_column[node] && tight[node][column] &&
              !seen[column_node]) {
            seen[column_node] = true;
            parent[column_node] = node;
            forward_search.push(column_node);
          }
        }
      }
      if (!seen[target]) {
        throw std::logic_error("tight assignment cycle was not recoverable");
      }

      std::vector<int> path;
      for (int node = target; node != -1; node = parent[node]) {
        path.push_back(node);
        if (node == start) {
          break;
        }
      }
      std::reverse(path.begin(), path.end());
      row_to_column[fixed_row] = chosen_column;
      for (std::size_t index = 1; index + 1 < path.size(); index += 2) {
        const int row = path[index];
        const int column = path[index + 1] - size;
        row_to_column[row] = column;
      }
      std::fill(column_to_row.begin(), column_to_row.end(), -1);
      for (int row = 0; row < size; ++row) {
        column_to_row[row_to_column[row]] = row;
      }
    }
    fixed_column[chosen_column] = true;
  }
  return row_to_column;
}

std::vector<int> Hungarian(const std::vector<std::vector<double>>& costs) {
  return CanonicalizeOptimalAssignment(costs, FindOptimalAssignment(costs));
}

Assignment Assign(const std::vector<TrackPtr>& tracks,
                  const std::vector<Detection>& detections, float threshold,
                  bool fuse_score) {
  Assignment result;
  if (tracks.empty() || detections.empty()) {
    for (std::size_t i = 0; i < tracks.size(); ++i) {
      result.unmatched_tracks.push_back(i);
    }
    for (std::size_t i = 0; i < detections.size(); ++i) {
      result.unmatched_detections.push_back(i);
    }
    return result;
  }

  const double padding = static_cast<double>(threshold) + 1.0;
  std::vector<std::vector<double>> original(
      tracks.size(), std::vector<double>(detections.size(), padding));
  for (std::size_t track = 0; track < tracks.size(); ++track) {
    for (std::size_t detection = 0; detection < detections.size();
         ++detection) {
      const double iou =
          IntersectionOverUnion(tracks[track]->Box(), detections[detection].box);
      const double cost =
          1.0 - iou * (fuse_score ? detections[detection].score : 1.0);
      original[track][detection] = cost;
    }
  }

  const std::vector<int> columns =
      internal::SolveDeterministicAssignment(original, padding);
  std::vector<bool> track_matched(tracks.size(), false);
  std::vector<bool> detection_matched(detections.size(), false);
  for (std::size_t track = 0; track < tracks.size(); ++track) {
    const int detection = columns[track];
    if (detection < 0 ||
        static_cast<std::size_t>(detection) >= detections.size() ||
        original[track][detection] > static_cast<double>(threshold)) {
      continue;
    }
    result.matches.emplace_back(track, static_cast<std::size_t>(detection));
    track_matched[track] = true;
    detection_matched[detection] = true;
  }
  for (std::size_t track = 0; track < tracks.size(); ++track) {
    if (!track_matched[track]) {
      result.unmatched_tracks.push_back(track);
    }
  }
  for (std::size_t detection = 0; detection < detections.size(); ++detection) {
    if (!detection_matched[detection]) {
      result.unmatched_detections.push_back(detection);
    }
  }
  return result;
}

std::vector<TrackPtr> Joint(const std::vector<TrackPtr>& first,
                            const std::vector<TrackPtr>& second) {
  std::vector<TrackPtr> result;
  std::unordered_set<int> ids;
  for (const auto& track : first) {
    if (ids.insert(track->id).second) {
      result.push_back(track);
    }
  }
  for (const auto& track : second) {
    if (ids.insert(track->id).second) {
      result.push_back(track);
    }
  }
  return result;
}

std::vector<TrackPtr> SelectState(const std::vector<TrackPtr>& tracks,
                                  TrackState state) {
  std::vector<TrackPtr> result;
  std::unordered_set<int> ids;
  for (const auto& track : tracks) {
    if (track->state == state && ids.insert(track->id).second) {
      result.push_back(track);
    }
  }
  return result;
}

std::vector<Detection> SelectDetections(
    const std::vector<Detection>& detections,
    const std::vector<std::size_t>& indices) {
  std::vector<Detection> result;
  result.reserve(indices.size());
  for (const std::size_t index : indices) {
    result.push_back(detections[index]);
  }
  return result;
}

}  // namespace

std::vector<int> internal::SolveDeterministicAssignment(
    const std::vector<std::vector<double>>& costs, double padding_cost) {
  const std::size_t rows = costs.size();
  const std::size_t columns = rows == 0 ? 0 : costs.front().size();
  const std::size_t size = std::max(rows, columns);
  std::vector<std::vector<double>> padded(
      size, std::vector<double>(size, padding_cost));
  for (std::size_t row = 0; row < rows; ++row) {
    if (costs[row].size() != columns) {
      throw std::invalid_argument("assignment cost matrix must be rectangular");
    }
    std::copy(costs[row].begin(), costs[row].end(), padded[row].begin());
  }
  auto assignment = Hungarian(padded);
  assignment.resize(rows);
  for (int& column : assignment) {
    if (column >= static_cast<int>(columns)) {
      column = -1;
    }
  }
  return assignment;
}

struct ByteTracker::Impl {
  explicit Impl(Config tracker_config)
      : config(std::move(tracker_config)),
        buffer_size(static_cast<int>(std::round(
            config.frame_rate / 30.0 * config.track_buffer))) {}

  std::vector<Detection> Filter(const std::vector<Detection>& input,
                                float minimum, float maximum) const {
    std::vector<Detection> result;
    for (const auto& detection : input) {
      const bool below_upper = maximum == 1.0f
                                   ? detection.score <= maximum
                                   : detection.score < maximum;
      if (detection.score >= minimum && below_upper) {
        result.push_back(detection);
      }
    }
    return result;
  }

  void RemoveLostOlderThan() {
    for (const auto& track : lost) {
      if (frame_id - track->frame_id > buffer_size) {
        track->state = TrackState::kRemoved;
      }
    }
    lost = SelectState(lost, TrackState::kLost);
  }

  void RemoveDuplicateTracks(float max_cost) {
    std::vector<bool> remove_tracked(tracked.size(), false);
    std::vector<bool> remove_lost(lost.size(), false);
    for (std::size_t tracked_index = 0; tracked_index < tracked.size();
         ++tracked_index) {
      for (std::size_t lost_index = 0; lost_index < lost.size(); ++lost_index) {
        const float cost = 1.0f - IntersectionOverUnion(
                                      tracked[tracked_index]->Box(),
                                      lost[lost_index]->Box());
        if (cost >= max_cost) {
          continue;
        }
        const int tracked_age = tracked[tracked_index]->frame_id -
                                tracked[tracked_index]->start_frame;
        const int lost_age =
            lost[lost_index]->frame_id - lost[lost_index]->start_frame;
        if (tracked_age > lost_age) {
          remove_lost[lost_index] = true;
        } else {
          remove_tracked[tracked_index] = true;
        }
      }
    }

    std::vector<TrackPtr> kept_tracked;
    for (std::size_t i = 0; i < tracked.size(); ++i) {
      if (!remove_tracked[i]) {
        kept_tracked.push_back(tracked[i]);
      } else {
        tracked[i]->state = TrackState::kRemoved;
      }
    }
    std::vector<TrackPtr> kept_lost;
    for (std::size_t i = 0; i < lost.size(); ++i) {
      if (!remove_lost[i]) {
        kept_lost.push_back(lost[i]);
      } else {
        lost[i]->state = TrackState::kRemoved;
      }
    }
    tracked = std::move(kept_tracked);
    lost = std::move(kept_lost);
  }

  std::vector<TrackView> ActiveViews() const {
    std::vector<TrackView> result;
    for (const auto& track : tracked) {
      if (track->state == TrackState::kTracked && track->activated) {
        result.push_back(
            {track->id, track->Box(), track->score, TrackState::kTracked});
      }
    }
    std::sort(result.begin(), result.end(),
              [](const TrackView& a, const TrackView& b) {
                return a.id < b.id;
              });
    return result;
  }

  std::vector<TrackView> Update(const std::vector<Detection>& input) {
    ++frame_id;
    const auto high = Filter(input, config.high_thresh, 1.0f);
    const auto low = Filter(input, config.low_thresh, config.high_thresh);

    std::vector<TrackPtr> confirmed;
    std::vector<TrackPtr> unconfirmed;
    for (const auto& track : tracked) {
      (track->activated ? confirmed : unconfirmed).push_back(track);
    }

    const auto pool = Joint(confirmed, lost);
    for (const auto& track : pool) {
      track->Predict();
    }

    // Stage 1: score-fused IoU cost against high detections.
    const auto first = Assign(pool, high, config.match_thresh,
                              /*fuse_score=*/true);
    std::vector<TrackPtr> touched;
    for (const auto& match : first.matches) {
      const auto& track = pool[match.first];
      if (track->state == TrackState::kTracked) {
        track->Update(high[match.second], frame_id);
      } else {
        track->Reactivate(high[match.second], frame_id);
      }
      touched.push_back(track);
    }

    // Stage 2: plain IoU cost against low detections.
    std::vector<TrackPtr> remaining_tracked;
    for (const std::size_t index : first.unmatched_tracks) {
      if (pool[index]->state == TrackState::kTracked) {
        remaining_tracked.push_back(pool[index]);
      }
    }
    const auto second = Assign(remaining_tracked, low, 0.50f,
                               /*fuse_score=*/false);
    for (const auto& match : second.matches) {
      const auto& track = remaining_tracked[match.first];
      track->Update(low[match.second], frame_id);
      touched.push_back(track);
    }
    std::vector<TrackPtr> newly_lost;
    for (const std::size_t index : second.unmatched_tracks) {
      const auto& track = remaining_tracked[index];
      track->state = TrackState::kLost;
      newly_lost.push_back(track);
    }

    // Confirm one-frame tracks at cost <= 0.70; remove unconfirmed misses.
    const auto unmatched_high =
        SelectDetections(high, first.unmatched_detections);
    const auto confirmation = Assign(unconfirmed, unmatched_high, 0.70f,
                                     /*fuse_score=*/true);
    for (const auto& match : confirmation.matches) {
      const auto& track = unconfirmed[match.first];
      track->Update(unmatched_high[match.second], frame_id);
      touched.push_back(track);
    }
    for (const std::size_t index : confirmation.unmatched_tracks) {
      unconfirmed[index]->state = TrackState::kRemoved;
    }

    // Only unmatched detections with score >= high_thresh create tracks.
    std::vector<TrackPtr> activated;
    for (const std::size_t index : confirmation.unmatched_detections) {
      const auto& detection = unmatched_high[index];
      if (detection.score < config.high_thresh) {
        continue;
      }
      auto track = std::make_shared<Track>(detection);
      track->Activate(next_id++, frame_id);
      ++total_created;
      activated.push_back(std::move(track));
    }

    tracked = SelectState(Joint(Joint(tracked, touched), activated),
                          TrackState::kTracked);
    lost = SelectState(Joint(lost, newly_lost), TrackState::kLost);
    RemoveLostOlderThan();
    RemoveDuplicateTracks(/*max_cost=*/0.15f);

    return ActiveViews();
  }

  Config config;
  int frame_id = 0;
  int next_id = 1;
  int buffer_size = 0;
  int total_created = 0;
  std::vector<TrackPtr> tracked;
  std::vector<TrackPtr> lost;
};

std::vector<Detection> SelectPersonDetections(
    const std::vector<SourceDetection>& detections, float low_thresh) {
  std::vector<Detection> selected;
  for (const auto& detection : detections) {
    if (detection.class_id == 0 && detection.score >= low_thresh) {
      selected.push_back({detection.box, detection.score});
    }
  }
  return selected;
}

bool ShouldRenderRawDetection(int class_id) { return class_id != 0; }

int CountDisplayedObjects(int active_tracks,
                          const std::vector<SourceDetection>& detections,
                          float visible_thresh) {
  int count = active_tracks;
  for (const auto& detection : detections) {
    if (ShouldRenderRawDetection(detection.class_id) &&
        detection.score >= visible_thresh) {
      ++count;
    }
  }
  return count;
}

std::string ValidateConfig(const Config& config) {
  if (!(config.low_thresh >= 0.0f &&
        config.low_thresh < config.high_thresh &&
        config.high_thresh <= 1.0f)) {
    return "tracking thresholds must satisfy 0 <= low < high <= 1";
  }
  if (!(config.match_thresh >= 0.0f && config.match_thresh <= 1.0f)) {
    return "match threshold must be in [0, 1]";
  }
  if (config.track_buffer <= 0) {
    return "track buffer must be greater than zero";
  }
  if (config.frame_rate <= 0) {
    return "frame rate must be greater than zero";
  }
  return {};
}

ByteTracker::ByteTracker(Config config) {
  const std::string error = ValidateConfig(config);
  if (!error.empty()) {
    throw std::invalid_argument(error);
  }
  impl_ = std::make_unique<Impl>(std::move(config));
}

ByteTracker::~ByteTracker() = default;
ByteTracker::ByteTracker(ByteTracker&&) noexcept = default;
ByteTracker& ByteTracker::operator=(ByteTracker&&) noexcept = default;

std::vector<TrackView> ByteTracker::Update(
    const std::vector<Detection>& detections) {
  return impl_->Update(detections);
}

TrackerStats ByteTracker::Snapshot() const {
  const auto active = impl_->ActiveViews();
  return {static_cast<int>(active.size()), static_cast<int>(impl_->lost.size()),
          impl_->total_created};
}

}  // namespace tracking
