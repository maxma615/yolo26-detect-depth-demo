#include "tracking/byte_tracker.h"
#include "tracking/byte_tracker_internal.h"

#include <cmath>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::ostringstream message;                                              \
      message << __FILE__ << ':' << __LINE__ << ": CHECK failed: "            \
              << #condition;                                                   \
      throw std::runtime_error(message.str());                                 \
    }                                                                          \
  } while (false)

void TestMovingPersonKeepsId() {
  tracking::ByteTracker t(tracking::Config{});
  auto first = t.Update({{{10, 10, 40, 80}, 0.90f}});
  auto second = t.Update({{{14, 11, 40, 80}, 0.88f}});
  CHECK(first.size() == 1);
  CHECK(second.size() == 1);
  CHECK(first[0].id == second[0].id);
}

void TestLowScoreRecoversExistingTrackButCannotCreateOne() {
  tracking::ByteTracker empty(tracking::Config{});
  CHECK(empty.Update({{{10, 10, 40, 80}, 0.20f}}).empty());
  CHECK(empty.Snapshot().total_created == 0);

  tracking::ByteTracker active(tracking::Config{});
  const auto created = active.Update({{{10, 10, 40, 80}, 0.90f}});
  CHECK(created.size() == 1);
  const int id = created[0].id;
  auto recovered = active.Update({{{12, 10, 40, 80}, 0.20f}});
  CHECK(recovered.size() == 1);
  CHECK(recovered[0].id == id);
}

void TestLostTrackExpiresAfterBuffer() {
  tracking::Config cfg;
  cfg.track_buffer = 2;
  tracking::ByteTracker t(cfg);
  t.Update({{{10, 10, 40, 80}, 0.90f}});
  CHECK(t.Update({}).empty());
  CHECK(t.Snapshot().lost == 1);
  t.Update({});
  t.Update({});
  CHECK(t.Snapshot().lost == 0);
}

void TestMultiplePeopleHaveUniqueIds() {
  tracking::ByteTracker t(tracking::Config{});
  auto tracks = t.Update({{{10, 10, 30, 70}, 0.9f},
                          {{200, 20, 35, 75}, 0.85f}});
  CHECK(tracks.size() == 2);
  CHECK(tracks[0].id != tracks[1].id);
}

void TestLostTrackReactivatesWithSameId() {
  tracking::ByteTracker t(tracking::Config{});
  const auto created = t.Update({{{40, 20, 30, 70}, 0.90f}});
  CHECK(created.size() == 1);
  const int id = created[0].id;
  CHECK(t.Update({}).empty());
  auto recovered = t.Update({{{41, 20, 30, 70}, 0.91f}});
  CHECK(recovered.size() == 1);
  CHECK(recovered[0].id == id);
  CHECK(t.Snapshot().lost == 0);
}

void TestLaterTrackNeedsConfirmationAndMissRemovesIt() {
  tracking::ByteTracker t(tracking::Config{});
  CHECK(t.Update({}).empty());
  CHECK(t.Update({{{20, 20, 20, 50}, 0.90f}}).empty());
  CHECK(t.Snapshot().total_created == 1);
  CHECK(t.Update({}).empty());
  CHECK(t.Snapshot().active == 0);
  CHECK(t.Snapshot().lost == 0);
}

void TestLaterTrackBecomesActiveAfterConfirmation() {
  tracking::ByteTracker t(tracking::Config{});
  CHECK(t.Update({}).empty());
  CHECK(t.Update({{{20, 20, 20, 50}, 0.90f}}).empty());
  auto confirmed = t.Update({{{21, 20, 20, 50}, 0.92f}});
  CHECK(confirmed.size() == 1);
  CHECK(confirmed[0].id == 1);
  CHECK(confirmed[0].state == tracking::TrackState::kTracked);
}

void TestUnconfirmedTrackAdoptsConfirmationMeasurement() {
  tracking::ByteTracker t(tracking::Config{});
  CHECK(t.Update({}).empty());
  CHECK(t.Update({{{20, 20, 20, 50}, 0.90f}}).empty());
  auto confirmed = t.Update({{{30, 20, 20, 50}, 0.92f}});
  CHECK(confirmed.size() == 1);
  CHECK(confirmed[0].box.x > 20.0f);
}

void TestDeterministicSequenceProducesIdenticalViews() {
  tracking::ByteTracker first(tracking::Config{});
  tracking::ByteTracker second(tracking::Config{});
  const std::vector<std::vector<tracking::Detection>> frames = {
      {{{10, 10, 30, 70}, 0.90f}, {{150, 15, 35, 75}, 0.90f}},
      {{{13, 11, 30, 70}, 0.85f}, {{147, 16, 35, 75}, 0.85f}},
      {{{16, 12, 30, 70}, 0.20f}, {{144, 17, 35, 75}, 0.20f}},
      {},
      {{{20, 13, 30, 70}, 0.88f}, {{140, 18, 35, 75}, 0.88f}},
  };
  const std::vector<std::size_t> expected_sizes = {2, 2, 2, 0, 2};

  for (std::size_t frame_index = 0; frame_index < frames.size();
       ++frame_index) {
    const auto& frame = frames[frame_index];
    const auto a = first.Update(frame);
    const auto b = second.Update(frame);
    CHECK(a.size() == expected_sizes[frame_index]);
    CHECK(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
      CHECK(a[i].id == b[i].id);
      CHECK(a[i].state == b[i].state);
      CHECK(a[i].score == b[i].score);
      CHECK(std::abs(a[i].box.x - b[i].box.x) < 1e-6f);
      CHECK(std::abs(a[i].box.y - b[i].box.y) < 1e-6f);
      CHECK(std::abs(a[i].box.width - b[i].box.width) < 1e-6f);
      CHECK(std::abs(a[i].box.height - b[i].box.height) < 1e-6f);
    }
  }
}

void TestEqualCostTieUsesLowerTrackAndDetectionIndices() {
  tracking::ByteTracker t(tracking::Config{});
  auto created = t.Update({{{30, 30, 40, 80}, 0.90f},
                           {{30, 30, 40, 80}, 0.90f}});
  CHECK(created.size() == 2);
  auto matched = t.Update({{{31, 30, 40, 80}, 0.91f},
                           {{31, 30, 40, 80}, 0.81f}});
  CHECK(matched.size() == 2);
  CHECK(matched[0].id == 1);
  CHECK(matched[0].score == 0.91f);
  CHECK(matched[1].id == 2);
  CHECK(matched[1].score == 0.81f);
}

void TestHungarianFindsLexicographicallyLowestGlobalOptimum() {
  const std::vector<std::vector<double>> costs = {
      {0.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
      {0.0, 0.0, 0.0},
  };
  const auto assignment =
      tracking::internal::SolveDeterministicAssignment(costs, 2.0);
  CHECK(assignment == std::vector<int>({0, 2, 1}));
}

void TestHungarianRectangularAssignmentIsNonEmptyAndDeterministic() {
  const std::vector<std::vector<double>> costs = {
      {0.0, 0.0},
      {0.0, 0.0},
      {0.0, 0.0},
  };
  const auto first =
      tracking::internal::SolveDeterministicAssignment(costs, 2.0);
  const auto second =
      tracking::internal::SolveDeterministicAssignment(costs, 2.0);
  CHECK(first == std::vector<int>({0, 1, -1}));
  CHECK(second == first);
}

void TestValidateConfigReturnsExactErrors() {
  tracking::Config cfg;
  CHECK(tracking::ValidateConfig(cfg).empty());

  cfg.low_thresh = cfg.high_thresh;
  CHECK(tracking::ValidateConfig(cfg) ==
        "tracking thresholds must satisfy 0 <= low < high <= 1");
  cfg = tracking::Config{};
  cfg.low_thresh = -0.01f;
  CHECK(tracking::ValidateConfig(cfg) ==
        "tracking thresholds must satisfy 0 <= low < high <= 1");
  cfg = tracking::Config{};
  cfg.high_thresh = 1.01f;
  CHECK(tracking::ValidateConfig(cfg) ==
        "tracking thresholds must satisfy 0 <= low < high <= 1");
  cfg = tracking::Config{};
  cfg.match_thresh = -0.01f;
  CHECK(tracking::ValidateConfig(cfg) ==
        "match threshold must be in [0, 1]");
  cfg.match_thresh = 1.01f;
  CHECK(tracking::ValidateConfig(cfg) ==
        "match threshold must be in [0, 1]");
  cfg = tracking::Config{};
  cfg.track_buffer = 0;
  CHECK(tracking::ValidateConfig(cfg) ==
        "track buffer must be greater than zero");
  cfg = tracking::Config{};
  cfg.frame_rate = 0;
  CHECK(tracking::ValidateConfig(cfg) ==
        "frame rate must be greater than zero");
}

void TestSelectPersonDetectionsKeepsOnlyPersonsAtLowThreshold() {
  std::vector<tracking::SourceDetection> input = {
      {{0, 0, 10, 20}, 0.09f, 0},
      {{1, 1, 10, 20}, 0.10f, 0},
      {{2, 2, 10, 20}, 0.95f, 2}};
  auto got = tracking::SelectPersonDetections(input, 0.10f);
  CHECK(got.size() == 1);
  CHECK(got[0].score == 0.10f);
}

void TestRawDetectionRenderingExcludesPersons() {
  CHECK(!tracking::ShouldRenderRawDetection(0));
  CHECK(tracking::ShouldRenderRawDetection(2));
}

void TestDisplayedObjectCountUsesTracksAndVisibleNonPersons() {
  std::vector<tracking::SourceDetection> input = {
      {{0, 0, 10, 20}, 0.95f, 0},
      {{1, 1, 10, 20}, 0.24f, 2},
      {{2, 2, 10, 20}, 0.25f, 2},
      {{3, 3, 10, 20}, 0.90f, 7}};
  CHECK(tracking::CountDisplayedObjects(3, input, 0.25f) == 5);
}

template <typename Test>
void Run(const char* name, Test test, int* failures) {
  try {
    test();
    std::cout << "[PASS] " << name << '\n';
  } catch (const std::exception& error) {
    ++*failures;
    std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
  }
}

}  // namespace

int main() {
  int failures = 0;
  Run("moving person keeps ID", TestMovingPersonKeepsId, &failures);
  Run("low score recovers but cannot create",
      TestLowScoreRecoversExistingTrackButCannotCreateOne, &failures);
  Run("lost track expires after buffer", TestLostTrackExpiresAfterBuffer,
      &failures);
  Run("multiple people have unique IDs", TestMultiplePeopleHaveUniqueIds,
      &failures);
  Run("lost track reactivates with same ID", TestLostTrackReactivatesWithSameId,
      &failures);
  Run("later unconfirmed miss removes track",
      TestLaterTrackNeedsConfirmationAndMissRemovesIt, &failures);
  Run("later track activates after confirmation",
      TestLaterTrackBecomesActiveAfterConfirmation, &failures);
  Run("unconfirmed track adopts confirmation measurement",
      TestUnconfirmedTrackAdoptsConfirmationMeasurement, &failures);
  Run("deterministic sequence has identical views",
      TestDeterministicSequenceProducesIdenticalViews, &failures);
  Run("equal-cost ties prefer lower indices",
      TestEqualCostTieUsesLowerTrackAndDetectionIndices, &failures);
  Run("Hungarian chooses lexicographically lowest global optimum",
      TestHungarianFindsLexicographicallyLowestGlobalOptimum, &failures);
  Run("Hungarian rectangular assignment is deterministic",
      TestHungarianRectangularAssignmentIsNonEmptyAndDeterministic, &failures);
  Run("config validation returns exact errors",
      TestValidateConfigReturnsExactErrors, &failures);
  Run("person adapter keeps only persons at low threshold",
      TestSelectPersonDetectionsKeepsOnlyPersonsAtLowThreshold, &failures);
  Run("raw detection rendering excludes persons",
      TestRawDetectionRenderingExcludesPersons, &failures);
  Run("display count uses tracks and visible non-persons",
      TestDisplayedObjectCountUsesTracksAndVisibleNonPersons, &failures);
  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << "All ByteTrack tests passed\n";
  return 0;
}
