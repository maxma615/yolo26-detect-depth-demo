#include "ui/browser_overlay.h"

#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

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

void TestLabelsAndUnits() {
  const tracking::TrackView track{12, {20, 30, 40, 80}, 0.86f,
                                  tracking::TrackState::kTracked};
  CHECK(ui::FormatTrackLabel(track) == "PERSON #12");
  CHECK(ui::FormatTrackScore(track) == "0.86");
  CHECK(ui::FormatFocusDistance(0.18f, 0.0) == "0.18");
  CHECK(ui::FormatFocusDistance(0.18f, 10.0) == "1.8m");
}

void TestTrackColorsCycleDeterministically() {
  CHECK(ui::TrackColor(1) == cv::Scalar(0x00, 0x3C, 0xFF));
  CHECK(ui::TrackColor(6) == cv::Scalar(0x00, 0xAE, 0xFF));
  CHECK(ui::TrackColor(7) == ui::TrackColor(1));
}

void TestDrawingChangesOnlyExpectedRegions() {
  cv::Mat image(240, 320, CV_8UC3, cv::Scalar(246, 246, 244));
  const cv::Mat before = image.clone();
  const tracking::TrackView track{1, {20, 30, 40, 80}, 0.9f,
                                  tracking::TrackState::kTracked};
  ui::DrawTrackBox(image, track);

  cv::Mat diff;
  cv::absdiff(image, before, diff);
  CHECK(cv::countNonZero(diff.reshape(1)) > 0);
  CHECK(image.at<cv::Vec3b>(30, 20) != before.at<cv::Vec3b>(30, 20));
  CHECK(image.at<cv::Vec3b>(200, 300) == before.at<cv::Vec3b>(200, 300));
}

void TestTrackLabelIsClampedIntoImage() {
  cv::Mat image(80, 100, CV_8UC3, cv::Scalar(246, 246, 244));
  const cv::Mat before = image.clone();
  const tracking::TrackView track{2, {-12, 2, 20, 30}, 0.9f,
                                  tracking::TrackState::kTracked};
  ui::DrawTrackBox(image, track);

  cv::Mat diff;
  cv::absdiff(image, before, diff);
  CHECK(cv::countNonZero(diff.rowRange(0, 20).reshape(1)) > 0);
}

void TestNarrowImageReflowsFullTrackTagInsideBounds() {
  cv::Mat image(80, 76, CV_8UC3, cv::Scalar(246, 246, 244));
  const tracking::TrackView track{12, {4, 52, 20, 20}, 0.86f,
                                  tracking::TrackState::kTracked};
  ui::DrawTrackBox(image, track);

  cv::Mat orange;
  const cv::Scalar color = ui::TrackColor(track.id);
  cv::inRange(image, color, color, orange);
  // A reflowed two-line tag reaches this in-bounds upper ROI. A too-wide,
  // clipped one-line tag starts much lower and cannot satisfy this assertion.
  CHECK(cv::countNonZero(orange(cv::Rect(0, 18, 76, 6))) >= 45);

  cv::Mat white;
  cv::inRange(image, cv::Scalar(255, 255, 255), cv::Scalar(255, 255, 255),
              white);
  // The lower tag row must retain visible label/score content inside the image.
  CHECK(cv::countNonZero(white(cv::Rect(8, 35, 44, 10))) >= 20);
}

void TestFocusMarkerIsCenteredAndLeavesCornerUntouched() {
  cv::Mat image(240, 320, CV_8UC3, cv::Scalar(246, 246, 244));
  const cv::Mat before = image.clone();
  ui::DrawFocusMarker(image, 0.18f, 0.0);

  cv::Mat diff;
  cv::absdiff(image, before, diff);
  CHECK(cv::countNonZero(diff.reshape(1)) > 0);
  CHECK(image.at<cv::Vec3b>(120, 160) != before.at<cv::Vec3b>(120, 160));
  CHECK(image.at<cv::Vec3b>(10, 10) == before.at<cv::Vec3b>(10, 10));
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
  Run("labels and units", TestLabelsAndUnits, &failures);
  Run("track colors cycle deterministically", TestTrackColorsCycleDeterministically,
      &failures);
  Run("track drawing changes only expected regions",
      TestDrawingChangesOnlyExpectedRegions, &failures);
  Run("track label is clamped into image", TestTrackLabelIsClampedIntoImage,
      &failures);
  Run("narrow image reflows full track tag inside bounds",
      TestNarrowImageReflowsFullTrackTagInsideBounds, &failures);
  Run("focus marker is centered and leaves corner untouched",
      TestFocusMarkerIsCenteredAndLeavesCornerUntouched, &failures);
  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << "All browser overlay tests passed\n";
  return 0;
}
