#pragma once

#include <string>

#include <opencv2/core.hpp>

#include "tracking/byte_tracker.h"

namespace ui {

cv::Scalar TrackColor(int track_id);
std::string FormatTrackLabel(const tracking::TrackView& track);
std::string FormatTrackScore(const tracking::TrackView& track);
std::string FormatFocusDistance(float relative, double meters_per_unit);
void DrawTrackBox(cv::Mat& image, const tracking::TrackView& track);
void DrawFocusMarker(cv::Mat& image, float relative,
                     double meters_per_unit);

}  // namespace ui
