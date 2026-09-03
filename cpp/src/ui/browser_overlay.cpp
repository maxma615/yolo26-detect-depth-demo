#include "ui/browser_overlay.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>

#include <opencv2/imgproc.hpp>

namespace ui {
namespace {

constexpr int kFontFace = cv::FONT_HERSHEY_SIMPLEX;
constexpr double kFontScale = 0.70;
constexpr int kFontThickness = 2;
constexpr int kTagPaddingX = 6;
constexpr int kTagPaddingY = 4;
constexpr int kSeparatorRadius = 3;

const std::array<cv::Scalar, 6>& TrackColors() {
  static const std::array<cv::Scalar, 6> colors = {
      cv::Scalar(0x00, 0x3C, 0xFF), cv::Scalar(0x00, 0x55, 0xFF),
      cv::Scalar(0x00, 0x73, 0xFF), cv::Scalar(0x00, 0x8C, 0xFF),
      cv::Scalar(0x00, 0x9D, 0xFF), cv::Scalar(0x00, 0xAE, 0xFF),
  };
  return colors;
}

int Clamp(int value, int low, int high) {
  return std::max(low, std::min(value, high));
}

cv::Point TagOrigin(const cv::Mat& image, const cv::Rect& box, int width,
                    int height) {
  const int max_x = std::max(0, image.cols - width);
  const int max_y = std::max(0, image.rows - height);
  return {Clamp(box.x, 0, max_x), Clamp(box.y - height - 4, 0, max_y)};
}

struct TextMetrics {
  cv::Size size;
  int baseline = 0;
};

TextMetrics MeasureText(const std::string& text, double scale) {
  TextMetrics metrics;
  metrics.size = cv::getTextSize(text, kFontFace, scale, kFontThickness,
                                 &metrics.baseline);
  return metrics;
}

cv::Mat FitTagToImage(const cv::Mat& tag, const cv::Mat& image) {
  if (tag.empty() || image.empty()) {
    return {};
  }
  if (tag.cols <= image.cols && tag.rows <= image.rows) {
    return tag;
  }

  const double scale = std::min(
      static_cast<double>(image.cols) / static_cast<double>(tag.cols),
      static_cast<double>(image.rows) / static_cast<double>(tag.rows));
  const int width = std::max(1, std::min(image.cols, cvRound(tag.cols * scale)));
  const int height =
      std::max(1, std::min(image.rows, cvRound(tag.rows * scale)));
  cv::Mat resized;
  cv::resize(tag, resized, {width, height}, 0.0, 0.0, cv::INTER_AREA);
  return resized;
}

void BlitTag(cv::Mat& image, const cv::Mat& tag, int requested_x,
             int requested_y) {
  const cv::Mat fitted = FitTagToImage(tag, image);
  if (fitted.empty()) {
    return;
  }
  const int x = Clamp(requested_x, 0, image.cols - fitted.cols);
  const int y = Clamp(requested_y, 0, image.rows - fitted.rows);
  fitted.copyTo(image(cv::Rect(x, y, fitted.cols, fitted.rows)));
}

cv::Mat MakeFocusBadge(const std::string& text, const cv::Scalar& color) {
  const TextMetrics metrics = MeasureText(text, kFontScale);
  const int width = metrics.size.width + 2 * kTagPaddingX;
  const int height = metrics.size.height + 2 * kTagPaddingY;
  cv::Mat tag(height, width, CV_8UC3, color);
  cv::putText(tag, text,
              {kTagPaddingX, kTagPaddingY + metrics.size.height}, kFontFace,
              kFontScale, cv::Scalar(255, 255, 255), kFontThickness,
              cv::LINE_AA);
  return tag;
}

cv::Mat MakeTrackTag(const tracking::TrackView& track, int image_width,
                     const cv::Scalar& color) {
  if (image_width <= 0) {
    return {};
  }
  const std::string label = FormatTrackLabel(track);
  const std::string score = FormatTrackScore(track);
  const TextMetrics normal_label = MeasureText(label, kFontScale);
  const TextMetrics normal_score = MeasureText(score, kFontScale);
  const int normal_line_height =
      std::max(normal_label.size.height + normal_label.baseline,
               normal_score.size.height + normal_score.baseline);
  const int normal_width = normal_label.size.width + normal_score.size.width +
                           2 * kTagPaddingX + 4 * kSeparatorRadius + 8;
  const int normal_height = normal_line_height + 2 * kTagPaddingY;
  if (normal_width <= image_width) {
    cv::Mat tag(normal_height, normal_width, CV_8UC3, color);
    const int baseline_y = kTagPaddingY + normal_label.size.height;
    const int label_x = kTagPaddingX;
    cv::putText(tag, label, {label_x, baseline_y}, kFontFace, kFontScale,
                cv::Scalar(255, 255, 255), kFontThickness, cv::LINE_AA);
    const int separator_x =
        label_x + normal_label.size.width + 2 * kSeparatorRadius + 2;
    cv::circle(tag, {separator_x, normal_height / 2}, kSeparatorRadius,
               cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_AA);
    cv::putText(tag, score,
                {separator_x + 2 * kSeparatorRadius + 4, baseline_y},
                kFontFace, kFontScale, cv::Scalar(255, 255, 255),
                kFontThickness, cv::LINE_AA);
    return tag;
  }

  double compact_scale = kFontScale;
  TextMetrics compact_label = normal_label;
  TextMetrics compact_score = normal_score;
  int compact_width = normal_width;
  while (compact_scale > 0.05 && compact_width > image_width) {
    compact_scale -= 0.01;
    compact_label = MeasureText(label, compact_scale);
    compact_score = MeasureText(score, compact_scale);
    compact_width = std::max(compact_label.size.width,
                             2 * kSeparatorRadius + 4 + compact_score.size.width) +
                    2 * kTagPaddingX;
  }

  const int line_height =
      std::max(compact_label.size.height + compact_label.baseline,
               compact_score.size.height + compact_score.baseline);
  const int width = std::max(compact_label.size.width,
                             2 * kSeparatorRadius + 4 + compact_score.size.width) +
                    2 * kTagPaddingX;
  const int height = 2 * line_height + 2 * kTagPaddingY + 3;
  cv::Mat tag(height, width, CV_8UC3, color);
  const int label_baseline = kTagPaddingY + compact_label.size.height;
  cv::putText(tag, label, {kTagPaddingX, label_baseline}, kFontFace,
              compact_scale, cv::Scalar(255, 255, 255), kFontThickness,
              cv::LINE_AA);
  const int second_row_top = kTagPaddingY + line_height + 3;
  const int separator_x = kTagPaddingX + kSeparatorRadius;
  cv::circle(tag, {separator_x, second_row_top + line_height / 2},
             kSeparatorRadius, cv::Scalar(255, 255, 255), cv::FILLED,
             cv::LINE_AA);
  cv::putText(tag, score,
              {separator_x + kSeparatorRadius + 4,
               second_row_top + compact_score.size.height},
              kFontFace, compact_scale, cv::Scalar(255, 255, 255),
              kFontThickness, cv::LINE_AA);
  return tag;
}

void DrawFocusBadge(cv::Mat& image, const std::string& text, int center_x,
                    int requested_y, const cv::Scalar& color) {
  const cv::Mat tag = FitTagToImage(MakeFocusBadge(text, color), image);
  if (tag.empty()) {
    return;
  }
  const int x = Clamp(center_x - tag.cols / 2, 0, image.cols - tag.cols);
  const int y = Clamp(requested_y, 0, image.rows - tag.rows);
  tag.copyTo(image(cv::Rect(x, y, tag.cols, tag.rows)));
}

}  // namespace

cv::Scalar TrackColor(int track_id) {
  const int index = track_id > 0 ? track_id - 1 : 0;
  return TrackColors()[static_cast<std::size_t>(index) % TrackColors().size()];
}

std::string FormatTrackLabel(const tracking::TrackView& track) {
  return "PERSON #" + std::to_string(track.id);
}

std::string FormatTrackScore(const tracking::TrackView& track) {
  std::ostringstream output;
  output << std::fixed << std::setprecision(2) << track.score;
  return output.str();
}

std::string FormatFocusDistance(float relative, double meters_per_unit) {
  std::ostringstream output;
  if (meters_per_unit > 0.0) {
    output << std::fixed << std::setprecision(1)
           << static_cast<double>(relative) * meters_per_unit << "m";
  } else {
    output << std::fixed << std::setprecision(2) << relative;
  }
  return output.str();
}

void DrawTrackBox(cv::Mat& image, const tracking::TrackView& track) {
  if (image.empty()) {
    return;
  }

  const cv::Rect track_box(cvRound(track.box.x), cvRound(track.box.y),
                           cvRound(track.box.width), cvRound(track.box.height));
  const cv::Rect image_bounds(0, 0, image.cols, image.rows);
  const cv::Rect visible_box = track_box & image_bounds;
  const cv::Scalar color = TrackColor(track.id);
  if (visible_box.width > 0 && visible_box.height > 0) {
    cv::rectangle(image, visible_box, color, 3, cv::LINE_AA);
  }

  const cv::Mat tag = MakeTrackTag(track, image.cols, color);
  if (tag.empty()) {
    return;
  }
  const cv::Mat fitted = FitTagToImage(tag, image);
  const cv::Point origin = TagOrigin(image, track_box, fitted.cols, fitted.rows);
  BlitTag(image, tag, origin.x, origin.y);
}

void DrawFocusMarker(cv::Mat& image, float relative, double meters_per_unit) {
  if (image.empty()) {
    return;
  }

  const cv::Point center(image.cols / 2, image.rows / 2);
  const cv::Scalar color = TrackColor(1);
  cv::circle(image, center, 18, color, 2, cv::LINE_AA);
  cv::circle(image, center, 10, color, 2, cv::LINE_AA);
  cv::line(image, {center.x - 26, center.y}, {center.x + 26, center.y}, color,
           2, cv::LINE_AA);
  cv::line(image, {center.x, center.y - 26}, {center.x, center.y + 26}, color,
           2, cv::LINE_AA);

  DrawFocusBadge(image, "FOCUS DISTANCE", center.x, center.y - 54, color);
  DrawFocusBadge(image, FormatFocusDistance(relative, meters_per_unit), center.x,
                 center.y + 34, color);
}

}  // namespace ui
