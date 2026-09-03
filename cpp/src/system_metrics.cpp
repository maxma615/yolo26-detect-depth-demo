#include "system_metrics.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

namespace system_metrics {
namespace {

std::optional<std::uint64_t> ParseUnsigned(const std::string& token) {
  if (token.empty()) return std::nullopt;
  std::uint64_t value = 0;
  for (const char ch : token) {
    if (ch < '0' || ch > '9') return std::nullopt;
    const std::uint64_t digit = static_cast<std::uint64_t>(ch - '0');
    if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) {
      return std::nullopt;
    }
    value = value * 10 + digit;
  }
  return value;
}

double ClampPercent(double value) {
  if (!std::isfinite(value)) return 0.0;
  return std::max(0.0, std::min(100.0, value));
}

std::optional<std::string> ReadFile(const std::string& path) {
  std::ifstream input(path);
  if (!input) return std::nullopt;
  std::ostringstream contents;
  contents << input.rdbuf();
  if (input.bad()) return std::nullopt;
  return contents.str();
}

}  // namespace

std::optional<CpuTimes> ParseProcStat(const std::string& text) {
  std::istringstream input(text);
  std::string line;
  if (!std::getline(input, line)) return std::nullopt;

  std::istringstream fields(line);
  std::string label;
  if (!(fields >> label) || label != "cpu") return std::nullopt;

  std::vector<std::uint64_t> values;
  std::string token;
  while (fields >> token) {
    const auto value = ParseUnsigned(token);
    if (!value.has_value()) return std::nullopt;
    values.push_back(*value);
  }
  if (values.size() < 4) return std::nullopt;

  std::uint64_t total = 0;
  for (const std::uint64_t value : values) {
    if (total > std::numeric_limits<std::uint64_t>::max() - value) {
      return std::nullopt;
    }
    total += value;
  }
  std::uint64_t idle = values[3];
  if (values.size() > 4) {
    if (idle > std::numeric_limits<std::uint64_t>::max() - values[4]) {
      return std::nullopt;
    }
    idle += values[4];
  }
  return CpuTimes{total, idle};
}

std::optional<double> CpuUtilizationPercent(
    const std::optional<CpuTimes>& previous, const CpuTimes& current) {
  if (!previous.has_value() || current.total <= previous->total ||
      current.idle < previous->idle) {
    return std::nullopt;
  }
  const std::uint64_t total_delta = current.total - previous->total;
  const std::uint64_t idle_delta = current.idle - previous->idle;
  const double busy_fraction =
      1.0 - static_cast<double>(idle_delta) / static_cast<double>(total_delta);
  return ClampPercent(busy_fraction * 100.0);
}

std::optional<double> ParseMeminfoUtilizationPercent(const std::string& text) {
  std::optional<std::uint64_t> total;
  std::optional<std::uint64_t> available;
  std::istringstream input(text);
  std::string line;
  while (std::getline(input, line)) {
    const std::size_t colon = line.find(':');
    if (colon == std::string::npos) continue;
    const std::string key = line.substr(0, colon);
    if (key != "MemTotal" && key != "MemAvailable") continue;
    std::istringstream value_fields(line.substr(colon + 1));
    std::string token;
    if (!(value_fields >> token)) return std::nullopt;
    const auto value = ParseUnsigned(token);
    if (!value.has_value()) return std::nullopt;
    if (key == "MemTotal")
      total = *value;
    else
      available = *value;
  }
  if (!total.has_value() || *total == 0 || !available.has_value()) {
    return std::nullopt;
  }
  const double used_fraction =
      1.0 - static_cast<double>(*available) / static_cast<double>(*total);
  return ClampPercent(used_fraction * 100.0);
}

Sampler::Sampler() : Sampler(ReadFile) {}

Sampler::Sampler(Reader reader) : reader_(std::move(reader)) {}

Metrics Sampler::Sample() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (reader_) {
    const auto stat_text = reader_("/proc/stat");
    if (stat_text.has_value()) {
      const auto current = ParseProcStat(*stat_text);
      if (current.has_value()) {
        const auto utilization =
            CpuUtilizationPercent(previous_cpu_, *current);
        previous_cpu_ = current;
        if (utilization.has_value()) {
          last_.cpu_util_pct = ClampPercent(*utilization);
        }
      }
    }

    const auto meminfo_text = reader_("/proc/meminfo");
    if (meminfo_text.has_value()) {
      const auto utilization =
          ParseMeminfoUtilizationPercent(*meminfo_text);
      if (utilization.has_value()) {
        last_.memory_util_pct = ClampPercent(*utilization);
      }
    }
  }

  last_.cpu_util_pct = ClampPercent(last_.cpu_util_pct);
  last_.memory_util_pct = ClampPercent(last_.memory_util_pct);
  return last_;
}

}  // namespace system_metrics
