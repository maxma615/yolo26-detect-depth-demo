#include "system_metrics.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
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

void CheckNear(double actual, double expected, double tolerance = 0.001) {
  CHECK(std::fabs(actual - expected) <= tolerance);
}

void TestFirstCpuSampleHasNoDelta() {
  const auto current = system_metrics::ParseProcStat(
      "cpu  100 20 30 400 10 5 3 2 0 0\n");
  CHECK(current.has_value());
  CHECK(!system_metrics::CpuUtilizationPercent(std::nullopt, *current)
             .has_value());
}

void TestCpuUtilizationUsesBusyDelta() {
  const auto previous =
      system_metrics::ParseProcStat("cpu  100 0 50 850 0 0 0 0\n");
  const auto current =
      system_metrics::ParseProcStat("cpu  130 0 70 900 0 0 0 0\n");
  CHECK(previous.has_value());
  CHECK(current.has_value());
  const auto utilization =
      system_metrics::CpuUtilizationPercent(previous, *current);
  CHECK(utilization.has_value());
  CheckNear(*utilization, 50.0);
}

void TestCpuZeroDeltaIsUnavailable() {
  const auto sample =
      system_metrics::ParseProcStat("cpu  100 0 50 850 0 0 0 0\n");
  CHECK(sample.has_value());
  CHECK(!system_metrics::CpuUtilizationPercent(sample, *sample).has_value());
}

void TestMemoryUsesMemAvailable() {
  const auto utilization = system_metrics::ParseMeminfoUtilizationPercent(
      "MemTotal:       1000 kB\n"
      "MemFree:         100 kB\n"
      "MemAvailable:    250 kB\n"
      "Buffers:          50 kB\n");
  CHECK(utilization.has_value());
  CheckNear(*utilization, 75.0);
}

void TestMalformedInputsAreRejected() {
  CHECK(!system_metrics::ParseProcStat("intr 1 2 3\n").has_value());
  CHECK(!system_metrics::ParseProcStat("cpu 1 two 3 4\n").has_value());
  CHECK(!system_metrics::ParseMeminfoUtilizationPercent(
             "MemTotal: nope kB\nMemAvailable: 10 kB\n")
             .has_value());
  CHECK(!system_metrics::ParseMeminfoUtilizationPercent(
             "MemTotal: 100 kB\nMemFree: 10 kB\n")
             .has_value());
}

void TestSamplerKeepsFiniteDefaultsWhenFilesAreUnavailable() {
  system_metrics::Sampler sampler(
      [](const std::string&) -> std::optional<std::string> {
        return std::nullopt;
      });
  const system_metrics::Metrics metrics = sampler.Sample();
  CHECK(std::isfinite(metrics.cpu_util_pct));
  CHECK(std::isfinite(metrics.memory_util_pct));
  CheckNear(metrics.cpu_util_pct, 0.0);
  CheckNear(metrics.memory_util_pct, 0.0);
}

void TestSamplerComputesDeltaAndClampsMemory() {
  int stat_reads = 0;
  system_metrics::Sampler sampler(
      [&stat_reads](const std::string& path) -> std::optional<std::string> {
        if (path == "/proc/stat") {
          ++stat_reads;
          if (stat_reads == 1) return "cpu  100 0 50 850 0 0 0 0\n";
          return "cpu  130 0 70 900 0 0 0 0\n";
        }
        if (path == "/proc/meminfo") {
          return "MemTotal: 1000 kB\nMemAvailable: 1200 kB\n";
        }
        return std::nullopt;
      });
  const system_metrics::Metrics first = sampler.Sample();
  const system_metrics::Metrics second = sampler.Sample();
  CheckNear(first.cpu_util_pct, 0.0);
  CheckNear(second.cpu_util_pct, 50.0);
  CheckNear(second.memory_util_pct, 0.0);
  CHECK(second.cpu_util_pct >= 0.0 && second.cpu_util_pct <= 100.0);
  CHECK(second.memory_util_pct >= 0.0 && second.memory_util_pct <= 100.0);
}

void TestOverlappingSamplesSerializeReaderAccess() {
  std::atomic<int> active_readers{0};
  std::atomic<int> max_active_readers{0};
  system_metrics::Sampler sampler(
      [&active_readers,
       &max_active_readers](const std::string& path)
          -> std::optional<std::string> {
        const int active = active_readers.fetch_add(1) + 1;
        int observed_max = max_active_readers.load();
        while (active > observed_max &&
               !max_active_readers.compare_exchange_weak(observed_max,
                                                         active)) {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        active_readers.fetch_sub(1);
        if (path == "/proc/stat") return "cpu  100 0 50 850 0 0 0 0\n";
        if (path == "/proc/meminfo")
          return "MemTotal: 1000 kB\nMemAvailable: 250 kB\n";
        return std::nullopt;
      });

  std::vector<std::thread> callers;
  for (int i = 0; i < 8; ++i) {
    callers.emplace_back([&sampler] { sampler.Sample(); });
  }
  for (auto& caller : callers) caller.join();
  CHECK(max_active_readers.load() == 1);
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
  Run("first CPU sample has no delta", TestFirstCpuSampleHasNoDelta, &failures);
  Run("CPU utilization uses busy delta", TestCpuUtilizationUsesBusyDelta,
      &failures);
  Run("CPU zero delta is unavailable", TestCpuZeroDeltaIsUnavailable,
      &failures);
  Run("memory uses MemAvailable", TestMemoryUsesMemAvailable, &failures);
  Run("malformed inputs are rejected", TestMalformedInputsAreRejected,
      &failures);
  Run("sampler keeps finite defaults when files are unavailable",
      TestSamplerKeepsFiniteDefaultsWhenFilesAreUnavailable, &failures);
  Run("sampler computes delta and clamps memory",
      TestSamplerComputesDeltaAndClampsMemory, &failures);
  Run("overlapping samples serialize reader access",
      TestOverlappingSamplesSerializeReaderAccess, &failures);
  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << "All system metrics tests passed\n";
  return 0;
}
