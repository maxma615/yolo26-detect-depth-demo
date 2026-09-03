#pragma once

#include <vector>

namespace tracking::internal {

// Returns one detection-column index per input row. Rectangular inputs are
// square-padded with padding_cost; rows assigned to a padded column return -1.
std::vector<int> SolveDeterministicAssignment(
    const std::vector<std::vector<double>>& costs, double padding_cost);

}  // namespace tracking::internal
