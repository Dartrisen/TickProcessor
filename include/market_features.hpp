#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

// ============================================================
// Market feature extraction
// ============================================================

class MarketFeatures {
public:
  // Logarithmic returns:
  //
  // r[i] = log(P[i] / P[i-1])

  static std::vector<double> log_returns(const MarketDataset &data) {

    std::vector<double> returns;

    if (data.size() < 2) {
      return returns;
    }

    returns.reserve(data.size() - 1);

    for (std::size_t i = 1; i < data.size(); ++i) {

      const double p0 = data[i - 1].close;
      const double p1 = data[i].close;

      returns.push_back(std::log(p1 / p0));
    }

    return returns;
  }

  // Simple returns:
  //
  // r[i] = P[i] / P[i-1] - 1

  static std::vector<double> simple_returns(const MarketDataset &data) {

    std::vector<double> returns;

    if (data.size() < 2) {
      return returns;
    }

    returns.reserve(data.size() - 1);

    for (std::size_t i = 1; i < data.size(); ++i) {

      const double p0 = data[i - 1].close;
      const double p1 = data[i].close;

      returns.push_back(p1 / p0 - 1.0);
    }

    return returns;
  }
};
