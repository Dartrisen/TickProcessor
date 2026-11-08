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

#include "include/csv_reader.hpp"
#include "include/market_features.hpp"

// ============================================================
// Main
// ============================================================

int main() {

  try {

    // Load historical data
    auto market =
        CsvReader::load("/Users/cdg546/Downloads/data/SPY_clean_1min.csv");

    std::cout << "Loaded observations: " << market.size() << "\n\n";

    // Inspect first observations
    const std::size_t count = std::min<std::size_t>(5, market.size());

    for (std::size_t i = 0; i < count; ++i) {

      const auto &bar = market[i];

      std::cout << bar.datetime << " | Open: " << bar.open
                << " | High: " << bar.high << " | Low: " << bar.low
                << " | Close: " << bar.close << " | Volume: " << bar.volume
                << " | Source: " << bar.source << "\n";
    }

    // Extract features
    auto returns = MarketFeatures::log_returns(market);

    std::cout << "\nLog returns: " << returns.size() << "\n";

    const std::size_t return_count = std::min<std::size_t>(5, returns.size());

    for (std::size_t i = 0; i < return_count; ++i) {
      std::cout << returns[i] << "\n";
    }

  } catch (const std::exception &e) {

    std::cerr << "Error: " << e.what() << "\n";

    return 1;
  }

  return 0;
}
