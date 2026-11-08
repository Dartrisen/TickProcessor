#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dataset.hpp"

// ============================================================
// CSV Reader
// ============================================================

class CsvReader {
public:
  static MarketDataset load(const std::string &filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
      throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;

    if (!std::getline(file, line)) {
      throw std::runtime_error("Empty CSV file");
    }

    const auto header = parse_line(line);

    std::unordered_map<std::string, std::size_t> columns;

    for (std::size_t i = 0; i < header.size(); ++i) {
      columns[std::string(trim(header[i]))] = i;
    }

    const auto index = [&](const std::string &name) {
      const auto it = columns.find(name);

      if (it == columns.end()) {
        throw std::runtime_error("Missing CSV column: " + name);
      }

      return it->second;
    };

    const auto datetime = index("datetime");
    const auto open = index("Open");
    const auto high = index("High");
    const auto low = index("Low");
    const auto close = index("Close");
    const auto volume = index("Volume");
    const auto source = index("source");

    MarketDataset dataset;

    std::size_t line_number = 1;

    while (std::getline(file, line)) {
      ++line_number;

      if (line.empty()) {
        continue;
      }

      try {
        const auto fields = parse_line(line);

        const auto get = [&](std::size_t i) -> std::string_view {
          if (i >= fields.size()) {
            throw std::runtime_error("Missing field");
          }

          return fields[i];
        };

        Tick bar;

        bar.datetime = std::string(get(datetime));
        bar.open = parse_double(get(open));
        bar.high = parse_double(get(high));
        bar.low = parse_double(get(low));
        bar.close = parse_double(get(close));
        bar.volume = parse_volume(get(volume));
        bar.source = std::string(get(source));

        validate(bar);

        dataset.add(std::move(bar));
      } catch (const std::exception &e) {
        throw std::runtime_error("CSV error at line " +
                                 std::to_string(line_number) + ": " + e.what());
      }
    }

    return dataset;
  }

private:
  // --------------------------------------------------------
  // Remove surrounding whitespace
  // --------------------------------------------------------

  static std::string_view trim(std::string_view value) {
    constexpr std::string_view whitespace = " \t\r\n";

    const auto first = value.find_first_not_of(whitespace);

    if (first == std::string_view::npos) {
      return {};
    }

    const auto last = value.find_last_not_of(whitespace);

    return value.substr(first, last - first + 1);
  }

  // --------------------------------------------------------
  // Parse CSV line (supports quoted fields)
  // --------------------------------------------------------

  static std::vector<std::string> parse_line(const std::string &line) {

    std::vector<std::string> fields;

    std::string field;
    bool quoted = false;

    for (std::size_t i = 0; i < line.size(); ++i) {
      const char c = line[i];

      if (c == '"') {
        if (quoted && i + 1 < line.size() && line[i + 1] == '"') {

          field += '"';
          ++i;
        } else {
          quoted = !quoted;
        }
      } else if (c == ',' && !quoted) {
        fields.push_back(std::move(field));
        field.clear();
      } else {
        field += c;
      }
    }

    if (quoted) {
      throw std::runtime_error("Unterminated quoted field");
    }

    fields.push_back(std::move(field));

    return fields;
  }

  // --------------------------------------------------------
  // Numeric parsing
  // --------------------------------------------------------

  static double parse_double(std::string_view value) {
    const std::string str(trim(value));

    if (str.empty()) {
      throw std::runtime_error("Empty numeric field");
    }

    std::size_t pos = 0;
    const double result = std::stod(str, &pos);

    if (pos != str.size() || !std::isfinite(result)) {
      throw std::runtime_error("Invalid floating-point value");
    }

    return result;
  }

  static std::uint64_t parse_volume(std::string_view value) {
    const std::string str(trim(value));

    if (str.empty() || str.front() == '-') {
      throw std::runtime_error("Invalid volume");
    }

    std::size_t pos = 0;
    const auto result = std::stoull(str, &pos);

    if (pos != str.size()) {
      throw std::runtime_error("Invalid volume");
    }

    return result;
  }

  // --------------------------------------------------------
  // Validate OHLC
  // --------------------------------------------------------

  static void validate(const Tick &bar) {
    if (bar.datetime.empty()) {
      throw std::runtime_error("Empty datetime");
    }

    if (bar.open <= 0 || bar.high <= 0 || bar.low <= 0 || bar.close <= 0) {

      throw std::runtime_error("Invalid price");
    }

    if (bar.high < std::max(bar.open, bar.close) ||
        bar.low > std::min(bar.open, bar.close) || bar.high < bar.low) {

      throw std::runtime_error("Invalid OHLC values");
    }
  }
};
