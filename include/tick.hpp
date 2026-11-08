#include <cstdint>
#include <string>

// ============================================================
// Market data
// ============================================================

struct Tick {
  std::string datetime;

  double open{};
  double high{};
  double low{};
  double close{};

  std::uint64_t volume{};

  std::string source;
};