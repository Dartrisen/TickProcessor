#pragma once

#include <cstdint>
#include <string>

struct Tick
{
    std::string   datetime;
    double        open{0.0};
    double        high{0.0};
    double        low{0.0};
    double        close{0.0};
    std::uint64_t volume{0};
    std::string   source;

    [[nodiscard]] constexpr double typical_price() const noexcept
    {
        return (high + low + close) / 3.0;
    }
};
