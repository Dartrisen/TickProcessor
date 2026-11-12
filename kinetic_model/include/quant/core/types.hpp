#pragma once

#include <cstdint>
#include <string>

enum class PositionSide
{
    Flat,
    Long,
    Short
};

enum class OrderSignal
{
    Hold,
    Buy,  // Target long exposure
    Sell, // Target short exposure
    Close // Exit existing position to flat (alpha fade / time stop)
};

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

struct DerivedFeatures
{
    double log_return{0.0};
    double simple_return{0.0};
    double rolling_volatility{0.0};
    double volume_ratio{0.0};
};