#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

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

struct FixedTimestamp
{
    char data[32]{};

    constexpr FixedTimestamp() noexcept = default;

    constexpr void assign(std::string_view sv) noexcept
    {
        const std::size_t len = std::min(sv.size(), sizeof(data) - 1);
        for (std::size_t i = 0; i < len; ++i)
        {
            data[i] = sv[i];
        }
        data[len] = '\0';
    }

    [[nodiscard]] constexpr std::string_view view() const noexcept
    {
        return std::string_view(data);
    }

    [[nodiscard]] constexpr bool empty() const noexcept
    {
        return data[0] == '\0';
    }
};

struct Tick
{
    FixedTimestamp datetime{};
    double         open{0.0};
    double         high{0.0};
    double         low{0.0};
    double         close{0.0};
    std::uint64_t  volume{0};

    [[nodiscard]] constexpr double typical_price() const noexcept
    {
        return (high + low + close) / 3.0;
    }
};

static_assert(std::is_trivially_copyable_v<Tick>, "Tick must be trivially copyable for lock-free ring buffer transport");

struct DerivedFeatures
{
    double log_return{0.0};
    double simple_return{0.0};
    double rolling_volatility{0.0};
    double volume_ratio{0.0};
};