#pragma once

#include "quant/core/types.hpp"
#include "quant/features/online_stats.hpp"

#include <cmath>

template <std::size_t VolatilityWindow = 20>
class OnlineFeatureEngine
{
public:
    constexpr OnlineFeatureEngine() = default;

    // HOT PATH: Process tick deterministically
    [[nodiscard]] bool process(const Tick& tick, DerivedFeatures& out_features) noexcept
    {
        if (!has_prev_tick_)
        {
            prev_tick_     = tick;
            has_prev_tick_ = true;
            return false;
        }

        out_features.log_return    = std::log(tick.close / prev_tick_.close);
        out_features.simple_return = (tick.close / prev_tick_.close) - 1.0;

        stats_.push(out_features.log_return);
        out_features.rolling_volatility = stats_.stddev();

        out_features.volume_ratio = (prev_tick_.volume > 0) ? (static_cast<double>(tick.volume) / static_cast<double>(prev_tick_.volume)) : 1.0;

        prev_tick_ = tick;
        return true;
    }

private:
    bool                                 has_prev_tick_{false};
    Tick                                 prev_tick_{};
    RollingWindowStats<VolatilityWindow> stats_{};
};
