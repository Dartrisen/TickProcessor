#pragma once

#include "quant/core/types.hpp"
#include "quant/features/online_stats.hpp"

#include <cmath>
#include <cstddef>

template <std::size_t VolatilityWindow = 20>
class OnlineFeatureEngine
{
public:
    constexpr OnlineFeatureEngine() = default;

    // HOT PATH: Process tick deterministically with zero allocations
    [[nodiscard]] bool process(const Tick& tick, DerivedFeatures& out_features) noexcept
    {
        const double current_vol = static_cast<double>(tick.volume);

        if (!has_prev_tick_)
        {
            prev_tick_     = tick;
            has_prev_tick_ = true;
            if (current_vol > 0.0)
            {
                volume_stats_.push(current_vol);
            }
            return false;
        }

        out_features.log_return    = std::log(tick.close / prev_tick_.close);
        out_features.simple_return = (tick.close / prev_tick_.close) - 1.0;

        return_stats_.push(out_features.log_return);
        out_features.rolling_volatility = return_stats_.stddev();

        // Compute relative volume (RVOL) against trailing rolling mean volume
        // before pushing current bar volume to avoid self-dilution on volume spikes.
        const double baseline_vol = volume_stats_.mean();
        if (current_vol > 0.0 && baseline_vol > 0.0)
        {
            out_features.volume_ratio = current_vol / baseline_vol;
        }
        else if (prev_tick_.volume > 0)
        {
            out_features.volume_ratio = current_vol / static_cast<double>(prev_tick_.volume);
        }
        else
        {
            out_features.volume_ratio = 1.0;
        }

        if (current_vol > 0.0)
        {
            volume_stats_.push(current_vol);
        }

        prev_tick_ = tick;
        return true;
    }

private:
    bool                                 has_prev_tick_{false};
    Tick                                 prev_tick_{};
    RollingWindowStats<VolatilityWindow> return_stats_{};
    RollingWindowStats<VolatilityWindow> volume_stats_{};
};
