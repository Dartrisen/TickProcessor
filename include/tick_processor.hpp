#pragma once

#include "market_features.hpp"
#include "tick.hpp"

#include <cmath>
#include <deque>
#include <functional>
#include <numeric>

class TickProcessor
{
public:
    using FeatureCallback = std::function<void(const Tick&, const DerivedFeatures&)>;

    explicit TickProcessor(std::size_t vol_window = 20) : vol_window_(vol_window) {}

    void register_callback(FeatureCallback cb)
    {
        callbacks_.push_back(std::move(cb));
    }

    void process(const Tick& tick)
    {
        DerivedFeatures feats{};

        if (has_prev_tick_)
        {
            feats.log_return    = std::log(tick.close / prev_tick_.close);
            feats.simple_return = (tick.close / prev_tick_.close) - 1.0;

            // Maintain rolling window for volatility calculation
            returns_window_.push_back(feats.log_return);
            if (returns_window_.size() > vol_window_)
            {
                returns_window_.pop_front();
            }

            feats.rolling_volatility = calculate_volatility();
        }

        has_prev_tick_ = true;
        prev_tick_     = tick;

        // Dispatch features to listeners/models
        for (const auto& cb : callbacks_)
        {
            cb(tick, feats);
        }
    }

private:
    [[nodiscard]] double calculate_volatility() const
    {
        if (returns_window_.empty())
            return 0.0;

        double mean = std::accumulate(returns_window_.begin(), returns_window_.end(), 0.0) / returns_window_.size();

        double sq_sum = 0.0;
        for (double r : returns_window_)
        {
            sq_sum += (r - mean) * (r - mean);
        }
        return std::sqrt(sq_sum / returns_window_.size());
    }

    std::size_t                  vol_window_;
    bool                         has_prev_tick_{false};
    Tick                         prev_tick_{};
    std::deque<double>           returns_window_;
    std::vector<FeatureCallback> callbacks_;
};
