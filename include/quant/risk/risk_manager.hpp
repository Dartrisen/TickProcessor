#pragma once

#include "quant/models/interfaces.hpp"

enum class OrderSignal
{
    Hold,
    Buy,
    Sell
};

struct RiskLimits
{
    double max_var_95{0.05};            // 5% max log-loss threshold
    double min_entropy{0.5};            // Minimum entropy threshold
    double min_expected_return{0.0001}; // Alpha threshold
};

class RiskEngine
{
public:
    explicit RiskEngine(RiskLimits limits = {}) : limits_(limits) {}

    [[nodiscard]] OrderSignal evaluate(const QuantForecast& forecast) const noexcept
    {
        if (std::abs(forecast.var_95) > limits_.max_var_95)
        {
            return OrderSignal::Hold;
        }

        if (forecast.distribution_entropy < limits_.min_entropy)
        {
            return OrderSignal::Hold;
        }

        if (forecast.expected_return > limits_.min_expected_return && forecast.prob_positive_return > 0.55)
        {
            return OrderSignal::Buy;
        }
        else if (forecast.expected_return < -limits_.min_expected_return && forecast.prob_positive_return < 0.45)
        {
            return OrderSignal::Sell;
        }

        return OrderSignal::Hold;
    }

private:
    RiskLimits limits_;
};
