#pragma once

#include "quant/models/interfaces.hpp"

#include <cmath>
#include <string>

enum class OrderSignal
{
    Hold,
    Buy, // target long
    Sell // target short
};

struct SignalDecision
{
    OrderSignal signal{OrderSignal::Hold};
    bool        var_passed{false};
    bool        entropy_passed{false};
    bool        return_passed{false};
    bool        risk_veto{false};
    std::string reason;
};

struct RiskLimits
{
    // Positive loss magnitude, e.g. 0.01 = maximum tolerated 1% 95%-VaR.
    double max_var_95{0.01};

    // Entropy is uncertainty. Higher entropy means a flatter, less informative forecast.
    // For 16 bins, uniform entropy is log2(16) = 4 bits.
    double max_entropy{3.20};

    double min_expected_return{0.00005};
    double min_bullish_prob{0.52};
    double max_bearish_prob{0.48};
};

class RiskEngine
{
public:
    explicit RiskEngine(RiskLimits limits = {}) : limits_(limits) {}

    [[nodiscard]] SignalDecision evaluate(const QuantForecast& forecast) const noexcept
    {
        SignalDecision decision;

        // var_95 is explicitly stored as a positive loss magnitude.
        decision.var_passed = forecast.var_95 <= limits_.max_var_95;

        // High entropy = high uncertainty, so reject forecasts above the limit.
        decision.entropy_passed = forecast.distribution_entropy <= limits_.max_entropy;

        if (!decision.var_passed)
        {
            decision.signal    = OrderSignal::Hold;
            decision.risk_veto = true;
            decision.reason    = "VETO: VaR(95%) loss " + std::to_string(forecast.var_95 * 100.0) + "% exceeds max loss limit " + std::to_string(limits_.max_var_95 * 100.0) + "%";
            return decision;
        }

        if (!decision.entropy_passed)
        {
            decision.signal    = OrderSignal::Hold;
            decision.risk_veto = true;
            decision.reason =
                    "VETO: Forecast entropy (" + std::to_string(forecast.distribution_entropy) + ") exceeds max uncertainty (" + std::to_string(limits_.max_entropy) + ")";
            return decision;
        }

        const bool strong_alpha_buy  = (forecast.expected_return > limits_.min_expected_return) && (forecast.prob_positive_return >= limits_.min_bullish_prob);
        const bool strong_alpha_sell = (forecast.expected_return < -limits_.min_expected_return) && (forecast.prob_positive_return <= limits_.max_bearish_prob);

        if (strong_alpha_buy)
        {
            decision.signal        = OrderSignal::Buy;
            decision.return_passed = true;
            decision.reason        = "BUY PASSED: E[r] (" + std::to_string(forecast.expected_return * 100.0) + "%) > hurdle and P(r>0) (" +
                              std::to_string(forecast.prob_positive_return * 100.0) + "%) >= " + std::to_string(limits_.min_bullish_prob * 100.0) + "%";
        }
        else if (strong_alpha_sell)
        {
            decision.signal        = OrderSignal::Sell;
            decision.return_passed = true;
            decision.reason        = "SELL/SHORT PASSED: E[r] (" + std::to_string(forecast.expected_return * 100.0) + "%) < -hurdle and P(r>0) (" +
                              std::to_string(forecast.prob_positive_return * 100.0) + "%) <= " + std::to_string(limits_.max_bearish_prob * 100.0) + "%";
        }
        else
        {
            decision.signal = OrderSignal::Hold;
            decision.reason = "HOLD: Alpha expectation neutral (E[r]=" + std::to_string(forecast.expected_return * 100.0) + "%)";
        }

        return decision;
    }

private:
    RiskLimits limits_;
};
