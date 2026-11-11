#pragma once

#include "quant/models/interfaces.hpp"

#include <string>

enum class OrderSignal
{
    Hold,
    Buy,
    Sell
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
    double max_var_95{0.05};             // 5% max log-loss VaR threshold
    double min_entropy{0.5};             // Minimum predictive certainty threshold
    double min_expected_return{0.00005}; // Alpha hurdle rate (+0.005% return)
    double min_bullish_prob{0.52};       // Minimum bullish probability for BUY
    double max_bearish_prob{0.48};       // Maximum probability threshold for SELL
};

class RiskEngine
{
public:
    explicit RiskEngine(RiskLimits limits = {}) : limits_(limits) {}

    [[nodiscard]] SignalDecision evaluate(const QuantForecast& forecast) const noexcept
    {
        SignalDecision decision;

        decision.var_passed     = std::abs(forecast.var_95) <= limits_.max_var_95;
        decision.entropy_passed = forecast.distribution_entropy >= limits_.min_entropy;

        if (!decision.var_passed)
        {
            decision.signal    = OrderSignal::Hold;
            decision.risk_veto = true;
            decision.reason    = "VETO: VaR(95%) " + std::to_string(forecast.var_95 * 100.0) + "% exceeds max limit " + std::to_string(limits_.max_var_95 * 100.0) + "%";
            return decision;
        }

        if (!decision.entropy_passed)
        {
            decision.signal    = OrderSignal::Hold;
            decision.risk_veto = true;
            decision.reason    = "VETO: Model entropy (" + std::to_string(forecast.distribution_entropy) + ") below certainty floor (" + std::to_string(limits_.min_entropy) + ")";
            return decision;
        }

        const bool strong_alpha_buy  = (forecast.expected_return > limits_.min_expected_return) && (forecast.prob_positive_return >= limits_.min_bullish_prob);
        const bool strong_alpha_sell = (forecast.expected_return < -limits_.min_expected_return) && (forecast.prob_positive_return <= limits_.max_bearish_prob);

        if (strong_alpha_buy)
        {
            decision.signal        = OrderSignal::Buy;
            decision.return_passed = true;
            decision.reason        = "BUY PASSED: E[r] (" + std::to_string(forecast.expected_return * 100.0) + "%) > Hurdle & P(r>0) (" +
                              std::to_string(forecast.prob_positive_return * 100.0) + "%) >= " + std::to_string(limits_.min_bullish_prob * 100.0) + "%";
        }
        else if (strong_alpha_sell)
        {
            decision.signal        = OrderSignal::Sell;
            decision.return_passed = true;
            decision.reason        = "SELL PASSED: E[r] (" + std::to_string(forecast.expected_return * 100.0) + "%) < Hurdle & P(r>0) (" +
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
