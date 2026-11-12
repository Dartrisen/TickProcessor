#pragma once

#include "quant/models/interfaces.hpp"

#include <algorithm>
#include <cmath>
#include <string>

enum class OrderSignal
{
    Hold,
    Buy, // target long
    Sell // target short
};

struct PortfolioState
{
    double equity{0.0};                // Total portfolio equity (cash + position_shares * price). If <= 0, unit-notional mode is used.
    double position_shares{0.0};       // Current signed share count (+long, -short, 0 flat).
    double current_price{0.0};         // Mark price for exposure valuation (falls back to forecast.current_price if <= 0).
    double target_allocation_pct{1.0}; // Target equity allocation fraction for candidate position sizing in [0, 1].
    bool   allow_pyramiding{false};    // False = single position per side; true = add target size to existing same-side exposure.
};

struct SignalDecision
{
    OrderSignal  signal{OrderSignal::Hold};
    PositionSide current_side{PositionSide::Flat};
    PositionSide candidate_side{PositionSide::Flat};

    bool var_passed{false};
    bool cvar_passed{false};
    bool entropy_passed{false};
    bool return_passed{false};
    bool risk_veto{false};
    bool should_flatten{false}; // True when an open position breaches live portfolio risk limits

    // Exposure & sizing diagnostics
    double current_exposure_notional{0.0};
    double target_exposure_notional{0.0};
    double post_trade_exposure_notional{0.0};
    double post_trade_exposure_weight{0.0};

    // Side-aware unit & portfolio loss contributions
    double unit_var_95{0.0};
    double unit_cvar_95{0.0};
    double portfolio_var_95{0.0};
    double portfolio_cvar_95{0.0};
    double var_95_dollar_loss{0.0};
    double cvar_95_dollar_loss{0.0};

    // Risk-constrained sizing guidance
    double max_allowed_allocation_pct{0.0};
    double recommended_shares{0.0};

    std::string reason;
};

struct RiskLimits
{
    // Positive fractional portfolio loss limits (e.g., 0.01 = max 1.0% of portfolio equity at 95% confidence).
    double max_var_95{0.01};
    double max_cvar_95{0.015};

    // Optional short-side overrides (< 0 falls back to max_var_95 / max_cvar_95).
    double short_max_var_95{-1.0};
    double short_max_cvar_95{-1.0};

    // Entropy is uncertainty. Higher entropy means a flatter, less informative forecast.
    // For 16 bins, uniform entropy is log2(16) = 4 bits.
    double max_entropy{3.20};

    double min_expected_return{0.00005};
    double min_bullish_prob{0.52};
    double max_bearish_prob{0.48};

    [[nodiscard]] constexpr double effective_var_limit(bool is_short) const noexcept
    {
        return (is_short && short_max_var_95 >= 0.0) ? short_max_var_95 : max_var_95;
    }

    [[nodiscard]] constexpr double effective_cvar_limit(bool is_short) const noexcept
    {
        return (is_short && short_max_cvar_95 >= 0.0) ? short_max_cvar_95 : max_cvar_95;
    }
};

class RiskEngine
{
public:
    explicit RiskEngine(RiskLimits limits = {}) : limits_(limits) {}

    [[nodiscard]] SignalDecision evaluate(const QuantForecast& forecast, const PortfolioState& portfolio = {}) const
    {
        SignalDecision decision;

        const double price              = (portfolio.current_price > 0.0) ? portfolio.current_price : forecast.current_price;
        const double alloc              = std::clamp(portfolio.target_allocation_pct, 0.0, 1.0);
        const bool   has_account_equity = (portfolio.equity > 0.0 && price > 0.0);

        // 1. Capture current portfolio exposure
        decision.current_side = (portfolio.position_shares > 0.0) ? PositionSide::Long : ((portfolio.position_shares < 0.0) ? PositionSide::Short : PositionSide::Flat);

        decision.current_exposure_notional = (price > 0.0) ? (portfolio.position_shares * price) : 0.0;
        const double current_weight        = has_account_equity ? (decision.current_exposure_notional / portfolio.equity) : 0.0;

        // 2. Predictive uncertainty (entropy) check
        decision.entropy_passed = forecast.distribution_entropy <= limits_.max_entropy;
        if (!decision.entropy_passed)
        {
            decision.signal         = OrderSignal::Hold;
            decision.candidate_side = decision.current_side;
            decision.risk_veto      = true;
            decision.should_flatten = (decision.current_side != PositionSide::Flat);
            decision.reason =
                    "VETO: Forecast entropy (" + std::to_string(forecast.distribution_entropy) + ") exceeds max uncertainty (" + std::to_string(limits_.max_entropy) + ")";
            return decision;
        }

        // 3. Determine candidate alpha side before selecting risk tail
        const bool strong_alpha_buy  = (forecast.expected_return > limits_.min_expected_return) && (forecast.prob_positive_return >= limits_.min_bullish_prob);
        const bool strong_alpha_sell = (forecast.expected_return < -limits_.min_expected_return) && (forecast.prob_positive_return <= limits_.max_bearish_prob);

        OrderSignal candidate_signal = OrderSignal::Hold;
        if (strong_alpha_buy)
        {
            candidate_signal        = OrderSignal::Buy;
            decision.candidate_side = PositionSide::Long;
        }
        else if (strong_alpha_sell)
        {
            candidate_signal        = OrderSignal::Sell;
            decision.candidate_side = PositionSide::Short;
        }
        else
        {
            decision.candidate_side = decision.current_side;
        }

        // 4. Compute candidate position sizing & post-trade portfolio exposure
        double target_shares   = 0.0;
        double target_notional = 0.0;
        double target_weight   = 0.0;

        if (candidate_signal == OrderSignal::Buy)
        {
            if (has_account_equity)
            {
                target_shares   = std::floor((portfolio.equity * alloc) / price);
                target_notional = target_shares * price;
                target_weight   = target_notional / portfolio.equity;
            }
            else
            {
                target_notional = alloc;
                target_weight   = alloc;
            }
        }
        else if (candidate_signal == OrderSignal::Sell)
        {
            if (has_account_equity)
            {
                target_shares   = -std::floor((portfolio.equity * alloc) / price);
                target_notional = target_shares * price;
                target_weight   = target_notional / portfolio.equity;
            }
            else
            {
                target_notional = -alloc;
                target_weight   = -alloc;
            }
        }

        decision.target_exposure_notional = target_notional;

        double post_notional = decision.current_exposure_notional;
        double post_weight   = current_weight;

        if (candidate_signal == OrderSignal::Buy)
        {
            if (portfolio.position_shares > 0.0 && !portfolio.allow_pyramiding)
            {
                post_notional = decision.current_exposure_notional;
                post_weight   = current_weight;
            }
            else if (portfolio.position_shares > 0.0 && portfolio.allow_pyramiding)
            {
                post_notional = decision.current_exposure_notional + target_notional;
                post_weight   = current_weight + target_weight;
            }
            else
            {
                // Flat or reversing Short -> Long
                post_notional = target_notional;
                post_weight   = target_weight;
            }
        }
        else if (candidate_signal == OrderSignal::Sell)
        {
            if (portfolio.position_shares < 0.0 && !portfolio.allow_pyramiding)
            {
                post_notional = decision.current_exposure_notional;
                post_weight   = current_weight;
            }
            else if (portfolio.position_shares < 0.0 && portfolio.allow_pyramiding)
            {
                post_notional = decision.current_exposure_notional + target_notional;
                post_weight   = current_weight + target_weight;
            }
            else
            {
                // Flat or reversing Long -> Short
                post_notional = target_notional;
                post_weight   = target_weight;
            }
        }

        decision.post_trade_exposure_notional = post_notional;
        decision.post_trade_exposure_weight   = post_weight;

        const PositionSide effective_side = (post_weight > 0.0) ? PositionSide::Long : ((post_weight < 0.0) ? PositionSide::Short : PositionSide::Flat);

        // 5. Compute side-aware unit tail risk and portfolio loss contribution
        const double gross_weight   = std::abs(post_weight);
        const double gross_notional = std::abs(post_notional);

        decision.unit_var_95  = forecast.side_var_95(effective_side);
        decision.unit_cvar_95 = forecast.side_cvar_95(effective_side);

        decision.portfolio_var_95    = gross_weight * decision.unit_var_95;
        decision.portfolio_cvar_95   = gross_weight * decision.unit_cvar_95;
        decision.var_95_dollar_loss  = gross_notional * decision.unit_var_95;
        decision.cvar_95_dollar_loss = gross_notional * decision.unit_cvar_95;

        // 6. Apply side-aware portfolio VaR & CVaR limits
        const bool   is_short   = (effective_side == PositionSide::Short);
        const double var_limit  = limits_.effective_var_limit(is_short);
        const double cvar_limit = limits_.effective_cvar_limit(is_short);

        decision.var_passed  = (decision.portfolio_var_95 <= var_limit);
        decision.cvar_passed = (decision.portfolio_cvar_95 <= cvar_limit);

        const double max_alloc_var          = (decision.unit_var_95 > 1e-12) ? (var_limit / decision.unit_var_95) : 1.0;
        const double max_alloc_cvar         = (decision.unit_cvar_95 > 1e-12) ? (cvar_limit / decision.unit_cvar_95) : 1.0;
        decision.max_allowed_allocation_pct = std::clamp(std::min({alloc, max_alloc_var, max_alloc_cvar}), 0.0, 1.0);

        if (has_account_equity)
        {
            const double dir_sign       = (effective_side == PositionSide::Short) ? -1.0 : ((effective_side == PositionSide::Long) ? 1.0 : 0.0);
            decision.recommended_shares = dir_sign * std::floor((portfolio.equity * decision.max_allowed_allocation_pct) / price);
        }

        // Check whether an existing open position independently breaches its own side limits
        if (decision.current_side != PositionSide::Flat)
        {
            const bool   curr_is_short  = (decision.current_side == PositionSide::Short);
            const double curr_gross_wt  = std::abs(current_weight);
            const double curr_port_var  = curr_gross_wt * forecast.side_var_95(decision.current_side);
            const double curr_port_cvar = curr_gross_wt * forecast.side_cvar_95(decision.current_side);
            decision.should_flatten     = (curr_port_var > limits_.effective_var_limit(curr_is_short)) || (curr_port_cvar > limits_.effective_cvar_limit(curr_is_short));
        }

        const char* side_str = (effective_side == PositionSide::Long) ? "LONG" : ((effective_side == PositionSide::Short) ? "SHORT" : "FLAT");

        if (!decision.var_passed)
        {
            decision.signal    = OrderSignal::Hold;
            decision.risk_veto = true;
            decision.reason    = std::string("VETO [") + side_str + "]: Portfolio VaR(95%) loss " + std::to_string(decision.portfolio_var_95 * 100.0) + "% (unit " +
                              std::to_string(decision.unit_var_95 * 100.0) + "% @ " + std::to_string(gross_weight * 100.0) + "% exposure) exceeds max limit " +
                              std::to_string(var_limit * 100.0) + "%";
            return decision;
        }

        if (!decision.cvar_passed)
        {
            decision.signal    = OrderSignal::Hold;
            decision.risk_veto = true;
            decision.reason    = std::string("VETO [") + side_str + "]: Portfolio CVaR(95%) loss " + std::to_string(decision.portfolio_cvar_95 * 100.0) + "% (unit " +
                              std::to_string(decision.unit_cvar_95 * 100.0) + "% @ " + std::to_string(gross_weight * 100.0) + "% exposure) exceeds max limit " +
                              std::to_string(cvar_limit * 100.0) + "%";
            return decision;
        }

        if (candidate_signal == OrderSignal::Buy)
        {
            decision.signal        = OrderSignal::Buy;
            decision.return_passed = true;
            decision.reason        = "BUY/LONG PASSED: E[r] (" + std::to_string(forecast.expected_return * 100.0) + "%) > hurdle, P(r>0) (" +
                              std::to_string(forecast.prob_positive_return * 100.0) + "%), Port VaR(95%)=" + std::to_string(decision.portfolio_var_95 * 100.0) +
                              "%, Port CVaR(95%)=" + std::to_string(decision.portfolio_cvar_95 * 100.0) + "%";
        }
        else if (candidate_signal == OrderSignal::Sell)
        {
            decision.signal        = OrderSignal::Sell;
            decision.return_passed = true;
            decision.reason        = "SELL/SHORT PASSED: E[r] (" + std::to_string(forecast.expected_return * 100.0) + "%) < -hurdle, P(r>0) (" +
                              std::to_string(forecast.prob_positive_return * 100.0) + "%), Port VaR(95%)=" + std::to_string(decision.portfolio_var_95 * 100.0) +
                              "%, Port CVaR(95%)=" + std::to_string(decision.portfolio_cvar_95 * 100.0) + "%";
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
