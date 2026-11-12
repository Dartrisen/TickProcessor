#pragma once

#include "quant/core/types.hpp"
#include "quant/risk/risk_manager.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

struct FeeStructure
{
    double commission_rate{0.0001};
    double fixed_fee_per_trade{0.0};
    double slippage_bps{0.0001};
};

struct ExecutionFill
{
    std::string timestamp;
    OrderSignal side{OrderSignal::Hold};

    double fill_price{0.0};
    double shares{0.0};
    double trade_value{0.0};
    double fee_paid{0.0};

    // Non-zero when an existing position was closed.
    // Includes both entry and exit fees.
    double realized_pnl{0.0};

    bool opened_position{false};
    bool closed_position{false};
};

struct AccountState
{
    double cash{10000.0};
    double position_shares{0.0};
    double avg_entry_price{0.0};
    double entry_fee_paid{0.0};

    double realized_pnl{0.0};
    double gross_profit{0.0};
    double gross_loss{0.0}; // Stored as positive magnitude
    double total_fees_paid{0.0};

    double peak_equity{10000.0};
    double max_drawdown{0.0};

    std::size_t total_trades{0};
    std::size_t winning_trades{0};
    std::size_t losing_trades{0};

    [[nodiscard]] double equity(double current_price) const noexcept
    {
        return cash + (position_shares * current_price);
    }

    [[nodiscard]] double unrealized_pnl(double current_price) const noexcept
    {
        if (position_shares > 0.0)
            return position_shares * (current_price - avg_entry_price) - entry_fee_paid;

        if (position_shares < 0.0)
        {
            const double quantity = -position_shares;
            return quantity * (avg_entry_price - current_price) - entry_fee_paid;
        }

        return 0.0;
    }

    [[nodiscard]] double avg_win() const noexcept
    {
        return winning_trades > 0 ? (gross_profit / static_cast<double>(winning_trades)) : 0.0;
    }

    [[nodiscard]] double avg_loss() const noexcept
    {
        return losing_trades > 0 ? (gross_loss / static_cast<double>(losing_trades)) : 0.0;
    }

    [[nodiscard]] double payoff_ratio() const noexcept
    {
        const double loss = avg_loss();
        return loss > 1e-12 ? (avg_win() / loss) : 0.0;
    }

    [[nodiscard]] double profit_factor() const noexcept
    {
        return gross_loss > 1e-12 ? (gross_profit / gross_loss) : 0.0;
    }

    [[nodiscard]] double expectancy_per_trade() const noexcept
    {
        return total_trades > 0 ? (realized_pnl / static_cast<double>(total_trades)) : 0.0;
    }
};

class SimulatedExchange
{
public:
    explicit SimulatedExchange(double initial_capital = 10000.0, FeeStructure fees = {}) : fees_(fees)
    {
        account_.cash        = initial_capital;
        account_.peak_equity = initial_capital;
    }

    // Buy means target a long position.
    // Sell means target a short position.
    //
    // A reversal closes the existing position and opens the new position at
    // the same simulated open price. Every individual fill is stored in
    // fills_history_. The returned fill is the final action of this request.
    ExecutionFill execute_order(OrderSignal signal, const std::string& timestamp, double current_price, double allocation_pct = 0.95)
    {
        ExecutionFill result{};

        if (signal == OrderSignal::Hold || current_price <= 0.0)
            return result;

        allocation_pct = std::clamp(allocation_pct, 0.0, 1.0);
        update_drawdown(account_.equity(current_price));

        if (signal == OrderSignal::Buy)
        {
            if (account_.position_shares < 0.0)
                result = close_position(timestamp, current_price);

            if (account_.position_shares == 0.0)
                result = open_long(timestamp, current_price, allocation_pct);
        }
        else if (signal == OrderSignal::Sell)
        {
            if (account_.position_shares > 0.0)
                result = close_position(timestamp, current_price);

            if (account_.position_shares == 0.0)
                result = open_short(timestamp, current_price, allocation_pct);
        }

        // Equity is intentionally recorded only by update_tick(close).
        // This keeps the equity curve close-to-close and the Sharpe
        // annualization consistent with one observation per bar.
        return result;
    }

    ExecutionFill close_position(const std::string& timestamp, double current_price)
    {
        ExecutionFill fill{};
        fill.timestamp = timestamp;

        if (account_.position_shares == 0.0 || current_price <= 0.0)
            return fill;

        const bool   closing_long = account_.position_shares > 0.0;
        const double quantity     = std::abs(account_.position_shares);

        // Long exits sell at bid-like price. Short exits buy at ask-like price.
        const double fill_price = closing_long ? current_price * (1.0 - fees_.slippage_bps) : current_price * (1.0 + fees_.slippage_bps);

        const double trade_value = quantity * fill_price;
        const double exit_fee    = (trade_value * fees_.commission_rate) + fees_.fixed_fee_per_trade;

        const double gross_pnl = closing_long ? quantity * (fill_price - account_.avg_entry_price) : quantity * (account_.avg_entry_price - fill_price);

        const double net_pnl = gross_pnl - account_.entry_fee_paid - exit_fee;

        if (closing_long)
        {
            // Sell long inventory.
            account_.cash += trade_value - exit_fee;
            fill.side = OrderSignal::Sell;
        }
        else
        {
            // Buy back borrowed shares.
            account_.cash -= trade_value + exit_fee;
            fill.side = OrderSignal::Buy;
        }

        account_.realized_pnl += net_pnl;
        account_.total_fees_paid += exit_fee;

        account_.position_shares = 0.0;
        account_.avg_entry_price = 0.0;
        account_.entry_fee_paid  = 0.0;

        account_.total_trades++;

        if (net_pnl > 0.0)
        {
            account_.winning_trades++;
            account_.gross_profit += net_pnl;
        }
        else if (net_pnl < 0.0)
        {
            account_.losing_trades++;
            account_.gross_loss += std::abs(net_pnl);
        }

        fill.fill_price      = fill_price;
        fill.shares          = quantity;
        fill.trade_value     = trade_value;
        fill.fee_paid        = exit_fee;
        fill.realized_pnl    = net_pnl;
        fill.closed_position = true;

        fills_history_.push_back(fill);
        return fill;
    }

    void update_tick(double current_price)
    {
        if (current_price <= 0.0)
            return;

        const double current_equity = account_.equity(current_price);
        update_drawdown(current_equity);
        equity_curve_.push_back(current_equity);
    }

    [[nodiscard]] const AccountState& account() const noexcept
    {
        return account_;
    }

    [[nodiscard]] const std::vector<ExecutionFill>& fills() const noexcept
    {
        return fills_history_;
    }

    [[nodiscard]] const std::vector<double>& equity_curve() const noexcept
    {
        return equity_curve_;
    }

    [[nodiscard]] double sharpe_ratio(double risk_free_rate_annual = 0.02) const noexcept
    {
        if (equity_curve_.size() < 3)
            return 0.0;

        constexpr double bars_per_year = 98280.0; // one-minute bars

        double      sum_returns = 0.0;
        std::size_t n           = 0;

        for (std::size_t i = 1; i < equity_curve_.size(); ++i)
        {
            if (equity_curve_[i - 1] <= 0.0)
                continue;

            sum_returns += (equity_curve_[i] / equity_curve_[i - 1]) - 1.0;
            ++n;
        }

        if (n < 2)
            return 0.0;

        const double mean_return = sum_returns / static_cast<double>(n);

        double squared_deviation_sum = 0.0;
        for (std::size_t i = 1; i < equity_curve_.size(); ++i)
        {
            if (equity_curve_[i - 1] <= 0.0)
                continue;

            const double r = (equity_curve_[i] / equity_curve_[i - 1]) - 1.0;
            const double d = r - mean_return;
            squared_deviation_sum += d * d;
        }

        const double stddev = std::sqrt(squared_deviation_sum / static_cast<double>(n - 1));

        if (stddev <= 1e-12)
            return 0.0;

        const double risk_free_per_bar = risk_free_rate_annual / bars_per_year;
        return std::sqrt(bars_per_year) * ((mean_return - risk_free_per_bar) / stddev);
    }

private:
    ExecutionFill open_long(const std::string& timestamp, double current_price, double allocation_pct)
    {
        ExecutionFill fill{};
        fill.timestamp = timestamp;
        fill.side      = OrderSignal::Buy;

        const double fill_price     = current_price * (1.0 + fees_.slippage_bps);
        const double target_capital = account_.equity(current_price) * allocation_pct;
        const double quantity       = std::floor(target_capital / fill_price);

        if (quantity <= 0.0)
            return fill;

        const double trade_value = quantity * fill_price;
        const double entry_fee   = (trade_value * fees_.commission_rate) + fees_.fixed_fee_per_trade;

        if (account_.cash < trade_value + entry_fee)
            return fill;

        account_.cash -= trade_value + entry_fee;
        account_.position_shares = quantity;
        account_.avg_entry_price = fill_price;
        account_.entry_fee_paid  = entry_fee;
        account_.total_fees_paid += entry_fee;

        fill.fill_price      = fill_price;
        fill.shares          = quantity;
        fill.trade_value     = trade_value;
        fill.fee_paid        = entry_fee;
        fill.opened_position = true;

        fills_history_.push_back(fill);
        return fill;
    }

    ExecutionFill open_short(const std::string& timestamp, double current_price, double allocation_pct)
    {
        ExecutionFill fill{};
        fill.timestamp = timestamp;
        fill.side      = OrderSignal::Sell;

        const double fill_price     = current_price * (1.0 - fees_.slippage_bps);
        const double target_capital = account_.equity(current_price) * allocation_pct;
        const double quantity       = std::floor(target_capital / fill_price);

        if (quantity <= 0.0)
            return fill;

        const double trade_value = quantity * fill_price;
        const double entry_fee   = (trade_value * fees_.commission_rate) + fees_.fixed_fee_per_trade;

        account_.cash += trade_value - entry_fee;
        account_.position_shares = -quantity;
        account_.avg_entry_price = fill_price;
        account_.entry_fee_paid  = entry_fee;
        account_.total_fees_paid += entry_fee;

        fill.fill_price      = fill_price;
        fill.shares          = quantity;
        fill.trade_value     = trade_value;
        fill.fee_paid        = entry_fee;
        fill.opened_position = true;

        fills_history_.push_back(fill);
        return fill;
    }

    void update_drawdown(double current_equity) noexcept
    {
        if (current_equity > account_.peak_equity)
        {
            account_.peak_equity = current_equity;
            return;
        }

        if (account_.peak_equity <= 0.0)
            return;

        const double drawdown = (account_.peak_equity - current_equity) / account_.peak_equity;

        if (drawdown > account_.max_drawdown)
            account_.max_drawdown = drawdown;
    }

    FeeStructure               fees_;
    AccountState               account_;
    std::vector<ExecutionFill> fills_history_;
    std::vector<double>        equity_curve_;
};
