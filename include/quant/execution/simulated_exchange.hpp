#pragma once

#include "quant/core/types.hpp"
#include "quant/risk/risk_manager.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct FeeStructure
{
    double commission_rate{0.0001};  // 1 bps per trade value (e.g. 0.01%)
    double fixed_fee_per_trade{0.0}; // $0.00 per trade (Alpaca structure)
    double slippage_bps{0.0001};     // 1 bps execution slippage
};

struct ExecutionFill
{
    std::string timestamp;
    OrderSignal side{OrderSignal::Hold};
    double      fill_price{0.0};
    double      shares{0.0};
    double      trade_value{0.0};
    double      fee_paid{0.0};
    double      realized_pnl{0.0};
};

struct AccountState
{
    double      cash{10000.0};       // $10,000 initial capital
    double      position_shares{0.0}; // Current holdings (+ long, - short)
    double      avg_entry_price{0.0};
    double      realized_pnl{0.0};
    double      total_fees_paid{0.0};
    double      peak_equity{10000.0};
    double      max_drawdown{0.0};
    std::size_t total_trades{0};
    std::size_t winning_trades{0};
    std::size_t losing_trades{0};

    [[nodiscard]] double equity(double current_price) const noexcept
    {
        return cash + (position_shares * current_price);
    }

    [[nodiscard]] double unrealized_pnl(double current_price) const noexcept
    {
        if (position_shares == 0.0)
            return 0.0;
        return position_shares * (current_price - avg_entry_price);
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

    // Executes order requests from signal decisions
    ExecutionFill execute_order(OrderSignal signal, const std::string& timestamp, double current_price, double allocation_pct = 0.95)
    {
        ExecutionFill fill{};
        fill.timestamp = timestamp;
        fill.side      = signal;

        if (signal == OrderSignal::Hold || current_price <= 0.0)
        {
            return fill;
        }

        double current_equity = account_.equity(current_price);
        update_drawdown(current_equity);

        if (signal == OrderSignal::Buy)
        {
            if (account_.position_shares < 0.0)
            {
                close_position(timestamp, current_price);
            }

            if (account_.position_shares == 0.0)
            {
                double fill_price     = current_price * (1.0 + fees_.slippage_bps);
                double target_capital = account_.cash * allocation_pct;
                double shares         = std::floor(target_capital / fill_price);

                if (shares <= 0.0)
                    return fill;

                double trade_val = shares * fill_price;
                double fee       = (trade_val * fees_.commission_rate) + fees_.fixed_fee_per_trade;

                if (account_.cash >= (trade_val + fee))
                {
                    account_.cash -= (trade_val + fee);
                    account_.position_shares = shares;
                    account_.avg_entry_price = fill_price;
                    account_.total_fees_paid += fee;
                    account_.total_trades++;

                    fill.fill_price  = fill_price;
                    fill.shares      = shares;
                    fill.trade_value = trade_val;
                    fill.fee_paid    = fee;
                    fills_history_.push_back(fill);
                }
            }
        }
        else if (signal == OrderSignal::Sell)
        {
            if (account_.position_shares > 0.0)
            {
                fill = close_position(timestamp, current_price);
            }
        }

        equity_curve_.push_back(account_.equity(current_price));
        return fill;
    }

    ExecutionFill close_position(const std::string& timestamp, double current_price)
    {
        ExecutionFill fill{};
        fill.timestamp = timestamp;
        fill.side      = OrderSignal::Sell;

        if (account_.position_shares == 0.0)
            return fill;

        double fill_price = current_price * (1.0 - fees_.slippage_bps);
        double shares     = account_.position_shares;
        double trade_val  = shares * fill_price;
        double fee        = (trade_val * fees_.commission_rate) + fees_.fixed_fee_per_trade;

        double cost_basis = shares * account_.avg_entry_price;
        double pnl        = (trade_val - cost_basis) - fee;

        account_.cash += (trade_val - fee);
        account_.realized_pnl += pnl;
        account_.total_fees_paid += fee;
        account_.position_shares = 0.0;
        account_.avg_entry_price = 0.0;

        if (pnl > 0.0)
            account_.winning_trades++;
        else if (pnl < 0.0)
            account_.losing_trades++;

        fill.fill_price   = fill_price;
        fill.shares       = shares;
        fill.trade_value  = trade_val;
        fill.fee_paid     = fee;
        fill.realized_pnl = pnl;

        fills_history_.push_back(fill);
        return fill;
    }

    void update_tick(double current_price)
    {
        double current_equity = account_.equity(current_price);
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
        if (equity_curve_.size() < 2)
            return 0.0;

        std::vector<double> returns;
        returns.reserve(equity_curve_.size() - 1);
        double sum_ret = 0.0;

        for (std::size_t i = 1; i < equity_curve_.size(); ++i)
        {
            double ret = (equity_curve_[i] / equity_curve_[i - 1]) - 1.0;
            returns.push_back(ret);
            sum_ret += ret;
        }

        double mean_ret = sum_ret / returns.size();
        double variance = 0.0;
        for (double r : returns)
        {
            variance += (r - mean_ret) * (r - mean_ret);
        }
        double stddev = std::sqrt(variance / returns.size());

        if (stddev <= 1e-12)
            return 0.0;

        // Annualized Sharpe assuming 1-minute bars (~98,280 bars/year)
        double rf_per_bar = risk_free_rate_annual / 98280.0;
        return std::sqrt(98280.0) * ((mean_ret - rf_per_bar) / stddev);
    }

private:
    void update_drawdown(double current_equity) noexcept
    {
        if (current_equity > account_.peak_equity)
        {
            account_.peak_equity = current_equity;
        }
        else
        {
            double dd = (account_.peak_equity - current_equity) / account_.peak_equity;
            if (dd > account_.max_drawdown)
            {
                account_.max_drawdown = dd;
            }
        }
    }

    FeeStructure               fees_;
    AccountState               account_;
    std::vector<ExecutionFill> fills_history_;
    std::vector<double>        equity_curve_;
};
