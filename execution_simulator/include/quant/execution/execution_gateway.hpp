#pragma once

#include "quant/core/types.hpp"

#include <cstddef>
#include <vector>

struct FeeStructure
{
    double commission_rate{0.0001};
    double fixed_fee_per_trade{0.0};
    double slippage_bps{0.0001};
};

struct ExecutionFill
{
    FixedTimestamp timestamp{};
    OrderSignal    side{OrderSignal::Hold};

    double fill_price{0.0};
    double shares{0.0};
    double trade_value{0.0};
    double fee_paid{0.0};

    // Non-zero when an existing position (or partial slice) was closed.
    // Includes proportional entry fee and exit fee.
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

// Unified execution interface allowing paper simulation (SimulatedExchange)
// and live broker execution adapters to be swapped without touching strategy logic.
class IExecutionGateway
{
public:
    virtual ~IExecutionGateway() = default;

    virtual ExecutionFill execute_order(OrderSignal signal, FixedTimestamp timestamp, double current_price, double allocation_pct = 0.95) = 0;
    virtual ExecutionFill close_position(FixedTimestamp timestamp, double current_price)                                                  = 0;

    // Evaluates venue-side protective brackets (SL/TP) on incoming bar updates and marks equity to market.
    // Returns a closing ExecutionFill if a bracket triggered; otherwise returns an empty fill.
    virtual ExecutionFill on_bar_update(const Tick& bar, double stop_loss_pct = 0.0, double take_profit_pct = 0.0) = 0;

    [[nodiscard]] virtual const AccountState&               account() const noexcept                                         = 0;
    [[nodiscard]] virtual const std::vector<ExecutionFill>& fills() const noexcept                                           = 0;
    [[nodiscard]] virtual double                            sharpe_ratio(double risk_free_rate_annual = 0.02) const noexcept = 0;
};
