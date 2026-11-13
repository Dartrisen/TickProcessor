#pragma once

#include "quant/core/matrix.hpp"
#include "quant/execution/execution_gateway.hpp"
#include "quant/models/interfaces.hpp"
#include "quant/risk/risk_manager.hpp"
#include "quant/session/session_observer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>

class QuantConsoleReporter final : public ISessionObserver
{
public:
    // ANSI Color Escape Codes
    static constexpr const char* RESET   = "\033[0m";
    static constexpr const char* BOLD    = "\033[1m";
    static constexpr const char* RED     = "\033[1;31m";
    static constexpr const char* GREEN   = "\033[1;32m";
    static constexpr const char* YELLOW  = "\033[1;33m";
    static constexpr const char* CYAN    = "\033[1;36m";
    static constexpr const char* MAGENTA = "\033[1;35m";
    static constexpr const char* GRAY    = "\033[1;30m";
    static constexpr const char* WHITE   = "\033[1;37m";

    void on_session_start() override
    {
        render_header("ORDER GATEWAY & TRADE EXECUTION STREAM");
    }

    void on_close_fill(std::string_view label, const ExecutionFill& fill) override
    {
        print_close_fill(label, fill);
    }

    void on_execution_fill(const ExecutionFill& fill) override
    {
        print_execution_fill(fill);
    }

    void on_session_end(const SessionSummary& summary, const AccountState& acc) override
    {
        print_account_performance(summary.initial_capital, summary.final_price, summary.start_price, summary.sharpe_ratio, acc);
    }

    static void render_header(std::string_view title)
    {
        std::cout << "\n" << CYAN << "========================================================================================================\n";
        std::cout << "  " << BOLD << WHITE << title << RESET << CYAN << "\n";
        std::cout << "========================================================================================================\n" << RESET;
    }

    static void print_signal_event(std::string_view timestamp, double price, const SignalDecision& decision, const QuantForecast& forecast)
    {
        std::cout << std::fixed << std::setprecision(4);

        std::string signal_tag;
        if (decision.signal == OrderSignal::Buy)
        {
            signal_tag = std::string(GREEN) + BOLD + "[SIGNAL: BUY] " + RESET;
        }
        else if (decision.signal == OrderSignal::Sell)
        {
            signal_tag = std::string(RED) + BOLD + "[SIGNAL: SELL]" + RESET;
        }
        else if (decision.should_flatten)
        {
            signal_tag = std::string(MAGENTA) + BOLD + "[SIGNAL: EXIT]" + RESET;
        }
        else
        {
            signal_tag = std::string(YELLOW) + BOLD + "[SIGNAL: HOLD]" + RESET;
        }

        const char* side_label = (decision.candidate_side == PositionSide::Long) ? "LONG" : ((decision.candidate_side == PositionSide::Short) ? "SHORT" : "FLAT");

        const bool   has_side_risk = (decision.candidate_side != PositionSide::Flat);
        const double display_var   = has_side_risk ? decision.portfolio_var_95 : forecast.var_95;
        const double display_cvar  = has_side_risk ? decision.portfolio_cvar_95 : forecast.cvar_95;
        const double exposure_pct  = std::abs(decision.post_trade_exposure_weight) * 100.0;

        std::cout << GRAY << "[" << timestamp << "] " << RESET << signal_tag << " [" << CYAN << side_label << RESET << " @ " << std::setprecision(1) << exposure_pct << "%] "
                  << std::setprecision(4) << "Price: " << BOLD << "$" << price << RESET << " | E[r]: " << (forecast.expected_return >= 0 ? GREEN : RED)
                  << (forecast.expected_return * 100.0) << "%" << RESET << " | P(r>0): " << (forecast.prob_positive_return > 0.5 ? GREEN : RED)
                  << (forecast.prob_positive_return * 100.0) << "%" << RESET << " | VaR(95): " << (decision.var_passed ? GREEN : RED) << (display_var * 100.0) << "%" << RESET
                  << " | CVaR(95): " << (decision.cvar_passed ? GREEN : RED) << (display_cvar * 100.0) << "%" << RESET << "\n  " << GRAY << "└─► Reason: " << WHITE
                  << decision.reason << RESET << "\n";
    }

    static void print_forecast(const QuantForecast& f, std::size_t horizon_steps = 1)
    {
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "\n" << CYAN << "+------------------------------------------------------------------------------------------------------+\n";
        std::cout << "| " << BOLD << WHITE << "MARKET FORECAST & RISK PROFILE (Horizon: +" << horizon_steps << " step(s))" << RESET << CYAN
                  << "                                         |\n";
        std::cout << "+------------------------------------------------------------------------------------------------------+\n" << RESET;
        std::cout << "  Current Asset Price       : " << BOLD << "$" << f.current_price << RESET << "\n";
        std::cout << "  Expected Price (E[P])     : " << BOLD << "$" << f.expected_price << RESET << "  (Return: " << (f.expected_return >= 0 ? GREEN : RED)
                  << (f.expected_return * 100.0) << "%" << RESET << ")\n";
        std::cout << "  Most Likely Price (Mode)  : $" << f.mode_price << "\n";
        std::cout << "  95% Confidence Interval   : [$" << f.price_ci_lower_95 << " --- $" << f.price_ci_upper_95 << "]\n";
        std::cout << GRAY << "--------------------------------------------------------------------------------------------------------\n" << RESET;
        std::cout << "  Expected Volatility (σ)   : " << YELLOW << (f.expected_volatility * 100.0) << "%" << RESET << "\n";
        std::cout << "  Bullish Probability P(r>0): " << (f.prob_positive_return >= 0.5 ? GREEN : RED) << (f.prob_positive_return * 100.0) << "%" << RESET << "\n";
        std::cout << "  Long VaR / CVaR (95%)     : " << RED << (f.long_var_95 * 100.0) << "% / " << (f.long_cvar_95 * 100.0) << "% downside loss" << RESET << "\n";
        std::cout << "  Short VaR / CVaR (95%)    : " << RED << (f.short_var_95 * 100.0) << "% / " << (f.short_cvar_95 * 100.0) << "% upside loss" << RESET << "\n";
        std::cout << "  Kinetic Entropy (Uncert.) : " << MAGENTA << f.distribution_entropy << " bits" << RESET << "\n";
        std::cout << CYAN << "+------------------------------------------------------------------------------------------------------+\n" << RESET;
    }

    template <std::size_t Bins>
    static void print_distribution_ascii(double dx, double x_max, const std::array<double, Bins>& dist, std::string_view label)
    {
        std::cout << "\n" << BOLD << WHITE << "--- Density Distribution: " << label << " ---" << RESET << "\n";
        constexpr std::size_t max_bar_width = 40;

        double max_p = 0.0;
        for (double p : dist)
        {
            if (p > max_p)
                max_p = p;
        }

        for (std::size_t i = 0; i < Bins; ++i)
        {
            double center  = -x_max + (static_cast<double>(i) + 0.5) * dx;
            double prob    = dist[i];
            auto   bar_len = static_cast<std::size_t>((prob / (max_p > 0 ? max_p : 1.0)) * max_bar_width);

            std::string bar(bar_len, '#');
            const char* color = (center > 0.0) ? GREEN : (center < 0.0 ? RED : GRAY);

            std::cout << std::right << std::setw(7) << std::fixed << std::setprecision(4) << center << " | " << color << std::left << std::setw(max_bar_width) << bar << RESET
                      << " | " << std::setw(6) << std::setprecision(2) << (prob * 100.0) << "%\n";
        }
    }

    template <std::size_t Bins>
    static void print_transition_matrix(const FlatMatrix<double, Bins>& W, std::size_t max_display_bins = 8)
    {
        std::size_t bins = std::min(Bins, max_display_bins);

        std::cout << "\n" << BOLD << WHITE << "--- Dynamic Transition Kernel Matrix W(i -> j) (First " << bins << "x" << bins << " state subset) ---" << RESET << "\n";
        std::cout << "       ";
        for (std::size_t j = 0; j < bins; ++j)
        {
            std::cout << CYAN << "  Bin" << std::setw(2) << j << RESET;
        }
        std::cout << "\n";

        for (std::size_t i = 0; i < bins; ++i)
        {
            std::cout << CYAN << "Bin" << std::setw(2) << i << RESET << " |";
            for (std::size_t j = 0; j < bins; ++j)
            {
                double val = W(i, j);
                if (val > 0.2)
                    std::cout << BOLD << GREEN;
                else if (val > 0.05)
                    std::cout << WHITE;
                else
                    std::cout << GRAY;

                std::cout << std::right << std::setw(7) << std::fixed << std::setprecision(3) << val << RESET;
            }
            std::cout << "\n";
        }
    }

    static void print_execution_summary(std::size_t total_ticks, std::size_t buys, std::size_t sells, std::size_t holds, std::size_t risk_rejections)
    {
        std::cout << "\n" << CYAN << "========================================================================================================\n";
        std::cout << "  " << BOLD << WHITE << "PRE-TRADE RISK & EXECUTION TELEMETRY SUMMARY" << RESET << CYAN << "\n";
        std::cout << "========================================================================================================\n" << RESET;
        std::cout << "  Total Hot Path Ticks Evaluated : " << BOLD << total_ticks << RESET << "\n";
        std::cout << "  " << GREEN << BOLD << "Buy Signals Executed [BUY]    : " << buys << RESET << "\n";
        std::cout << "  " << RED << BOLD << "Sell Signals Executed [SELL]  : " << sells << RESET << "\n";
        std::cout << "  " << YELLOW << "Hold Decisions [HOLD]         : " << holds << RESET << "\n";
        std::cout << "  " << MAGENTA << "Risk Limits Vetoed Trades     : " << risk_rejections << RESET << "\n";
        std::cout << CYAN << "========================================================================================================\n" << RESET;
    }

    static void print_close_fill(std::string_view label, const ExecutionFill& fill)
    {
        if (!fill.closed_position)
            return;

        const char* pnl_color = fill.realized_pnl >= 0.0 ? GREEN : RED;
        std::cout << GRAY << "[" << fill.timestamp.view() << "] " << RESET << YELLOW << BOLD << label << RESET << " | Shares: " << fill.shares << " @ $" << std::fixed
                  << std::setprecision(2) << fill.fill_price << " | Value: $" << fill.trade_value << " | Fee: $" << fill.fee_paid << " | Realized PnL: " << pnl_color << BOLD << "$"
                  << fill.realized_pnl << RESET << "\n";
    }

    static void print_execution_fill(const ExecutionFill& fill)
    {
        const bool  is_buy = fill.side == OrderSignal::Buy;
        const char* color  = is_buy ? GREEN : RED;

        std::cout << GRAY << "[" << fill.timestamp.view() << "] " << RESET << color << BOLD << (is_buy ? "EXECUTED BUY/COVER " : "EXECUTED SELL/SHORT") << RESET
                  << " | Shares: " << fill.shares << " @ $" << std::fixed << std::setprecision(2) << fill.fill_price << " | Value: $" << fill.trade_value << " | Fee: $"
                  << fill.fee_paid;

        if (fill.closed_position)
        {
            const char* pnl_color = fill.realized_pnl >= 0.0 ? GREEN : RED;
            std::cout << " | Realized PnL: " << pnl_color << BOLD << "$" << fill.realized_pnl << RESET;
        }
        std::cout << "\n";
    }

    static void print_account_performance(double initial_cash, double final_price, double start_oos_price, double sharpe, const AccountState& acc)
    {
        const double buy_hold_pct  = (start_oos_price > 0.0) ? (((final_price / start_oos_price) - 1.0) * 100.0) : 0.0;
        const double final_equity  = acc.equity(final_price);
        const double total_pnl     = final_equity - initial_cash;
        const double return_pct    = (total_pnl / initial_cash) * 100.0;
        const double win_rate      = (acc.total_trades > 0) ? (static_cast<double>(acc.winning_trades) / acc.total_trades * 100.0) : 0.0;
        const double profit_factor = acc.profit_factor();
        const double payoff_ratio  = acc.payoff_ratio();
        const double expectancy    = acc.expectancy_per_trade();

        render_header("SIMULATED ACCOUNT PERFORMANCE & P&L STATEMENT");
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Initial Starting Capital     : $" << initial_cash << "\n";
        std::cout << "  Final Account Equity         : $" << final_equity << "\n";
        std::cout << "  Net Realized Profit / Loss   : " << (total_pnl >= 0 ? GREEN : RED) << BOLD << "$" << total_pnl << " (" << return_pct << "%)" << RESET << "\n";
        std::cout << "  Out-of-Sample Buy & Hold     : " << (buy_hold_pct >= 0 ? GREEN : RED) << buy_hold_pct << "%" << RESET << " (Benchmark)\n";
        std::cout << "  Total Exchange Fees Paid     : " << YELLOW << "$" << acc.total_fees_paid << RESET << " (Slippage & Commission)\n";
        std::cout << "  Max Equity Drawdown          : " << RED << (acc.max_drawdown * 100.0) << "%" << RESET << "\n";
        std::cout << "  Completed Round-Trip Trades  : " << acc.total_trades << " (Wins: " << acc.winning_trades << " | Losses: " << acc.losing_trades << ")\n";
        std::cout << "  Strategy Win Rate            : " << (win_rate >= 50.0 ? GREEN : YELLOW) << win_rate << "%" << RESET << "\n";
        std::cout << "  Avg Win / Avg Loss (Payoff)  : " << GREEN << "$" << acc.avg_win() << RESET << " / " << RED << "$" << acc.avg_loss() << RESET << " (" << std::setprecision(2)
                  << payoff_ratio << ":1)\n";
        std::cout << "  Profit Factor (Gross W/L)    : " << (profit_factor >= 1.2 ? GREEN : YELLOW) << std::setprecision(2) << profit_factor << RESET << "\n";
        std::cout << "  Net Expectancy per Trade     : " << (expectancy >= 0.0 ? GREEN : RED) << "$" << std::setprecision(2) << expectancy << RESET << "\n";
        std::cout << "  Annualized Sharpe Ratio      : " << (sharpe >= 1.0 ? GREEN : YELLOW) << std::setprecision(3) << sharpe << RESET << "\n";
        std::cout << CYAN << "========================================================================================================\n" << RESET;
    }
};
