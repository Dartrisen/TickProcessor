#include "quant/core/config.hpp"
#include "quant/core/matrix.hpp"
#include "quant/core/ring_buffer.hpp"
#include "quant/core/types.hpp"
#include "quant/data/csv_reader.hpp"
#include "quant/execution/simulated_exchange.hpp"
#include "quant/features/feature_engine.hpp"
#include "quant/models/empirical_kernel.hpp"
#include "quant/models/kinetic_model.hpp"
#include "quant/risk/risk_manager.hpp"
#include "quant/telemetry/console_reporter.hpp"

#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

constexpr std::size_t BINS = 16;
using Matrix               = FlatMatrix<double, BINS>;
using KineticEngine        = HighPerformanceKineticEngine<BINS>;

int main()
{
    try
    {
        QuantConsoleReporter::render_header("LIVE API BACKTEST & PAPER TRADING SIMULATION ENGINE");

        // configure exchange simulation & fee model (Alpaca/IBKR)
        FeeStructure alpaca_fees{
                .commission_rate     = 0.0001, // 0.01% fee (1 bps)
                .fixed_fee_per_trade = 0.00,   // $0.00 per trade
                .slippage_bps        = 0.0001  // 1 bps execution slippage
        };
        double            initial_cash = 10000.0; // $10,000 starting account
        SimulatedExchange exchange(initial_cash, alpaca_fees);

        // load historical market CSV ticks
        std::string data_path = "/Users/cdg546/Downloads/data/SPY_clean_1min.csv";
        std::cout << "[+] Connecting to simulated market feed: " << QuantConsoleReporter::CYAN << data_path << QuantConsoleReporter::RESET << " ...\n";

        MarketDataset                market;
        std::vector<DerivedFeatures> historical_features;
        try
        {
            market = CsvReader::load(data_path);
            historical_features.reserve(market.size());

            OnlineFeatureEngine<20> feature_prepass;
            DerivedFeatures         feat{};
            for (const auto& tick : market.bars())
            {
                if (feature_prepass.process(tick, feat))
                {
                    historical_features.push_back(feat);
                }
            }
        }
        catch (const std::exception&)
        {
            std::cout << QuantConsoleReporter::YELLOW << "[!] CSV missing. Generating synthetic high-frequency price feed...\n" << QuantConsoleReporter::RESET;
            Tick synth_tick;
            synth_tick.close  = 400.0;
            synth_tick.volume = 1000;
            OnlineFeatureEngine<20> feature_prepass;
            DerivedFeatures         feat{};

            for (std::size_t i = 0; i < 600; ++i)
            {
                synth_tick.datetime = "2026-03-31 09:" + std::to_string(i / 60) + ":" + std::to_string(i % 60);
                double ret          = 0.0008 * std::sin(static_cast<double>(i) * 0.15) + ((i % 7 == 0) ? 0.002 : -0.0005);
                synth_tick.close *= (1.0 + ret);
                synth_tick.open = synth_tick.close;
                synth_tick.high = synth_tick.close * 1.0005;
                synth_tick.low  = synth_tick.close * 0.9995;
                market.add(synth_tick);

                if (feature_prepass.process(synth_tick, feat))
                {
                    historical_features.push_back(feat);
                }
            }
        }

        std::cout << "[+] Loaded " << QuantConsoleReporter::BOLD << market.size() << QuantConsoleReporter::RESET << " market ticks into live feed simulator.\n";

        // setup kinetic strategy engine & calibrate baseline kernel
        double        x_max = 0.015;
        KineticEngine engine(x_max);
        auto          kernel = std::make_unique<Matrix>();

        EmpiricalKernel<BINS> empirical_builder;
        empirical_builder.build_kernel(engine.dx(), engine.x_max(), historical_features, *kernel);
        engine.update_kernel(kernel.get());

        // live pipeline components
        OnlineFeatureEngine<20>    hot_feature_engine;
        RiskEngine                 risk_engine;
        KineticEngine::StateVector current_distribution = engine.initial_uniform_distribution();
        KineticEngine::StateVector next_distribution{};
        DerivedFeatures            feat{};

        QuantConsoleReporter::render_header("SIMULATED ORDER GATEWAY & LIVE TRADE EXECUTION STREAM");

        // run live feed tick processing & execution loop
        for (const auto& tick : market.bars())
        {
            if (hot_feature_engine.process(tick, feat))
            {
                std::size_t state_idx = engine.state_index(feat.log_return);
                current_distribution  = engine.state_delta(state_idx);

                // strategy forward density propagation
                engine.evolve(current_distribution, next_distribution);

                // risk & signal analysis
                QuantForecast  forecast = engine.analyze_and_predict(tick.close, next_distribution);
                SignalDecision decision = risk_engine.evaluate(forecast);

                // route signal to simulated order gateway
                if (decision.signal == OrderSignal::Buy || decision.signal == OrderSignal::Sell)
                {
                    ExecutionFill fill = exchange.execute_order(decision.signal, tick.datetime, tick.close, 0.95);
                    if (fill.shares > 0.0)
                    {
                        const char* tag = (fill.side == OrderSignal::Buy) ? QuantConsoleReporter::GREEN : QuantConsoleReporter::RED;
                        std::cout << QuantConsoleReporter::GRAY << "[" << fill.timestamp << "] " << QuantConsoleReporter::RESET << tag << QuantConsoleReporter::BOLD
                                  << (fill.side == OrderSignal::Buy ? "EXECUTED BUY " : "EXECUTED SELL") << QuantConsoleReporter::RESET << " | Shares: " << fill.shares << " @ $"
                                  << std::fixed << std::setprecision(2) << fill.fill_price << " | Value: $" << std::setprecision(2) << fill.trade_value << " | Fee: $"
                                  << std::setprecision(2) << fill.fee_paid;

                        if (fill.realized_pnl != 0.0)
                        {
                            const char* pnl_color = fill.realized_pnl >= 0 ? QuantConsoleReporter::GREEN : QuantConsoleReporter::RED;
                            std::cout << " | Realized PnL: " << pnl_color << QuantConsoleReporter::BOLD << "$" << fill.realized_pnl << QuantConsoleReporter::RESET;
                        }
                        std::cout << "\n";
                    }
                }
                else
                {
                    exchange.update_tick(tick.close);
                }
            }
        }

        // force close open position at end of simulation for final accounting
        if (!market.empty() && exchange.account().position_shares != 0.0)
        {
            const auto& last_tick = market.bars().back();
            exchange.close_position(last_tick.datetime, last_tick.close);
        }

        // output portfolio PnL & performance metrics
        const auto& acc          = exchange.account();
        double      final_price  = market.empty() ? 400.0 : market.bars().back().close;
        double      final_equity = acc.equity(final_price);
        double      total_pnl    = final_equity - initial_cash;
        double      return_pct   = (total_pnl / initial_cash) * 100.0;
        double      sharpe       = exchange.sharpe_ratio();
        double      win_rate     = (acc.total_trades > 0) ? (static_cast<double>(acc.winning_trades) / acc.total_trades * 100.0) : 0.0;

        QuantConsoleReporter::render_header("SIMULATED ACCOUNT PERFORMANCE & P&L STATEMENT");
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Initial Starting Capital     : $" << initial_cash << "\n";
        std::cout << "  Final Account Equity         : $" << final_equity << "\n";
        std::cout << "  Net Realized Profit / Loss   : " << (total_pnl >= 0 ? QuantConsoleReporter::GREEN : QuantConsoleReporter::RED) << QuantConsoleReporter::BOLD << "$"
                  << total_pnl << " (" << return_pct << "%)" << QuantConsoleReporter::RESET << "\n";
        std::cout << "  Total Exchange Fees Paid     : " << QuantConsoleReporter::YELLOW << "$" << acc.total_fees_paid << QuantConsoleReporter::RESET
                  << " (Slippage & Commission)\n";
        std::cout << "  Max Equity Drawdown          : " << QuantConsoleReporter::RED << (acc.max_drawdown * 100.0) << "%" << QuantConsoleReporter::RESET << "\n";
        std::cout << "  Total Round-Trip Trades      : " << acc.total_trades << " (Wins: " << acc.winning_trades << " | Losses: " << acc.losing_trades << ")\n";
        std::cout << "  Strategy Win Rate            : " << (win_rate >= 50.0 ? QuantConsoleReporter::GREEN : QuantConsoleReporter::RED) << win_rate << "%"
                  << QuantConsoleReporter::RESET << "\n";
        std::cout << "  Annualized Sharpe Ratio      : " << (sharpe >= 1.0 ? QuantConsoleReporter::GREEN : QuantConsoleReporter::YELLOW) << std::setprecision(3) << sharpe
                  << QuantConsoleReporter::RESET << "\n";
        std::cout << QuantConsoleReporter::CYAN << "========================================================================================================\n"
                  << QuantConsoleReporter::RESET;
    }
    catch (const std::exception& e)
    {
        std::cerr << QuantConsoleReporter::RED << "Fatal Simulation Error: " << e.what() << QuantConsoleReporter::RESET << "\n";
        return 1;
    }

    return 0;
}
