#include "quant/core/config.hpp"
#include "quant/core/matrix.hpp"
#include "quant/core/ring_buffer.hpp"
#include "quant/core/types.hpp"
#include "quant/data/csv_reader.hpp"
#include "quant/execution/simulated_exchange.hpp"
#include "quant/features/feature_engine.hpp"
#include "quant/models/kernels.hpp"
#include "quant/models/kinetic_model.hpp"
#include "quant/risk/risk_manager.hpp"
#include "quant/telemetry/console_reporter.hpp"

#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
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

        EngineConfig cfg = ConfigLoader::load("quant.env");

        FeeStructure      fees{.commission_rate = cfg.commission_rate, .fixed_fee_per_trade = cfg.fixed_fee, .slippage_bps = cfg.slippage_bps};
        double            initial_cash = cfg.initial_capital;
        SimulatedExchange exchange(initial_cash, fees);

        std::string data_path = cfg.data_path;
        std::cout << "[+] Connecting to simulated market feed: " << QuantConsoleReporter::CYAN << data_path << QuantConsoleReporter::RESET << " ...\n";

        MarketDataset market;
        try
        {
            market = CsvReader::load(data_path);
        }
        catch (const std::exception&)
        {
            std::cout << QuantConsoleReporter::YELLOW << "[!] CSV missing. Please provide a valid data file path in `quant.env`.\n" << QuantConsoleReporter::RESET;
            return 1;
        }

        std::cout << "[+] Loaded " << QuantConsoleReporter::BOLD << market.size() << QuantConsoleReporter::RESET << " market ticks into live feed simulator.\n";

        // ---------------------------------------------------------------------
        // IN-SAMPLE TRAINING PHASE (First 30% of data ONLY)
        // ---------------------------------------------------------------------
        std::size_t total_bars = market.size();
        std::size_t train_size = static_cast<std::size_t>(total_bars * 0.30); // Use 30% for training
        std::size_t test_size  = total_bars - train_size;

        std::vector<DerivedFeatures> train_features;
        train_features.reserve(train_size);

        OnlineFeatureEngine<20> train_fe;
        DerivedFeatures         feat{};

        for (std::size_t i = 0; i < train_size; ++i)
        {
            if (train_fe.process(market[i], feat))
            {
                train_features.push_back(feat);
            }
        }

        std::cout << "[+] Calibrating transition kernel on " << QuantConsoleReporter::BOLD << train_features.size() << QuantConsoleReporter::RESET
                  << " IN-SAMPLE training bars...\n";

        // Build kernel STRICTLY from in-sample training features
        double        x_max = cfg.x_max;
        KineticEngine engine(x_max);
        auto          kernel = std::make_unique<Matrix>();

        // EmpiricalKernel/SpatialSmoothedKernel
        EmpiricalKernel<BINS> kernel_builder;
        kernel_builder.build_kernel(engine.dx(), engine.x_max(), train_features, *kernel);
        engine.update_kernel(kernel.get());

        // ---------------------------------------------------------------------
        // OUT-OF-SAMPLE TRADING PHASE (Remaining 70% of data)
        // ---------------------------------------------------------------------
        OnlineFeatureEngine<20>    hot_feature_engine;
        RiskEngine                 risk_engine(cfg.risk_limits);
        KineticEngine::StateVector current_distribution = engine.initial_uniform_distribution();
        KineticEngine::StateVector next_distribution{};

        QuantConsoleReporter::render_header("SIMULATED ORDER GATEWAY & LIVE TRADE EXECUTION STREAM");
        std::cout << "[+] Trading OUT-OF-SAMPLE on remaining " << test_size << " market ticks...\n";

        for (std::size_t i = train_size; i < total_bars; ++i)
        {
            const auto& tick = market[i];

            if (hot_feature_engine.process(tick, feat))
            {
                std::size_t state_idx = engine.state_index(feat.log_return);
                current_distribution  = engine.state_delta(state_idx);

                // strategy forward density propagation
                engine.evolve(current_distribution, next_distribution);

                // risk & signal analysis
                QuantForecast  forecast = engine.analyze_and_predict(tick.close, next_distribution);
                SignalDecision decision = risk_engine.evaluate(forecast);

                // route signal to simulated order gateway with configured position allocation %
                if (decision.signal == OrderSignal::Buy || decision.signal == OrderSignal::Sell)
                {
                    ExecutionFill fill = exchange.execute_order(decision.signal, tick.datetime, tick.close, cfg.allocation_pct);
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
