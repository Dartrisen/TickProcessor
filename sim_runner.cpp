#include "quant/core/config.hpp"
#include "quant/core/engine_dispatcher.hpp"
#include "quant/execution/simulated_exchange.hpp"
#include "quant/market/csv_reader.hpp"
#include "quant/market/market_feed.hpp"
#include "quant/session/trading_session.hpp"
#include "quant/telemetry/console_reporter.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>

template <template <std::size_t> class KernelT, std::size_t BINS, std::size_t VOL_WINDOW>
void run_simulation(const EngineConfig& cfg)
{
    const FeeStructure fees{
            .commission_rate     = cfg.commission_rate,
            .fixed_fee_per_trade = cfg.fixed_fee,
            .slippage_bps        = cfg.slippage_bps,
    };
    SimulatedExchange broker(cfg.initial_capital, fees, cfg.bars_per_year);

    std::cout << "[+] Connecting to market feed: " << QuantConsoleReporter::CYAN << cfg.data_path << QuantConsoleReporter::RESET << " ...\n";
    const MarketDataset dataset = CsvReader::load(cfg.data_path);
    if (dataset.size() < 10)
    {
        throw std::runtime_error("Insufficient market bars loaded for simulation.");
    }

    const std::size_t train_size  = std::max<std::size_t>(2, static_cast<std::size_t>(static_cast<double>(dataset.size()) * cfg.train_split_ratio));
    const auto        warmup_bars = dataset.bars().subspan(0, train_size);
    const auto        oos_bars    = dataset.bars().subspan(train_size);

    LiveRingBufferFeed<4096> feed(std::string(dataset.source()));

    std::cout << "[+] Loaded " << QuantConsoleReporter::BOLD << dataset.size() << QuantConsoleReporter::RESET << " total ticks (Source: " << feed.source_name()
              << " | In-Sample Calibration: " << warmup_bars.size() << " | Out-of-Sample Test: " << oos_bars.size() << ").\n";

    std::jthread csv_producer(
            [&feed, oos_bars](std::stop_token stoken)
            {
                for (const auto& bar : oos_bars)
                {
                    if (stoken.stop_requested())
                        break;
                    feed.push_blocking(bar, stoken);
                }
                feed.mark_exhausted();
            });

    QuantConsoleReporter                             reporter;
    KineticTradingSession<BINS, VOL_WINDOW, KernelT> session(cfg, broker, reporter);
    session.warmup(warmup_bars);
    session.run(feed);
}

int main()
{
    try
    {
        QuantConsoleReporter::render_header("LIVE API BACKTEST & PAPER TRADING SIMULATION ENGINE");
        const EngineConfig cfg = ConfigLoader::load("quant.env");

        EngineDispatcher::dispatch(cfg, []<template <std::size_t> class KernelT, std::size_t BINS, std::size_t VOL_WINDOW>(const EngineConfig& c)
                                   { run_simulation<KernelT, BINS, VOL_WINDOW>(c); });
    }
    catch (const std::exception& e)
    {
        std::cerr << QuantConsoleReporter::RED << "Fatal Simulation Error: " << e.what() << QuantConsoleReporter::RESET << "\n";
        return 1;
    }

    return 0;
}
