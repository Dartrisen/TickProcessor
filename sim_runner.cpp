#include "quant/core/config.hpp"
#include "quant/core/dispatch_table.hpp"
#include "quant/execution/simulated_exchange.hpp"
#include "quant/market/csv_reader.hpp"
#include "quant/market/market_feed.hpp"
#include "quant/models/kernels.hpp"
#include "quant/session/trading_session.hpp"
#include "quant/telemetry/console_reporter.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>

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

template <template <std::size_t> class KernelT, std::size_t BINS, std::size_t... Idx>
bool try_dispatch_window_indices(const EngineConfig& cfg, std::index_sequence<Idx...>)
{
    return ((cfg.volatility_window == DispatchLimits::Windows[Idx] ? (run_simulation<KernelT, BINS, DispatchLimits::Windows[Idx]>(cfg), true) : false) || ...);
}

template <template <std::size_t> class KernelT, std::size_t... Bins>
bool try_dispatch_bins(const EngineConfig& cfg, std::index_sequence<Bins...>)
{
    using WindowIndices = std::make_index_sequence<DispatchLimits::Windows.size()>;
    return ((cfg.bins == Bins ? try_dispatch_window_indices<KernelT, Bins>(cfg, WindowIndices{}) : false) || ...);
}

template <std::size_t... Idx>
bool dispatch_simulation(const EngineConfig& cfg, std::index_sequence<Idx...>)
{
    using BinSeq = std::index_sequence<DispatchLimits::Bins[Idx]...>;
    switch (cfg.kernel_type)
    {
        case KernelType::Empirical:
            return try_dispatch_bins<EmpiricalKernel>(cfg, BinSeq{});
        case KernelType::Gaussian:
            return try_dispatch_bins<GaussianDiffusionKernel>(cfg, BinSeq{});
        case KernelType::Dirichlet:
            return try_dispatch_bins<AdaptiveDirichletKernel>(cfg, BinSeq{});
        case KernelType::PhaseSpace:
            return try_dispatch_bins<FactorizedPhaseSpaceKernel>(cfg, BinSeq{});
    }
    return false;
}

int main()
{
    try
    {
        QuantConsoleReporter::render_header("LIVE API BACKTEST & PAPER TRADING SIMULATION ENGINE");
        const EngineConfig cfg = ConfigLoader::load("quant.env");

        if (!dispatch_simulation(cfg, std::make_index_sequence<DispatchLimits::Bins.size()>{}))
        {
            throw std::runtime_error("Unsupported MODEL_KERNEL, MODEL_BINS, or FEATURE_VOLATILITY_WINDOW.");
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << QuantConsoleReporter::RED << "Fatal Simulation Error: " << e.what() << QuantConsoleReporter::RESET << "\n";
        return 1;
    }

    return 0;
}
