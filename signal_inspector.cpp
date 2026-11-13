#include "quant/core/config.hpp"
#include "quant/core/engine_dispatcher.hpp"
#include "quant/core/types.hpp"
#include "quant/features/feature_engine.hpp"
#include "quant/market/csv_reader.hpp"
#include "quant/market/market_feed.hpp"
#include "quant/models/kinetic_model.hpp"
#include "quant/models/walk_forward_calibrator.hpp"
#include "quant/risk/risk_manager.hpp"
#include "quant/telemetry/console_reporter.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

template <template <std::size_t> class KernelT, std::size_t BINS, std::size_t VOL_WINDOW>
void run_signal_inspector(const EngineConfig& cfg)
{
    using KineticEngine = HighPerformanceKineticEngine<BINS>;

    std::cout << "[+] Ingesting historical market ticks from: " << QuantConsoleReporter::CYAN << cfg.data_path << QuantConsoleReporter::RESET << " ...\n";
    const MarketDataset dataset = CsvReader::load(cfg.data_path);
    if (dataset.size() < 10)
    {
        throw std::runtime_error("Insufficient market bars loaded for signal inspection.");
    }

    const std::size_t train_size  = std::max<std::size_t>(2, static_cast<std::size_t>(static_cast<double>(dataset.size()) * cfg.train_split_ratio));
    const auto        warmup_bars = dataset.bars().subspan(0, train_size);
    const auto        oos_bars    = dataset.bars().subspan(train_size);

    OnlineFeatureEngine<VOL_WINDOW> feature_engine;
    std::vector<DerivedFeatures>    warmup_features;
    warmup_features.reserve(warmup_bars.size());

    DerivedFeatures feat{};
    for (const auto& bar : warmup_bars)
    {
        if (feature_engine.process(bar, feat))
        {
            warmup_features.push_back(feat);
        }
    }

    std::cout << "[+] Extracted " << QuantConsoleReporter::BOLD << warmup_features.size() << QuantConsoleReporter::RESET
              << " in-sample feature frames (Out-of-Sample stream: " << oos_bars.size() << " bars).\n";

    KineticEngine                        engine(cfg.x_max);
    WalkForwardCalibrator<BINS, KernelT> calibrator(cfg.recalibration_bars);
    calibrator.initialize(engine, std::move(warmup_features));

    QuantConsoleReporter::print_transition_matrix(calibrator.active_kernel(), 8);

    LiveRingBufferFeed<4096> feed(std::string(dataset.source()));
    std::jthread             producer(
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

    RiskEngine                          risk_engine(cfg.risk_limits);
    typename KineticEngine::StateVector current_distribution = engine.initial_uniform_distribution();
    typename KineticEngine::StateVector next_distribution{};

    std::size_t processed_ticks{0};
    std::size_t count_buys{0};
    std::size_t count_sells{0};
    std::size_t count_holds{0};
    std::size_t count_risk_vetoes{0};
    double      last_price = warmup_bars.back().close;

    QuantConsoleReporter::render_header("LIVE SIGNAL ANALYSIS & PRE-TRADE RISK STREAM");

    Tick tick{};
    while (!feed.is_exhausted())
    {
        if (!feed.next(tick))
        {
            std::this_thread::yield();
            continue;
        }

        last_price = tick.close;
        if (!feature_engine.process(tick, feat))
            continue;

        current_distribution = engine.state_delta(engine.state_index(feat.log_return));
        engine.evolve(current_distribution, next_distribution);

        const QuantForecast  forecast = engine.analyze_and_predict_horizon(tick.close, next_distribution, cfg.forecast_horizon, 0.0);
        const SignalDecision decision = risk_engine.evaluate(forecast);

        if (decision.signal == OrderSignal::Buy)
            ++count_buys;
        else if (decision.signal == OrderSignal::Sell)
            ++count_sells;
        else
            ++count_holds;

        if (decision.risk_veto)
            ++count_risk_vetoes;

        if (decision.signal != OrderSignal::Hold || decision.risk_veto)
        {
            QuantConsoleReporter::print_signal_event(tick.datetime.view(), tick.close, decision, forecast);
        }

        calibrator.on_new_feature(engine, feat);
        ++processed_ticks;
    }

    QuantConsoleReporter::print_execution_summary(processed_ticks, count_buys, count_sells, count_holds, count_risk_vetoes);

    typename KineticEngine::StateVector forecast_pdf{};
    engine.evolve_steps(current_distribution, forecast_pdf, cfg.forecast_horizon);

    const QuantForecast final_forecast = engine.analyze_and_predict_horizon(last_price, next_distribution, cfg.forecast_horizon, 0.0);
    QuantConsoleReporter::print_forecast(final_forecast, cfg.forecast_horizon);
    QuantConsoleReporter::print_distribution_ascii(engine.dx(), engine.x_max(), forecast_pdf, "Hot Path Final State Density f(t+H)");
}

int main()
{
    try
    {
        QuantConsoleReporter::render_header("DUAL-SPEED KINETIC SIGNAL & RISK DIAGNOSTIC ENGINE");
        const EngineConfig cfg = ConfigLoader::load("quant.env");

        EngineDispatcher::dispatch(cfg, []<template <std::size_t> class KernelT, std::size_t BINS, std::size_t VOL_WINDOW>(const EngineConfig& c)
                                   { run_signal_inspector<KernelT, BINS, VOL_WINDOW>(c); });
    }
    catch (const std::exception& e)
    {
        std::cerr << QuantConsoleReporter::RED << "Fatal Exception: " << e.what() << QuantConsoleReporter::RESET << "\n";
        return 1;
    }

    return 0;
}
