#include "quant/core/config.hpp"
#include "quant/core/matrix.hpp"
#include "quant/core/ring_buffer.hpp"
#include "quant/core/types.hpp"
#include "quant/data/csv_reader.hpp"
#include "quant/features/feature_engine.hpp"
#include "quant/models/empirical_kernel.hpp"
#include "quant/models/kinetic_model.hpp"
#include "quant/risk/risk_manager.hpp"
#include "quant/telemetry/console_reporter.hpp"

#include <atomic>
#include <chrono>
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
        QuantConsoleReporter::render_header("DUAL-SPEED KINETIC QUANTITATIVE TRADING ENGINE");

        // load runtime configuration parameters from config loader
        EngineConfig cfg = ConfigLoader::load("quant.env");

        // double-buffered transition kernels owned by warm path calibration worker thread
        auto kernel_primary   = std::make_unique<Matrix>();
        auto kernel_secondary = std::make_unique<Matrix>();

        // kinetic hot path engine initialized with config return window
        double        x_max = cfg.x_max;
        KineticEngine engine(x_max);

        // cold path historical ingestion from configured data path
        std::string data_path = cfg.data_path;
        std::cout << "[+] Ingesting historical market ticks from: " << QuantConsoleReporter::CYAN << data_path << QuantConsoleReporter::RESET << " ...\n";

        MarketDataset                market;
        std::vector<DerivedFeatures> historical_features;
        try
        {
            market = CsvReader::load(data_path);
            historical_features.reserve(market.size());

            OnlineFeatureEngine<20> cold_fe;
            DerivedFeatures         feat{};
            for (const auto& tick : market.bars())
            {
                if (cold_fe.process(tick, feat))
                {
                    historical_features.push_back(feat);
                }
            }
        }
        catch (const std::exception&)
        {
            std::cout << QuantConsoleReporter::YELLOW << "[!] CSV missing/unreadable. Initializing synthetic market dataset...\n" << QuantConsoleReporter::RESET;
            for (std::size_t i = 0; i < 500; ++i)
            {
                double ret = 0.001 * std::sin(static_cast<double>(i) * 0.1);
                historical_features.push_back(DerivedFeatures{.log_return = ret, .rolling_volatility = 0.002});
            }
        }

        std::cout << "[+] Extracted " << QuantConsoleReporter::BOLD << historical_features.size() << QuantConsoleReporter::RESET
                  << " feature frames for background matrix calibration.\n";

        // baseline startup calibration & atomic kernel load
        EmpiricalKernel<BINS> empirical_builder;
        empirical_builder.build_kernel(engine.dx(), engine.x_max(), historical_features, *kernel_primary);
        engine.update_kernel(kernel_primary.get());

        QuantConsoleReporter::print_transition_matrix(*kernel_primary, 8);

        // lock-free spsc ring buffer for live hot path pipeline
        SPSCRingBuffer<Tick, 1024> tick_queue;
        OnlineFeatureEngine<20>    hot_feature_engine;
        RiskEngine                 risk_engine(cfg.risk_limits);

        std::atomic<bool>        running{true};
        std::atomic<std::size_t> processed_ticks{0};

        // telemetry counters
        std::size_t count_buys{0};
        std::size_t count_sells{0};
        std::size_t count_holds{0};
        std::size_t count_risk_vetoes{0};

        // async background calibration thread (warm path) using configured calibration frequency
        std::thread calibration_thread(
                [&]()
                {
                    bool use_primary = false;
                    while (running.load(std::memory_order_relaxed))
                    {
                        std::this_thread::sleep_for(std::chrono::milliseconds(cfg.async_calibration_ms));

                        Matrix* next_kernel = use_primary ? kernel_primary.get() : kernel_secondary.get();
                        empirical_builder.build_kernel(engine.dx(), engine.x_max(), historical_features, *next_kernel);

                        // lock-free release-acquire atomic matrix pointer swap
                        engine.update_kernel(next_kernel);
                        use_primary = !use_primary;
                    }
                });

        // hot path execution loop
        KineticEngine::StateVector current_distribution = engine.initial_uniform_distribution();
        KineticEngine::StateVector next_distribution{};
        DerivedFeatures            feat{};
        double                     last_price = market.empty() ? 400.0 : market.bars().back().close;

        QuantConsoleReporter::render_header("LIVE SIGNAL ANALYSIS & PRE-TRADE RISK STREAM");

        for (const auto& tick : market.bars())
        {
            tick_queue.push(tick);

            Tick dequeued_tick;
            if (tick_queue.pop(dequeued_tick))
            {
                // incremental feature processing
                if (hot_feature_engine.process(dequeued_tick, feat))
                {
                    std::size_t state_idx = engine.state_index(feat.log_return);
                    current_distribution  = engine.state_delta(state_idx);

                    // forward density evolution
                    engine.evolve(current_distribution, next_distribution);

                    // pre-trade risk filtering & signal dispatch
                    QuantForecast  forecast = engine.analyze_and_predict(dequeued_tick.close, next_distribution);
                    SignalDecision decision = risk_engine.evaluate(forecast);

                    // update signal telemetry statistics
                    if (decision.signal == OrderSignal::Buy)
                        count_buys++;
                    else if (decision.signal == OrderSignal::Sell)
                        count_sells++;
                    else
                        count_holds++;

                    if (decision.risk_veto)
                        count_risk_vetoes++;

                    // log active trade signals (buy / sell) or risk vetoes
                    if (decision.signal != OrderSignal::Hold || decision.risk_veto)
                    {
                        QuantConsoleReporter::print_signal_event(dequeued_tick.datetime, dequeued_tick.close, decision, forecast);
                    }

                    processed_ticks.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }

        running.store(false, std::memory_order_relaxed);
        if (calibration_thread.joinable())
        {
            calibration_thread.join();
        }

        // output complete trading execution telemetry summary
        QuantConsoleReporter::print_execution_summary(processed_ticks.load(), count_buys, count_sells, count_holds, count_risk_vetoes);

        // display final step rollout forecast
        KineticEngine::StateVector forecast_pdf{};
        engine.evolve(current_distribution, forecast_pdf);
        QuantForecast final_forecast = engine.analyze_and_predict(last_price, forecast_pdf);

        QuantConsoleReporter::print_forecast(final_forecast, 1);
        QuantConsoleReporter::print_distribution_ascii(engine.dx(), engine.x_max(), forecast_pdf, "Hot Path Final State Density f(t+1)");
    }
    catch (const std::exception& e)
    {
        std::cerr << QuantConsoleReporter::RED << "Fatal Exception: " << e.what() << QuantConsoleReporter::RESET << "\n";
        return 1;
    }

    return 0;
}
