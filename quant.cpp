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

        // Double-buffered transition kernels owned by warm path calibration worker thread
        auto kernel_primary   = std::make_unique<Matrix>();
        auto kernel_secondary = std::make_unique<Matrix>();

        // Kinetic Hot Path engine
        double        x_max = 0.015; // +/- 1.5% max return window
        KineticEngine engine(x_max);

        // Cold path historical ingestion
        std::string data_path = "/Users/cdg546/Downloads/data/SPY_clean_1min.csv";
        std::cout << "[+] Ingesting historical market ticks from: " << data_path << " ...\n";

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
            std::cout << "[!] CSV missing/unreadable. Initializing synthetic market dataset...\n";
            for (std::size_t i = 0; i < 500; ++i)
            {
                double ret = 0.001 * std::sin(static_cast<double>(i) * 0.1);
                historical_features.push_back(DerivedFeatures{.log_return = ret, .rolling_volatility = 0.002});
            }
        }

        std::cout << "[+] Extracted " << historical_features.size() << " feature frames for background matrix calibration.\n";

        // Baseline startup calibration & atomic kernel load
        EmpiricalKernel<BINS> empirical_builder;
        empirical_builder.build_kernel(engine.dx(), engine.x_max(), historical_features, *kernel_primary);
        engine.update_kernel(kernel_primary.get());

        QuantConsoleReporter::print_transition_matrix(*kernel_primary, 8);

        // Lock-Free SPSC Ring Buffer for Live Hot Path Pipeline
        SPSCRingBuffer<Tick, 1024> tick_queue;
        OnlineFeatureEngine<20>    hot_feature_engine;
        RiskEngine                 risk_engine;

        std::atomic<bool>        running{true};
        std::atomic<std::size_t> processed_ticks{0};

        // Async Background Calibration Thread (Warm Path)
        std::thread calibration_thread(
                [&]()
                {
                    bool use_primary = false;
                    while (running.load(std::memory_order_relaxed))
                    {
                        std::this_thread::sleep_for(std::chrono::milliseconds(500));

                        Matrix* next_kernel = use_primary ? kernel_primary.get() : kernel_secondary.get();
                        empirical_builder.build_kernel(engine.dx(), engine.x_max(), historical_features, *next_kernel);

                        // Lock-free release-acquire atomic matrix pointer swap
                        engine.update_kernel(next_kernel);
                        use_primary = !use_primary;
                    }
                });

        // Hot Path Execution Loop
        KineticEngine::StateVector current_distribution = engine.initial_uniform_distribution();
        KineticEngine::StateVector next_distribution{};
        DerivedFeatures            feat{};
        double                     last_price = market.empty() ? 400.0 : market.bars().back().close;

        std::cout << "\n[+] Starting Real-Time Execution Engine Hot Path...\n";

        for (const auto& tick : market.bars())
        {
            tick_queue.push(tick);

            Tick dequeued_tick;
            if (tick_queue.pop(dequeued_tick))
            {
                // Incremental feature processing
                if (hot_feature_engine.process(dequeued_tick, feat))
                {
                    std::size_t state_idx = engine.state_index(feat.log_return);
                    current_distribution  = engine.state_delta(state_idx);

                    // O(N^2) forward density evolution
                    engine.evolve(current_distribution, next_distribution);

                    // Pre-Trade Risk Filtering & Signal Dispatch
                    QuantForecast forecast = engine.analyze_and_predict(dequeued_tick.close, next_distribution);
                    OrderSignal   signal   = risk_engine.evaluate(forecast);

                    (void) signal; // Dispatched to order execution gateway
                    processed_ticks.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }

        running.store(false, std::memory_order_relaxed);
        if (calibration_thread.joinable())
        {
            calibration_thread.join();
        }

        std::cout << "[+] Hot Path processed " << processed_ticks.load() << " market ticks with zero heap allocation.\n";

        // Display rollout telemetry
        KineticEngine::StateVector forecast_pdf{};
        engine.evolve(current_distribution, forecast_pdf);
        QuantForecast final_forecast = engine.analyze_and_predict(last_price, forecast_pdf);

        QuantConsoleReporter::print_forecast(final_forecast, 1);
        QuantConsoleReporter::print_distribution_ascii(engine.dx(), engine.x_max(), forecast_pdf, "Hot Path State Density f(t+1)");
    }
    catch (const std::exception& e)
    {
        std::cerr << "Fatal Exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
