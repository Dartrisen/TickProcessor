#include "include/console_reporter.hpp"
#include "include/csv_reader.hpp"
#include "include/kinetic_model.hpp"
#include "include/tick_processor.hpp"

#include <iostream>
#include <memory>
#include <vector>

int main()
{
    try
    {
        QuantConsoleReporter::render_header("KINETIC QUANT MARKET MODEL & PREDICTION ENGINE");

        // 1. Setup online tick processing pipeline
        TickProcessor                processor(20);
        std::vector<DerivedFeatures> feature_stream;

        processor.register_callback([&](const Tick& /*t*/, const DerivedFeatures& f) { feature_stream.push_back(f); });

        // 2. Load dataset
        std::string data_path = "/Users/cdg546/Downloads/data/SPY_clean_1min.csv";
        std::cout << "[+] Ingesting historical ticks from: " << data_path << " ...\n";

        MarketDataset market;
        try
        {
            market = CsvReader::load(data_path);
        }
        catch (const std::exception&)
        {
            std::cout << "[!] Benchmark CSV file missing. Quitting...\n";
            return 1;
        }

        std::cout << "[+] Streamed " << market.size() << " market ticks into processing pipeline.\n";

        for (const auto& tick : market.bars())
        {
            processor.process(tick);
        }

        const double last_price      = market.bars().back().close;
        const double last_log_return = feature_stream.back().log_return;

        // 3. Initialize & Calibrate Kinetic Model with Empirical Kernel
        std::size_t bins  = 16;
        double      x_max = 0.015; // +/- 1.5% max return window

        KineticModel model(bins, x_max, std::make_unique<EmpiricalKernel>());
        model.fit(feature_stream);

        // 4. Inspect Current Market State & Dynamic Transitions
        std::size_t current_state_idx    = model.state_index(last_log_return);
        auto        current_distribution = model.state_delta(current_state_idx);

        std::cout << "\n[+] Current Market State: Return = " << (last_log_return * 100.0) << "% | State Bin Index = " << current_state_idx << "\n";

        QuantConsoleReporter::print_transition_matrix(model, 8);

        // 5. Predict Next Step (t + 1)
        auto f_t1        = model.evolve(current_distribution);
        auto forecast_t1 = model.analyze_and_predict(last_price, f_t1);

        QuantConsoleReporter::print_forecast(forecast_t1, 1);
        QuantConsoleReporter::print_distribution_ascii(model, f_t1, "1-Step Ahead Evolved State Density f(t+1)");

        // 6. Multi-Step Prediction Rollout (t + 5 Steps ahead propagation)
        auto f_t5 = f_t1;
        for (int k = 0; k < 4; ++k)
        {
            f_t5 = model.evolve(f_t5);
        }
        auto forecast_t5 = model.analyze_and_predict(last_price, f_t5);
        QuantConsoleReporter::print_forecast(forecast_t5, 5);

        // 7. Stationary Equilibrium & Model Swapping
        std::cout << "\n";
        QuantConsoleReporter::render_header("MODEL SWAP & EQUILIBRIUM ANALYSIS");

        auto stationary_dist     = model.stationary_distribution();
        auto stationary_forecast = model.analyze_and_predict(last_price, stationary_dist);

        std::cout << "\n[+] Empirical Stationary Distribution (Long-term Market Balance):\n";
        std::cout << "  Equilibrium Expected Return : " << (stationary_forecast.expected_return * 100.0) << "%\n";
        std::cout << "  Equilibrium Volatility      : " << (stationary_forecast.expected_volatility * 100.0) << "%\n";

        // Dynamic Swap to Gaussian Parametric Diffusion Kernel
        std::cout << "\n[->] Swapping Model Kernel dynamically to Gaussian Diffusion Kernel...\n";
        model.set_kernel(std::make_unique<GaussianDiffusionKernel>(0.0002, 0.003));
        model.fit(feature_stream);

        auto f_t1_gaussian     = model.evolve(current_distribution);
        auto forecast_gaussian = model.analyze_and_predict(last_price, f_t1_gaussian);

        QuantConsoleReporter::print_forecast(forecast_gaussian, 1);
        QuantConsoleReporter::print_distribution_ascii(model, f_t1_gaussian, "Parametric Gaussian Kernel State Density f(t+1)");
    }
    catch (const std::exception& e)
    {
        std::cerr << "Fatal Exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
