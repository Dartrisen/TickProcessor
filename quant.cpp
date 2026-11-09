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
        // setup online tick streaming processor
        TickProcessor                processor(10); // 10-period rolling volatility window
        std::vector<DerivedFeatures> feature_stream;

        processor.register_callback([&](const Tick& /*t*/, const DerivedFeatures& f) { feature_stream.push_back(f); });

        // load market data
        auto market = CsvReader::load("/Users/cdg546/Downloads/data/SPY_clean_1min.csv");
        std::cout << "Successfully loaded " << market.size() << " market ticks.\n";

        // Stream ticks through online processor
        for (const auto& tick : market.bars())
        {
            processor.process(tick);
        }

        // initialize kinetic model with default empirical transition kernel
        std::size_t bins  = 20;
        double      x_max = 0.05; // +/- 5% return range

        KineticModel model(bins, x_max, std::make_unique<EmpiricalKernel>());
        model.fit(feature_stream);

        std::cout << "\n[Model 1] Fitted Empirical Kinetic Model successfully across " << bins << " state space bins.\n";

        //eEvolve an initial state distribution
        auto f_t0 = model.initial_distribution();
        auto f_t1 = model.evolve(f_t0);
        std::cout << "Evolved state vector (bin 10 center): " << f_t1[10] << "\n";

        // dynamically swap kernel to gaussian diffusion model
        std::cout << "\n--> Swapping Model Kernel to Parametric Gaussian Diffusion Kernel...\n";
        model.set_kernel(std::make_unique<GaussianDiffusionKernel>(0.0001, 0.005));
        model.fit(feature_stream);

        auto f_t1_gaussian = model.evolve(f_t0);
        std::cout << "[Model 2] Evolved state vector with Gaussian Kernel (bin 10 center): " << f_t1_gaussian[10] << "\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "Execution Exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
