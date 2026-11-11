#pragma once

#include "quant/core/matrix.hpp"
#include "quant/models/interfaces.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>

class QuantConsoleReporter
{
public:
    static void render_header(const std::string& title)
    {
        std::cout << "\n========================================================================================================\n";
        std::cout << "  " << title << "\n";
        std::cout << "========================================================================================================\n";
    }

    static void print_forecast(const QuantForecast& f, std::size_t horizon_steps = 1)
    {
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "\n+------------------------------------------------------------------------------------------------------+\n";
        std::cout << "| MARKET FORECAST & RISK PROFILE (Horizon: +" << horizon_steps << " step(s))                                         |\n";
        std::cout << "+------------------------------------------------------------------------------------------------------+\n";
        std::cout << "  Current Asset Price       : $" << f.current_price << "\n";
        std::cout << "  Expected Price (E[P])     : $" << f.expected_price << "  (Return: " << (f.expected_return * 100.0) << "%)\n";
        std::cout << "  Most Likely Price (Mode)  : $" << f.mode_price << "\n";
        std::cout << "  95% Confidence Interval   : [$" << f.price_ci_lower_95 << " --- $" << f.price_ci_upper_95 << "]\n";
        std::cout << "--------------------------------------------------------------------------------------------------------\n";
        std::cout << "  Expected Volatility (σ)   : " << (f.expected_volatility * 100.0) << "%\n";
        std::cout << "  Bullish Probability P(r>0): " << (f.prob_positive_return * 100.0) << "%\n";
        std::cout << "  Value at Risk (95% VaR)   : " << (f.var_95 * 100.0) << "% log-loss\n";
        std::cout << "  Expected Shortfall (CVaR) : " << (f.cvar_95 * 100.0) << "% expected tail loss\n";
        std::cout << "  Kinetic Entropy (Uncert.) : " << f.distribution_entropy << " bits\n";
        std::cout << "+------------------------------------------------------------------------------------------------------+\n";
    }

    template <std::size_t Bins>
    static void print_distribution_ascii(double dx, double x_max, const std::array<double, Bins>& dist, const std::string& label)
    {
        std::cout << "\n--- Density Distribution: " << label << " ---\n";
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
            std::cout << std::right << std::setw(7) << std::fixed << std::setprecision(4) << center << " | " << std::left << std::setw(max_bar_width) << bar << " | "
                      << std::setw(6) << std::setprecision(2) << (prob * 100.0) << "%\n";
        }
    }

    template <std::size_t Bins>
    static void print_transition_matrix(const FlatMatrix<double, Bins>& W, std::size_t max_display_bins = 8)
    {
        std::size_t bins = std::min(Bins, max_display_bins);

        std::cout << "\n--- Dynamic Transition Kernel Matrix W(i -> j) (First " << bins << "x" << bins << " state subset) ---\n";
        std::cout << "       ";
        for (std::size_t j = 0; j < bins; ++j)
        {
            std::cout << "  Bin" << std::setw(2) << j;
        }
        std::cout << "\n";

        for (std::size_t i = 0; i < bins; ++i)
        {
            std::cout << "Bin" << std::setw(2) << i << " |";
            for (std::size_t j = 0; j < bins; ++j)
            {
                double val = W(i, j);
                std::cout << std::right << std::setw(7) << std::fixed << std::setprecision(3) << val;
            }
            std::cout << "\n";
        }
    }
};
