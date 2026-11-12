#pragma once

#include "quant/core/matrix.hpp"
#include "quant/models/interfaces.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>

template <std::size_t Bins>
class HighPerformanceKineticEngine
{
public:
    using StateVector = std::array<double, Bins>;
    using Matrix      = FlatMatrix<double, Bins>;

    explicit HighPerformanceKineticEngine(double x_max) : x_max_(x_max), dx_(2.0 * x_max / Bins) {}

    // HOT PATH: Forward density propagation ($f_{t+1} = W * f_t$)
    void evolve(const StateVector& in_pdf, StateVector& out_pdf) const noexcept
    {
        out_pdf.fill(0.0);
        const auto* W = active_kernel_.load(std::memory_order_acquire);
        if (!W)
            return;

        for (std::size_t i = 0; i < Bins; ++i)
        {
            const double f_i = in_pdf[i];
            if (f_i <= 1e-12)
                continue;

            for (std::size_t j = 0; j < Bins; ++j)
            {
                out_pdf[j] += f_i * (*W)(i, j);
            }
        }
    }

    // WARM/COLD PATH: Atomic kernel swap from background calibration thread
    void update_kernel(const Matrix* new_kernel) noexcept
    {
        active_kernel_.store(new_kernel, std::memory_order_release);
    }

    [[nodiscard]] std::size_t state_index(double return_val) const noexcept
    {
        double clamped = std::clamp(return_val, -x_max_, x_max_);
        return std::min(static_cast<std::size_t>((clamped + x_max_) / dx_), Bins - 1);
    }

    [[nodiscard]] double bin_center(std::size_t index) const noexcept
    {
        return -x_max_ + (static_cast<double>(index) + 0.5) * dx_;
    }

    [[nodiscard]] StateVector state_delta(std::size_t state_idx) const noexcept
    {
        StateVector f{};
        f.fill(0.0);
        if (state_idx < Bins)
            f[state_idx] = 1.0;
        return f;
    }

    [[nodiscard]] StateVector initial_uniform_distribution() const noexcept
    {
        StateVector dist{};
        dist.fill(1.0 / static_cast<double>(Bins));
        return dist;
    }

    [[nodiscard]] QuantForecast analyze_and_predict(double current_price, const StateVector& dist) const noexcept
    {
        QuantForecast forecast;
        forecast.current_price = current_price;

        double total_probability         = 0.0;
        double mean_log_return           = 0.0;
        double expected_price_multiplier = 0.0;
        double prob_positive             = 0.0;
        double entropy                   = 0.0;

        std::size_t mode_index = 0;
        double      mode_prob  = -1.0;

        for (std::size_t i = 0; i < Bins; ++i)
        {
            const double r = bin_center(i);
            const double p = std::max(0.0, dist[i]);

            total_probability += p;
            mean_log_return += p * r;
            expected_price_multiplier += p * std::exp(r);

            if (r > 0.0)
                prob_positive += p;

            if (p > mode_prob)
            {
                mode_prob  = p;
                mode_index = i;
            }

            if (p > 1e-12)
                entropy -= p * std::log2(p);
        }

        // Defensive normalization if a future kernel implementation produces
        // a nearly-normalized vector rather than an exact probability vector.
        if (total_probability <= 1e-12)
            return forecast;

        if (std::abs(total_probability - 1.0) > 1e-10)
        {
            mean_log_return /= total_probability;
            expected_price_multiplier /= total_probability;
            prob_positive /= total_probability;
            entropy = 0.0;

            for (std::size_t i = 0; i < Bins; ++i)
            {
                const double p = std::max(0.0, dist[i]) / total_probability;
                if (p > 1e-12)
                    entropy -= p * std::log2(p);
            }
        }

        double variance = 0.0;
        for (std::size_t i = 0; i < Bins; ++i)
        {
            const double p    = std::max(0.0, dist[i]) / total_probability;
            const double diff = bin_center(i) - mean_log_return;
            variance += p * diff * diff;
        }

        auto return_quantile = [&](double probability) noexcept
        {
            double cumulative = 0.0;

            for (std::size_t i = 0; i < Bins; ++i)
            {
                cumulative += std::max(0.0, dist[i]) / total_probability;
                if (cumulative >= probability)
                    return bin_center(i);
            }

            return bin_center(Bins - 1);
        };

        // Average log-return in the worst 5% tail, including a fractional
        // contribution from the bin that crosses the 5% probability boundary.
        constexpr double tail_probability           = 0.05;
        double           remaining_tail_probability = tail_probability;
        double           tail_return_sum            = 0.0;

        for (std::size_t i = 0; i < Bins && remaining_tail_probability > 1e-12; ++i)
        {
            const double p                = std::max(0.0, dist[i]) / total_probability;
            const double used_probability = std::min(p, remaining_tail_probability);

            tail_return_sum += used_probability * bin_center(i);
            remaining_tail_probability -= used_probability;
        }

        const double q025 = return_quantile(0.025);
        const double q05  = return_quantile(0.05);
        const double q975 = return_quantile(0.975);

        forecast.expected_return      = mean_log_return;
        forecast.expected_volatility  = std::sqrt(std::max(0.0, variance));
        forecast.prob_positive_return = prob_positive;
        forecast.distribution_entropy = entropy;

        // Arithmetic expectation: E[P(t+1)] = P(t) * sum(p_i * exp(r_i)).
        forecast.expected_price = current_price * expected_price_multiplier;
        forecast.mode_price     = current_price * std::exp(bin_center(mode_index));

        forecast.price_ci_lower_95 = current_price * std::exp(q025);
        forecast.price_ci_upper_95 = current_price * std::exp(q975);

        // Store VaR/CVaR as POSITIVE loss magnitudes, not negative returns.
        forecast.var_95  = std::max(0.0, -q05);
        forecast.cvar_95 = std::max(0.0, -(tail_return_sum / tail_probability));

        return forecast;
    }

    [[nodiscard]] constexpr std::size_t bins() const noexcept
    {
        return Bins;
    }
    [[nodiscard]] double x_max() const noexcept
    {
        return x_max_;
    }
    [[nodiscard]] double dx() const noexcept
    {
        return dx_;
    }

private:
    double                     x_max_;
    double                     dx_;
    std::atomic<const Matrix*> active_kernel_{nullptr};
};
