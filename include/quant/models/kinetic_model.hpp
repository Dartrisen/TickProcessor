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

        // Left-tail quantile (accumulated from lowest return bin upward)
        auto lower_return_quantile = [&](double tail_prob) noexcept
        {
            double cumulative = 0.0;

            for (std::size_t i = 0; i < Bins; ++i)
            {
                cumulative += std::max(0.0, dist[i]) / total_probability;
                if (cumulative >= tail_prob)
                    return bin_center(i);
            }

            return bin_center(Bins - 1);
        };

        // Right-tail quantile (accumulated from highest return bin downward)
        auto upper_return_quantile = [&](double tail_prob) noexcept
        {
            double cumulative = 0.0;

            for (std::size_t k = 0; k < Bins; ++k)
            {
                const std::size_t idx = Bins - 1 - k;
                cumulative += std::max(0.0, dist[idx]) / total_probability;
                if (cumulative >= tail_prob)
                    return bin_center(idx);
            }

            return bin_center(0);
        };

        constexpr double tail_probability = 0.05;

        // 1. Long-side tail risk (worst 5% downside returns)
        double lower_remaining_prob  = tail_probability;
        double lower_tail_return_sum = 0.0;

        for (std::size_t i = 0; i < Bins && lower_remaining_prob > 1e-12; ++i)
        {
            const double p         = std::max(0.0, dist[i]) / total_probability;
            const double used_prob = std::min(p, lower_remaining_prob);

            lower_tail_return_sum += used_prob * bin_center(i);
            lower_remaining_prob -= used_prob;
        }

        // 2. Short-side tail risk (worst 5% upside returns)
        double upper_remaining_prob  = tail_probability;
        double upper_tail_return_sum = 0.0;

        for (std::size_t k = 0; k < Bins && upper_remaining_prob > 1e-12; ++k)
        {
            const std::size_t idx       = Bins - 1 - k;
            const double      p         = std::max(0.0, dist[idx]) / total_probability;
            const double      used_prob = std::min(p, upper_remaining_prob);

            upper_tail_return_sum += used_prob * bin_center(idx);
            upper_remaining_prob -= used_prob;
        }

        const double q025 = lower_return_quantile(0.025);
        const double q05  = lower_return_quantile(0.05);
        const double q95  = upper_return_quantile(0.05);
        const double q975 = upper_return_quantile(0.025);

        forecast.expected_return      = mean_log_return;
        forecast.expected_volatility  = std::sqrt(std::max(0.0, variance));
        forecast.prob_positive_return = prob_positive;
        forecast.distribution_entropy = entropy;

        forecast.expected_price = current_price * expected_price_multiplier;
        forecast.mode_price     = current_price * std::exp(bin_center(mode_index));

        forecast.price_ci_lower_95 = current_price * std::exp(q025);
        forecast.price_ci_upper_95 = current_price * std::exp(q975);

        // Positive loss magnitudes for both position directions
        forecast.long_var_95   = std::max(0.0, -q05);
        forecast.long_cvar_95  = std::max(0.0, -(lower_tail_return_sum / tail_probability));
        forecast.short_var_95  = std::max(0.0, q95);
        forecast.short_cvar_95 = std::max(0.0, upper_tail_return_sum / tail_probability);

        forecast.var_95  = forecast.long_var_95;
        forecast.cvar_95 = forecast.long_cvar_95;

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
