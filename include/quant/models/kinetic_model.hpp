#pragma once

#include "quant/core/matrix.hpp"
#include "quant/models/interfaces.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>

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
                continue; // Early exit for sparse probabilities

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

        double mean_r   = 0.0;
        double var_r    = 0.0;
        double prob_pos = 0.0;
        double entropy  = 0.0;

        std::size_t max_idx  = 0;
        double      max_prob = -1.0;

        for (std::size_t i = 0; i < Bins; ++i)
        {
            double r = bin_center(i);
            double p = dist[i];

            mean_r += p * r;
            if (r > 0.0)
                prob_pos += p;
            if (p > max_prob)
            {
                max_prob = p;
                max_idx  = i;
            }
            if (p > 1e-12)
            {
                entropy -= p * std::log2(p);
            }
        }

        for (std::size_t i = 0; i < Bins; ++i)
        {
            double diff = bin_center(i) - mean_r;
            var_r += dist[i] * diff * diff;
        }

        forecast.expected_return      = mean_r;
        forecast.expected_volatility  = std::sqrt(var_r);
        forecast.prob_positive_return = prob_pos;
        forecast.distribution_entropy = entropy;

        forecast.expected_price = current_price * std::exp(mean_r);
        forecast.mode_price     = current_price * std::exp(bin_center(max_idx));

        double cum_prob   = 0.0;
        double var_95_val = bin_center(0);
        double cvar_sum   = 0.0;
        double cvar_prob  = 0.0;

        for (std::size_t i = 0; i < Bins; ++i)
        {
            cum_prob += dist[i];
            if (cum_prob <= 0.05)
            {
                cvar_sum += dist[i] * bin_center(i);
                cvar_prob += dist[i];
            }
            if (cum_prob >= 0.05 && forecast.var_95 == 0.0)
            {
                var_95_val      = bin_center(i);
                forecast.var_95 = var_95_val;
            }
            if (cum_prob >= 0.025 && forecast.price_ci_lower_95 == 0.0)
            {
                forecast.price_ci_lower_95 = current_price * std::exp(bin_center(i));
            }
            if (cum_prob >= 0.975 && forecast.price_ci_upper_95 == 0.0)
            {
                forecast.price_ci_upper_95 = current_price * std::exp(bin_center(i));
            }
        }

        if (forecast.price_ci_upper_95 == 0.0)
        {
            forecast.price_ci_upper_95 = current_price * std::exp(bin_center(Bins - 1));
        }

        forecast.cvar_95 = (cvar_prob > 0.0) ? (cvar_sum / cvar_prob) : var_95_val;
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