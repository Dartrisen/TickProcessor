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

    static constexpr std::size_t MaxHorizon = 10;

    explicit HighPerformanceKineticEngine(double x_max) : x_max_(x_max), dx_(2.0 * x_max / Bins) {}

    // HOT PATH: Single-step forward density propagation ($f_{t+1} = W * f_t$)
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

    // Multi-step Chapman-Kolmogorov state propagation ($f_{t+k} = W^k * f_t$)
    void evolve_steps(const StateVector& in_pdf, StateVector& out_pdf, std::size_t steps) const noexcept
    {
        if (steps == 0)
        {
            out_pdf = in_pdf;
            return;
        }

        evolve(in_pdf, out_pdf);
        if (steps == 1)
            return;

        StateVector scratch{};
        for (std::size_t s = 1; s < steps; ++s)
        {
            scratch = out_pdf;
            evolve(scratch, out_pdf);
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
        return analyze_and_predict_horizon(current_price, dist, 1, 0.0);
    }

    // Exact H-step cumulative return & dual-tail risk evaluation via Markov Additive Process DP.
    // `step1_dist` is the marginal state PDF for bar t+1; `realized_entry_gap = ln(O_{t+1} / C_t)`
    // is subtracted from step 1 so forecasts are conditioned on the actual entry price.
    [[nodiscard]] QuantForecast analyze_and_predict_horizon(double entry_price, const StateVector& step1_dist, std::size_t horizon_steps = 1,
                                                            double realized_entry_gap = 0.0) const noexcept
    {
        const auto*       W           = active_kernel_.load(std::memory_order_acquire);
        const std::size_t effective_H = (W != nullptr) ? std::clamp<std::size_t>(horizon_steps, 1, MaxHorizon) : 1;

        QuantForecast forecast;
        forecast.horizon_steps    = effective_H;
        forecast.entry_gap_return = realized_entry_gap;
        forecast.current_price    = entry_price;

        double total_probability = 0.0;
        for (std::size_t i = 0; i < Bins; ++i)
        {
            total_probability += std::max(0.0, step1_dist[i]);
        }
        if (total_probability <= 1e-12)
            return forecast;

        StateVector norm_step1{};
        for (std::size_t i = 0; i < Bins; ++i)
        {
            norm_step1[i] = std::max(0.0, step1_dist[i]) / total_probability;
        }

        // Stack-allocated ping-pong DP tables indexed by (cumulative_bin_sum, terminal_state).
        // Because bin centers are equidistant (-x_max + (i + 0.5)*dx), every H-step path with
        // index sum K = i_1 + ... + i_H has exact cumulative return:
        //   R(H, K) = H * (-x_max + 0.5 * dx) + K * dx - realized_entry_gap
        // which is strictly increasing in K (naturally sorted without std::sort).
        constexpr std::size_t MaxBinSum = MaxHorizon * (Bins - 1);
        constexpr std::size_t TableSize = (MaxBinSum + 1) * Bins;

        std::array<double, TableSize> dp_curr{};
        std::array<double, TableSize> dp_next{};

        // Step s = 1 initialization: K = j, terminal state = j
        for (std::size_t j = 0; j < Bins; ++j)
        {
            dp_curr[j * Bins + j] = norm_step1[j];
        }

        // Propagate exact joint (cumulative_sum, state) distribution from step 2 through H
        for (std::size_t step = 2; step <= effective_H; ++step)
        {
            const std::size_t prev_max_k = (step - 1) * (Bins - 1);
            const std::size_t next_max_k = step * (Bins - 1);

            std::fill_n(dp_next.begin(), (next_max_k + 1) * Bins, 0.0);

            for (std::size_t k = 0; k <= prev_max_k; ++k)
            {
                const std::size_t k_offset = k * Bins;
                for (std::size_t i = 0; i < Bins; ++i)
                {
                    const double p = dp_curr[k_offset + i];
                    if (p <= 1e-15)
                        continue;

                    for (std::size_t j = 0; j < Bins; ++j)
                    {
                        dp_next[(k + j) * Bins + j] += p * (*W)(i, j);
                    }
                }
            }

            const std::size_t active_entries = (next_max_k + 1) * Bins;
            std::copy_n(dp_next.begin(), active_entries, dp_curr.begin());
        }

        // Marginalize out terminal state j to obtain sorted cumulative return distribution,
        // and marginalize out index sum k to obtain terminal state PDF f(t+H) for entropy.
        const std::size_t num_sums = effective_H * (Bins - 1) + 1;
        const double      base_ret = static_cast<double>(effective_H) * (-x_max_ + 0.5 * dx_) - realized_entry_gap;

        std::array<double, MaxBinSum + 1> sum_probs{};
        StateVector                       terminal_pdf{};
        double                            norm_sum = 0.0;

        for (std::size_t k = 0; k < num_sums; ++k)
        {
            const std::size_t k_offset = k * Bins;
            double            p_k      = 0.0;
            for (std::size_t j = 0; j < Bins; ++j)
            {
                const double p_kj = dp_curr[k_offset + j];
                p_k += p_kj;
                terminal_pdf[j] += p_kj;
            }
            sum_probs[k] = p_k;
            norm_sum += p_k;
        }

        if (norm_sum <= 1e-12)
            return forecast;

        double terminal_entropy = 0.0;
        for (std::size_t j = 0; j < Bins; ++j)
        {
            const double p_state = terminal_pdf[j] / norm_sum;
            if (p_state > 1e-12)
                terminal_entropy -= p_state * std::log2(p_state);
        }

        auto return_at = [&](std::size_t k) noexcept -> double { return base_ret + static_cast<double>(k) * dx_; };

        double      mean_log_return           = 0.0;
        double      expected_price_multiplier = 0.0;
        double      prob_positive             = 0.0;
        std::size_t mode_k                    = 0;
        double      mode_prob                 = -1.0;

        for (std::size_t k = 0; k < num_sums; ++k)
        {
            const double p = sum_probs[k] / norm_sum;
            const double r = return_at(k);

            mean_log_return += p * r;
            expected_price_multiplier += p * std::exp(r);

            if (r > 0.0)
                prob_positive += p;

            if (p > mode_prob)
            {
                mode_prob = p;
                mode_k    = k;
            }
        }

        double variance = 0.0;
        for (std::size_t k = 0; k < num_sums; ++k)
        {
            const double p    = sum_probs[k] / norm_sum;
            const double diff = return_at(k) - mean_log_return;
            variance += p * diff * diff;
        }

        auto lower_quantile = [&](double target_prob) noexcept -> double
        {
            double cumulative = 0.0;
            for (std::size_t k = 0; k < num_sums; ++k)
            {
                cumulative += sum_probs[k] / norm_sum;
                if (cumulative >= target_prob)
                    return return_at(k);
            }
            return return_at(num_sums - 1);
        };

        auto upper_quantile = [&](double target_prob) noexcept -> double
        {
            double cumulative = 0.0;
            for (std::size_t offset = 0; offset < num_sums; ++offset)
            {
                const std::size_t k = num_sums - 1 - offset;
                cumulative += sum_probs[k] / norm_sum;
                if (cumulative >= target_prob)
                    return return_at(k);
            }
            return return_at(0);
        };

        constexpr double tail_probability = 0.05;
        double           rem_lower        = tail_probability;
        double           lower_tail_sum   = 0.0;

        for (std::size_t k = 0; k < num_sums && rem_lower > 1e-12; ++k)
        {
            const double p    = sum_probs[k] / norm_sum;
            const double used = std::min(p, rem_lower);
            lower_tail_sum += used * return_at(k);
            rem_lower -= used;
        }

        double rem_upper      = tail_probability;
        double upper_tail_sum = 0.0;

        for (std::size_t offset = 0; offset < num_sums && rem_upper > 1e-12; ++offset)
        {
            const std::size_t k    = num_sums - 1 - offset;
            const double      p    = sum_probs[k] / norm_sum;
            const double      used = std::min(p, rem_upper);
            upper_tail_sum += used * return_at(k);
            rem_upper -= used;
        }

        const double q025 = lower_quantile(0.025);
        const double q05  = lower_quantile(0.05);
        const double q95  = upper_quantile(0.05);
        const double q975 = upper_quantile(0.025);

        forecast.expected_return      = mean_log_return;
        forecast.expected_volatility  = std::sqrt(std::max(0.0, variance));
        forecast.prob_positive_return = prob_positive;
        forecast.distribution_entropy = terminal_entropy;

        forecast.expected_price    = entry_price * expected_price_multiplier;
        forecast.mode_price        = entry_price * std::exp(return_at(mode_k));
        forecast.price_ci_lower_95 = entry_price * std::exp(q025);
        forecast.price_ci_upper_95 = entry_price * std::exp(q975);

        forecast.long_var_95   = std::max(0.0, -q05);
        forecast.long_cvar_95  = std::max(0.0, -(lower_tail_sum / tail_probability));
        forecast.short_var_95  = std::max(0.0, q95);
        forecast.short_cvar_95 = std::max(0.0, upper_tail_sum / tail_probability);

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
