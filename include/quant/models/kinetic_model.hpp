#pragma once

#include "quant/core/matrix.hpp"
#include "quant/models/interfaces.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <utility>

template <std::size_t Bins>
class HighPerformanceKineticEngine
{
public:
    using StateVector = std::array<double, Bins>;
    using Matrix      = FlatMatrix<double, Bins>;

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

    // Evaluates cumulative return & dual-tail risk over `horizon_steps` starting from `entry_price`,
    // conditioned on `realized_entry_gap = ln(entry_price / signal_close_price)` already elapsed in step 1.
    [[nodiscard]] QuantForecast analyze_and_predict_horizon(double entry_price, const StateVector& step1_dist, std::size_t horizon_steps = 1,
                                                            double realized_entry_gap = 0.0) const noexcept
    {
        QuantForecast forecast;
        forecast.horizon_steps    = std::max<std::size_t>(1, horizon_steps);
        forecast.entry_gap_return = realized_entry_gap;
        forecast.current_price    = entry_price;

        double total_probability = 0.0;
        double entropy           = 0.0;
        for (std::size_t i = 0; i < Bins; ++i)
        {
            const double p = std::max(0.0, step1_dist[i]);
            total_probability += p;
        }
        if (total_probability <= 1e-12)
            return forecast;

        StateVector norm_step1{};
        for (std::size_t i = 0; i < Bins; ++i)
        {
            norm_step1[i] = std::max(0.0, step1_dist[i]) / total_probability;
            if (norm_step1[i] > 1e-12)
                entropy -= norm_step1[i] * std::log2(norm_step1[i]);
        }

        const auto* W = active_kernel_.load(std::memory_order_acquire);
        if (forecast.horizon_steps >= 2 && W != nullptr)
        {
            // Exact 2-step Markov path distribution:
            // Step 1 residual return (O_{i+1} -> C_{i+1}) + Step 2 return (C_{i+1} -> C_{i+2})
            struct ReturnAtom
            {
                double ret{0.0};
                double prob{0.0};
            };

            std::array<ReturnAtom, Bins * Bins> atoms{};
            std::size_t                         atom_count = 0;

            // If horizon_steps > 2, propagate intermediate state to step (horizon_steps - 1)
            StateVector prev_state = norm_step1;
            double      mid_drift  = 0.0;
            for (std::size_t s = 2; s < forecast.horizon_steps; ++s)
            {
                for (std::size_t i = 0; i < Bins; ++i)
                {
                    mid_drift += prev_state[i] * bin_center(i);
                }
                StateVector next_state{};
                evolve(prev_state, next_state);
                prev_state = next_state;
            }

            // Compute conditional marginal of step 1 given step (horizon - 1) for exact 2-step joint paths
            for (std::size_t i = 0; i < Bins; ++i)
            {
                const double p_i = norm_step1[i];
                if (p_i <= 1e-12)
                    continue;

                const double r_i = (bin_center(i) - realized_entry_gap) + mid_drift;
                for (std::size_t j = 0; j < Bins; ++j)
                {
                    const double p_ij = p_i * (*W)(i, j);
                    if (p_ij <= 1e-15)
                        continue;

                    atoms[atom_count++] = ReturnAtom{.ret = r_i + bin_center(j), .prob = p_ij};
                }
            }

            if (atom_count > 0)
            {
                std::sort(atoms.begin(), atoms.begin() + static_cast<std::ptrdiff_t>(atom_count), [](const ReturnAtom& a, const ReturnAtom& b) noexcept { return a.ret < b.ret; });

                double norm_sum                  = 0.0;
                double mean_log_return           = 0.0;
                double expected_price_multiplier = 0.0;
                double prob_positive             = 0.0;
                double mode_ret                  = atoms[0].ret;
                double mode_prob                 = -1.0;

                for (std::size_t k = 0; k < atom_count; ++k)
                {
                    norm_sum += atoms[k].prob;
                }

                for (std::size_t k = 0; k < atom_count; ++k)
                {
                    const double r = atoms[k].ret;
                    const double p = atoms[k].prob / norm_sum;

                    mean_log_return += p * r;
                    expected_price_multiplier += p * std::exp(r);
                    if (r > 0.0)
                        prob_positive += p;
                    if (p > mode_prob)
                    {
                        mode_prob = p;
                        mode_ret  = r;
                    }
                }

                double variance = 0.0;
                for (std::size_t k = 0; k < atom_count; ++k)
                {
                    const double p    = atoms[k].prob / norm_sum;
                    const double diff = atoms[k].ret - mean_log_return;
                    variance += p * diff * diff;
                }

                auto lower_quantile = [&](double target_prob) noexcept
                {
                    double cum = 0.0;
                    for (std::size_t k = 0; k < atom_count; ++k)
                    {
                        cum += atoms[k].prob / norm_sum;
                        if (cum >= target_prob)
                            return atoms[k].ret;
                    }
                    return atoms[atom_count - 1].ret;
                };

                auto upper_quantile = [&](double target_prob) noexcept
                {
                    double cum = 0.0;
                    for (std::size_t k = 0; k < atom_count; ++k)
                    {
                        const std::size_t idx = atom_count - 1 - k;
                        cum += atoms[idx].prob / norm_sum;
                        if (cum >= target_prob)
                            return atoms[idx].ret;
                    }
                    return atoms[0].ret;
                };

                constexpr double tail_prob = 0.05;
                double           rem_lower = tail_prob;
                double           lower_sum = 0.0;
                for (std::size_t k = 0; k < atom_count && rem_lower > 1e-12; ++k)
                {
                    const double p    = atoms[k].prob / norm_sum;
                    const double used = std::min(p, rem_lower);
                    lower_sum += used * atoms[k].ret;
                    rem_lower -= used;
                }

                double rem_upper = tail_prob;
                double upper_sum = 0.0;
                for (std::size_t k = 0; k < atom_count && rem_upper > 1e-12; ++k)
                {
                    const std::size_t idx  = atom_count - 1 - k;
                    const double      p    = atoms[idx].prob / norm_sum;
                    const double      used = std::min(p, rem_upper);
                    upper_sum += used * atoms[idx].ret;
                    rem_upper -= used;
                }

                const double q025 = lower_quantile(0.025);
                const double q05  = lower_quantile(0.05);
                const double q95  = upper_quantile(0.05);
                const double q975 = upper_quantile(0.025);

                forecast.expected_return      = mean_log_return;
                forecast.expected_volatility  = std::sqrt(std::max(0.0, variance));
                forecast.prob_positive_return = prob_positive;
                forecast.distribution_entropy = entropy;

                forecast.expected_price    = entry_price * expected_price_multiplier;
                forecast.mode_price        = entry_price * std::exp(mode_ret);
                forecast.price_ci_lower_95 = entry_price * std::exp(q025);
                forecast.price_ci_upper_95 = entry_price * std::exp(q975);

                forecast.long_var_95   = std::max(0.0, -q05);
                forecast.long_cvar_95  = std::max(0.0, -(lower_sum / tail_prob));
                forecast.short_var_95  = std::max(0.0, q95);
                forecast.short_cvar_95 = std::max(0.0, upper_sum / tail_prob);

                forecast.var_95  = forecast.long_var_95;
                forecast.cvar_95 = forecast.long_cvar_95;
                return forecast;
            }
        }

        // Single-step evaluation (shifted by realized_entry_gap = ln(O_{i+1} / C_i))
        double      mean_log_return           = 0.0;
        double      expected_price_multiplier = 0.0;
        double      prob_positive             = 0.0;
        std::size_t mode_index                = 0;
        double      mode_prob                 = -1.0;

        for (std::size_t i = 0; i < Bins; ++i)
        {
            const double r = bin_center(i) - realized_entry_gap;
            const double p = norm_step1[i];

            mean_log_return += p * r;
            expected_price_multiplier += p * std::exp(r);

            if (r > 0.0)
                prob_positive += p;

            if (p > mode_prob)
            {
                mode_prob  = p;
                mode_index = i;
            }
        }

        double variance = 0.0;
        for (std::size_t i = 0; i < Bins; ++i)
        {
            const double p    = norm_step1[i];
            const double diff = (bin_center(i) - realized_entry_gap) - mean_log_return;
            variance += p * diff * diff;
        }

        auto lower_return_quantile = [&](double tail_prob) noexcept
        {
            double cumulative = 0.0;
            for (std::size_t i = 0; i < Bins; ++i)
            {
                cumulative += norm_step1[i];
                if (cumulative >= tail_prob)
                    return bin_center(i) - realized_entry_gap;
            }
            return bin_center(Bins - 1) - realized_entry_gap;
        };

        auto upper_return_quantile = [&](double tail_prob) noexcept
        {
            double cumulative = 0.0;
            for (std::size_t k = 0; k < Bins; ++k)
            {
                const std::size_t idx = Bins - 1 - k;
                cumulative += norm_step1[idx];
                if (cumulative >= tail_prob)
                    return bin_center(idx) - realized_entry_gap;
            }
            return bin_center(0) - realized_entry_gap;
        };

        constexpr double tail_probability      = 0.05;
        double           lower_remaining_prob  = tail_probability;
        double           lower_tail_return_sum = 0.0;

        for (std::size_t i = 0; i < Bins && lower_remaining_prob > 1e-12; ++i)
        {
            const double p         = norm_step1[i];
            const double used_prob = std::min(p, lower_remaining_prob);

            lower_tail_return_sum += used_prob * (bin_center(i) - realized_entry_gap);
            lower_remaining_prob -= used_prob;
        }

        double upper_remaining_prob  = tail_probability;
        double upper_tail_return_sum = 0.0;

        for (std::size_t k = 0; k < Bins && upper_remaining_prob > 1e-12; ++k)
        {
            const std::size_t idx       = Bins - 1 - k;
            const double      p         = norm_step1[idx];
            const double      used_prob = std::min(p, upper_remaining_prob);

            upper_tail_return_sum += used_prob * (bin_center(idx) - realized_entry_gap);
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

        forecast.expected_price = entry_price * expected_price_multiplier;
        forecast.mode_price     = entry_price * std::exp(bin_center(mode_index) - realized_entry_gap);

        forecast.price_ci_lower_95 = entry_price * std::exp(q025);
        forecast.price_ci_upper_95 = entry_price * std::exp(q975);

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
