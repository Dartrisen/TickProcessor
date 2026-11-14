#pragma once

#include "quant/models/interfaces.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>

namespace detail
{
[[nodiscard]] inline double transition_relevance_weight(const DerivedFeatures& prev, const DerivedFeatures& next, double current_volatility,
                                                        double regime_bandwidth = 0.50) noexcept
{
    // Volume conviction weight: transitions accompanied by higher relative volume
    // carry stronger order-flow signal than thin/quiet noise bars.
    const double v_prev    = (prev.volume_ratio > 0.0) ? prev.volume_ratio : 1.0;
    const double v_next    = (next.volume_ratio > 0.0) ? next.volume_ratio : 1.0;
    const double vol_score = std::clamp(std::sqrt(v_prev * v_next), 0.25, 4.0);

    // Volatility regime similarity weight: prioritize historical transitions whose
    // rolling volatility matches the current market volatility regime.
    double regime_score = 1.0;
    if (current_volatility > 1e-8 && prev.rolling_volatility > 1e-8 && regime_bandwidth > 1e-6)
    {
        const double log_vol_diff = std::log(prev.rolling_volatility / current_volatility);
        const double z            = log_vol_diff / regime_bandwidth;
        regime_score              = std::exp(-0.5 * z * z);
    }

    return vol_score * regime_score;
}
} // namespace detail

// Volume- and Volatility-Regime Weighted Empirical Transition Kernel
template <std::size_t Bins>
class EmpiricalKernel : public ITransitionKernel<Bins>
{
public:
    explicit EmpiricalKernel(double regime_bandwidth = 0.50) : regime_bandwidth_(regime_bandwidth) {}

    void build_kernel(double dx, double x_max, std::span<const DerivedFeatures> data, FlatMatrix<double, Bins>& out_matrix) override
    {
        out_matrix.fill(0.0);
        std::array<double, Bins> row_weights{};

        auto state = [x_max, dx](double x) noexcept -> std::size_t
        {
            double clamped = std::clamp(x, -x_max, x_max);
            return std::min(static_cast<std::size_t>((clamped + x_max) / dx), Bins - 1);
        };

        if (data.size() >= 2)
        {
            const double current_vol = data.back().rolling_volatility;

            for (std::size_t t = 0; t + 1 < data.size(); ++t)
            {
                const std::size_t i = state(data[t].log_return);
                const std::size_t j = state(data[t + 1].log_return);
                const double      w = detail::transition_relevance_weight(data[t], data[t + 1], current_vol, regime_bandwidth_);

                out_matrix(i, j) += w;
                row_weights[i] += w;
            }
        }

        for (std::size_t i = 0; i < Bins; ++i)
        {
            if (row_weights[i] <= 1e-12)
            {
                for (std::size_t j = 0; j < Bins; ++j)
                {
                    out_matrix(i, j) = 1.0 / static_cast<double>(Bins);
                }
                continue;
            }

            for (std::size_t j = 0; j < Bins; ++j)
            {
                out_matrix(i, j) /= row_weights[i];
            }
        }
    }

private:
    double regime_bandwidth_;
};

// Data-Calibrated Ornstein-Uhlenbeck / AR(1) Gaussian Diffusion Kernel
template <std::size_t Bins>
class GaussianDiffusionKernel : public ITransitionKernel<Bins>
{
public:
    explicit GaussianDiffusionKernel(double drift_offset = 0.0, double fallback_diffusion = 0.001) : drift_offset_(drift_offset), fallback_diffusion_(fallback_diffusion) {}

    void build_kernel(double dx, double x_max, std::span<const DerivedFeatures> data, FlatMatrix<double, Bins>& out_matrix) override
    {
        out_matrix.fill(0.0);

        double mean_ret  = 0.0;
        double rho_1     = 0.0;
        double diffusion = fallback_diffusion_;

        if (data.size() >= 2)
        {
            // Estimate sample mean and lag-1 autocorrelation of log returns
            for (const auto& f : data)
            {
                mean_ret += f.log_return;
            }
            mean_ret /= static_cast<double>(data.size());

            double var_sum = 0.0;
            double cov_sum = 0.0;
            for (std::size_t t = 0; t < data.size(); ++t)
            {
                const double d0 = data[t].log_return - mean_ret;
                var_sum += d0 * d0;
                if (t + 1 < data.size())
                {
                    const double d1 = data[t + 1].log_return - mean_ret;
                    cov_sum += d0 * d1;
                }
            }

            if (var_sum > 1e-16)
            {
                rho_1 = std::clamp(cov_sum / var_sum, -0.50, 0.50);
            }

            // Blend current rolling volatility with sample return stddev, scaled by current RVOL
            const double sample_std = std::sqrt(var_sum / static_cast<double>(data.size()));
            const double latest_vol = data.back().rolling_volatility;
            const double base_sigma = (latest_vol > 1e-8) ? (0.70 * latest_vol + 0.30 * sample_std) : ((sample_std > 1e-8) ? sample_std : fallback_diffusion_);

            const double rvol_scale = (data.back().volume_ratio > 0.0) ? std::sqrt(std::clamp(data.back().volume_ratio, 0.50, 2.00)) : 1.0;
            diffusion               = base_sigma * rvol_scale;
        }

        // Prevent kernel collapse onto a single grid bin when volatility is very low
        const double min_sigma = std::max(1e-6, 0.25 * dx);
        diffusion              = std::max(diffusion, min_sigma);
        const double inv_2sig2 = 1.0 / (2.0 * diffusion * diffusion);

        for (std::size_t i = 0; i < Bins; ++i)
        {
            const double x_i         = -x_max + (static_cast<double>(i) + 0.5) * dx;
            const double cond_mean_j = mean_ret + rho_1 * (x_i - mean_ret) + drift_offset_;
            double       row_sum     = 0.0;

            for (std::size_t j = 0; j < Bins; ++j)
            {
                const double x_j  = -x_max + (static_cast<double>(j) + 0.5) * dx;
                const double diff = x_j - cond_mean_j;
                const double prob = std::exp(-diff * diff * inv_2sig2);
                out_matrix(i, j)  = prob;
                row_sum += prob;
            }

            if (row_sum > 0.0)
            {
                for (std::size_t j = 0; j < Bins; ++j)
                {
                    out_matrix(i, j) /= row_sum;
                }
            }
            else
            {
                for (std::size_t j = 0; j < Bins; ++j)
                {
                    out_matrix(i, j) = 1.0 / static_cast<double>(Bins);
                }
            }
        }
    }

private:
    double drift_offset_;
    double fallback_diffusion_;
};

// Adaptive Bayesian Dirichlet Kernel with Volatility-Scaled Prior & Regime Counts
template <std::size_t Bins>
class AdaptiveDirichletKernel : public ITransitionKernel<Bins>
{
public:
    explicit AdaptiveDirichletKernel(double pseudo_weight = 10.0, double kappa = 0.15, double fallback_prior_vol = 0.001)
            : alpha_0_(pseudo_weight), kappa_(kappa), fallback_prior_vol_(fallback_prior_vol)
    {
    }

    void build_kernel(double dx, double x_max, std::span<const DerivedFeatures> data, FlatMatrix<double, Bins>& out_matrix) override
    {
        out_matrix.fill(0.0);
        std::array<std::array<double, Bins>, Bins> weighted_counts{};
        std::array<double, Bins>                   row_totals{};

        auto state = [x_max, dx](double x) noexcept -> std::size_t
        {
            double clamped = std::clamp(x, -x_max, x_max);
            return std::min(static_cast<std::size_t>((clamped + x_max) / dx), Bins - 1);
        };

        double active_prior_vol = fallback_prior_vol_;
        double effective_kappa  = kappa_;

        if (data.size() >= 2)
        {
            const double current_vol = data.back().rolling_volatility;
            if (current_vol > 1e-8)
            {
                active_prior_vol = current_vol;
            }

            // High relative volume weakens mean-reversion damping (stronger breakout persistence)
            if (data.back().volume_ratio > 0.0)
            {
                const double rvol = std::clamp(data.back().volume_ratio, 0.50, 2.00);
                effective_kappa   = std::clamp(kappa_ / std::sqrt(rvol), 0.01, 0.95);
            }

            double raw_weight_sum = 0.0;
            for (std::size_t t = 0; t + 1 < data.size(); ++t)
            {
                const std::size_t i = state(data[t].log_return);
                const std::size_t j = state(data[t + 1].log_return);
                const double      w = detail::transition_relevance_weight(data[t], data[t + 1], current_vol);

                weighted_counts[i][j] += w;
                row_totals[i] += w;
                raw_weight_sum += w;
            }

            // Normalize total effective sample count so alpha_0_ retains its intended prior strength
            const double num_transitions = static_cast<double>(data.size() - 1);
            if (raw_weight_sum > 1e-12)
            {
                const double norm_scale = num_transitions / raw_weight_sum;
                for (std::size_t i = 0; i < Bins; ++i)
                {
                    row_totals[i] *= norm_scale;
                    for (std::size_t j = 0; j < Bins; ++j)
                    {
                        weighted_counts[i][j] *= norm_scale;
                    }
                }
            }
        }

        const double clamped_vol = std::max(active_prior_vol, std::max(1e-6, 0.25 * dx));
        const double inv_2sig2   = 1.0 / (2.0 * clamped_vol * clamped_vol);

        for (std::size_t i = 0; i < Bins; ++i)
        {
            const double x_i         = -x_max + (static_cast<double>(i) + 0.5) * dx;
            const double target_mean = (1.0 - effective_kappa) * x_i;

            std::array<double, Bins> p_0{};
            double                   p0_sum = 0.0;

            for (std::size_t j = 0; j < Bins; ++j)
            {
                const double x_j  = -x_max + (static_cast<double>(j) + 0.5) * dx;
                const double diff = x_j - target_mean;
                p_0[j]            = std::exp(-diff * diff * inv_2sig2);
                p0_sum += p_0[j];
            }

            const double total_row_weight = row_totals[i] + alpha_0_;
            for (std::size_t j = 0; j < Bins; ++j)
            {
                const double prior_j = (p0_sum > 0.0) ? (p_0[j] / p0_sum) : (1.0 / static_cast<double>(Bins));
                out_matrix(i, j)     = (weighted_counts[i][j] + alpha_0_ * prior_j) / total_row_weight;
            }
        }
    }

private:
    double alpha_0_;
    double kappa_;
    double fallback_prior_vol_;
};

// Multi-Feature Phase-Space Kernel (Return + Acceleration + Vol-Normalized Z-Score)
template <std::size_t Bins>
class FactorizedPhaseSpaceKernel : public ITransitionKernel<Bins>
{
public:
    explicit FactorizedPhaseSpaceKernel(double momentum_coupling = 0.20, double vol_z_coupling = 0.20, double laplace = 1e-4)
            : gamma_(std::clamp(momentum_coupling, 0.0, 0.80)), beta_(std::clamp(vol_z_coupling, 0.0, 0.80 - std::clamp(momentum_coupling, 0.0, 0.80))), laplace_(laplace)
    {
    }

    void build_kernel(double dx, double x_max, std::span<const DerivedFeatures> data, FlatMatrix<double, Bins>& out_matrix) override
    {
        out_matrix.fill(0.0);
        std::array<std::array<double, Bins>, Bins> counts_x{};
        std::array<std::array<double, Bins>, Bins> counts_v{};
        std::array<std::array<double, Bins>, Bins> counts_z{};
        std::array<double, Bins>                   totals_x{};
        std::array<double, Bins>                   totals_v{};
        std::array<double, Bins>                   totals_z{};

        auto return_state = [x_max, dx](double x) noexcept -> std::size_t
        {
            const double clamped = std::clamp(x, -x_max, x_max);
            return std::min(static_cast<std::size_t>((clamped + x_max) / dx), Bins - 1);
        };

        // Volatility-standardized return z-score discretized onto [-3.0, +3.0] sigma
        static constexpr double ZMax    = 3.0;
        const double     dz      = (2.0 * ZMax) / static_cast<double>(Bins);
        auto             z_state = [dz](const DerivedFeatures& f) noexcept -> std::size_t
        {
            const double z       = (f.rolling_volatility > 1e-8) ? (f.log_return / f.rolling_volatility) : 0.0;
            const double clamped = std::clamp(z, -ZMax, ZMax);
            return std::min(static_cast<std::size_t>((clamped + ZMax) / dz), Bins - 1);
        };

        if (data.size() >= 3)
        {
            const double current_vol = data.back().rolling_volatility;

            for (std::size_t t = 1; t + 1 < data.size(); ++t)
            {
                const double x_t    = data[t].log_return;
                const double v_t    = data[t].log_return - data[t - 1].log_return;
                const double x_next = data[t + 1].log_return;

                const std::size_t i = return_state(x_t);
                const std::size_t v = return_state(v_t);
                const std::size_t z = z_state(data[t]);
                const std::size_t j = return_state(x_next);
                const double      w = detail::transition_relevance_weight(data[t], data[t + 1], current_vol);

                counts_x[i][j] += w;
                totals_x[i] += w;

                counts_v[v][j] += w;
                totals_v[v] += w;

                counts_z[z][j] += w;
                totals_z[z] += w;
            }
        }

        std::size_t current_v = Bins / 2;
        std::size_t current_z = Bins / 2;
        if (data.size() >= 2)
        {
            const double v_latest = data.back().log_return - data[data.size() - 2].log_return;
            current_v             = return_state(v_latest);
            current_z             = z_state(data.back());
        }

        const double alpha_x = 1.0 - gamma_ - beta_;
        const double denom_v = totals_v[current_v] + static_cast<double>(Bins) * laplace_;
        const double denom_z = totals_z[current_z] + static_cast<double>(Bins) * laplace_;

        for (std::size_t i = 0; i < Bins; ++i)
        {
            const double denom_x = totals_x[i] + static_cast<double>(Bins) * laplace_;

            for (std::size_t j = 0; j < Bins; ++j)
            {
                const double p_x = (counts_x[i][j] + laplace_) / denom_x;
                const double p_v = (counts_v[current_v][j] + laplace_) / denom_v;
                const double p_z = (counts_z[current_z][j] + laplace_) / denom_z;

                out_matrix(i, j) = alpha_x * p_x + gamma_ * p_v + beta_ * p_z;
            }
        }
    }

private:
    double gamma_;
    double beta_;
    double laplace_;
};
