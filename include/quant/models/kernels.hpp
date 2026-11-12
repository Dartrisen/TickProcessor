#pragma once

#include "quant/models/interfaces.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>

template <std::size_t Bins>
class EmpiricalKernel : public ITransitionKernel<Bins>
{
public:
    void build_kernel(double dx, double x_max, std::span<const DerivedFeatures> data, FlatMatrix<double, Bins>& out_matrix) override
    {
        out_matrix.fill(0.0);
        double row_counts[Bins]{0.0};

        auto state = [x_max, dx](double x) -> std::size_t
        {
            double clamped = std::clamp(x, -x_max, x_max);
            return std::min(static_cast<std::size_t>((clamped + x_max) / dx), Bins - 1);
        };

        if (data.size() >= 2)
        {
            for (std::size_t t = 0; t + 1 < data.size(); ++t)
            {
                std::size_t i = state(data[t].log_return);
                std::size_t j = state(data[t + 1].log_return);
                out_matrix(i, j) += 1.0;
                row_counts[i] += 1.0;
            }
        }

        for (std::size_t i = 0; i < Bins; ++i)
        {
            if (row_counts[i] <= 0.0)
            {
                // No observations for this state: use an uninformative,
                // valid uniform distribution instead of producing 0/0 NaNs.
                for (std::size_t j = 0; j < Bins; ++j)
                {
                    out_matrix(i, j) = 1.0 / static_cast<double>(Bins);
                }
                continue;
            }

            for (std::size_t j = 0; j < Bins; ++j)
            {
                out_matrix(i, j) /= row_counts[i];
            }
        }
    }
};

template <std::size_t Bins>
class GaussianDiffusionKernel : public ITransitionKernel<Bins>
{
public:
    explicit GaussianDiffusionKernel(double drift = 0.0, double diffusion = 0.005) : drift_(drift), diffusion_(diffusion) {}

    void build_kernel(double dx, double x_max, std::span<const DerivedFeatures> /*data*/, FlatMatrix<double, Bins>& out_matrix) override
    {
        out_matrix.fill(0.0);

        for (std::size_t i = 0; i < Bins; ++i)
        {
            double x_i     = -x_max + (static_cast<double>(i) + 0.5) * dx;
            double row_sum = 0.0;

            for (std::size_t j = 0; j < Bins; ++j)
            {
                double x_j       = -x_max + (static_cast<double>(j) + 0.5) * dx;
                double diff      = (x_j - x_i - drift_);
                double prob      = std::exp(-0.5 * (diff * diff) / (diffusion_ * diffusion_));
                out_matrix(i, j) = prob;
                row_sum += prob;
            }

            if (row_sum > 0.0)
            {
                for (std::size_t j = 0; j < Bins; ++j)
                {
                    out_matrix(i, j) /= row_sum;
                }
            }
        }
    }

private:
    double drift_;
    double diffusion_;
};

template <std::size_t Bins>
class AdaptiveDirichletKernel : public ITransitionKernel<Bins>
{
public:
    explicit AdaptiveDirichletKernel(double pseudo_weight = 10.0, double kappa = 0.15, double prior_vol = 0.001) : alpha_0_(pseudo_weight), kappa_(kappa), prior_vol_(prior_vol) {}

    void build_kernel(double dx, double x_max, std::span<const DerivedFeatures> data, FlatMatrix<double, Bins>& out_matrix) override
    {
        out_matrix.fill(0.0);
        std::array<std::array<double, Bins>, Bins> counts{};
        std::array<double, Bins>                   row_totals{};

        auto state = [x_max, dx](double x) -> std::size_t
        {
            double clamped = std::clamp(x, -x_max, x_max);
            return std::min(static_cast<std::size_t>((clamped + x_max) / dx), Bins - 1);
        };

        if (data.size() >= 2)
        {
            for (std::size_t t = 0; t + 1 < data.size(); ++t)
            {
                std::size_t i = state(data[t].log_return);
                std::size_t j = state(data[t + 1].log_return);
                counts[i][j] += 1.0;
                row_totals[i] += 1.0;
            }
        }

        const double inv_2sig2 = 1.0 / (2.0 * prior_vol_ * prior_vol_);

        for (std::size_t i = 0; i < Bins; ++i)
        {
            double x_i         = -x_max + (static_cast<double>(i) + 0.5) * dx;
            double target_mean = (1.0 - kappa_) * x_i;

            std::array<double, Bins> p_0{};
            double                   p0_sum = 0.0;

            for (std::size_t j = 0; j < Bins; ++j)
            {
                double x_j  = -x_max + (static_cast<double>(j) + 0.5) * dx;
                double diff = x_j - target_mean;
                p_0[j]      = std::exp(-diff * diff * inv_2sig2);
                p0_sum += p_0[j];
            }

            double total_row_weight = row_totals[i] + alpha_0_;
            for (std::size_t j = 0; j < Bins; ++j)
            {
                double prior_j   = p_0[j] / p0_sum;
                out_matrix(i, j) = (counts[i][j] + alpha_0_ * prior_j) / total_row_weight;
            }
        }
    }

private:
    double alpha_0_;
    double kappa_;
    double prior_vol_;
};

template <std::size_t Bins>
class FactorizedPhaseSpaceKernel : public ITransitionKernel<Bins>
{
public:
    explicit FactorizedPhaseSpaceKernel(double momentum_coupling = 0.25, double laplace = 1e-4) : gamma_(momentum_coupling), laplace_(laplace) {}

    void build_kernel(double dx, double x_max, std::span<const DerivedFeatures> data, FlatMatrix<double, Bins>& out_matrix) override
    {
        out_matrix.fill(0.0);
        std::array<std::array<double, Bins>, Bins> counts_x{};
        std::array<std::array<double, Bins>, Bins> counts_v{};
        std::array<double, Bins>                   totals_x{};
        std::array<double, Bins>                   totals_v{};

        auto state = [x_max, dx](double x) -> std::size_t
        {
            double clamped = std::clamp(x, -x_max, x_max);
            return std::min(static_cast<std::size_t>((clamped + x_max) / dx), Bins - 1);
        };

        if (data.size() >= 3)
        {
            for (std::size_t t = 1; t + 1 < data.size(); ++t)
            {
                double x_t    = data[t].log_return;
                double v_t    = data[t].log_return - data[t - 1].log_return;
                double x_next = data[t + 1].log_return;

                std::size_t i = state(x_t);
                std::size_t v = state(v_t);
                std::size_t j = state(x_next);

                counts_x[i][j] += 1.0;
                totals_x[i] += 1.0;

                counts_v[v][j] += 1.0;
                totals_v[v] += 1.0;
            }
        }

        // Latest feature frame determines current acceleration state v
        std::size_t current_v = Bins / 2;
        if (data.size() >= 2)
        {
            double v_latest = data.back().log_return - data[data.size() - 2].log_return;
            current_v       = state(v_latest);
        }

        for (std::size_t i = 0; i < Bins; ++i)
        {
            double denom_x = totals_x[i] + Bins * laplace_;
            double denom_v = totals_v[current_v] + Bins * laplace_;

            for (std::size_t j = 0; j < Bins; ++j)
            {
                double p_x = (counts_x[i][j] + laplace_) / denom_x;
                double p_v = (counts_v[current_v][j] + laplace_) / denom_v;

                out_matrix(i, j) = (1.0 - gamma_) * p_x + gamma_ * p_v;
            }
        }
    }

private:
    double gamma_;
    double laplace_;
};
