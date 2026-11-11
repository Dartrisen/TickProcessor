#pragma once

#include "quant/models/interfaces.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <span>

// -----------------------------------------------------------------------------
// 1. Existing Baseline Empirical Kernel
// -----------------------------------------------------------------------------
template <std::size_t Bins>
class EmpiricalKernel : public ITransitionKernel<Bins>
{
public:
    void build_kernel(double dx, double x_max, std::span<const DerivedFeatures> data, FlatMatrix<double, Bins>& out_matrix) override
    {
        out_matrix.fill(0.0);
        double counts[Bins]{0.0};

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
                counts[i] += 1.0;
            }
        }

        for (std::size_t i = 0; i < Bins; ++i)
        {
            if (counts[i] > 0.0)
            {
                for (std::size_t j = 0; j < Bins; ++j)
                {
                    out_matrix(i, j) /= counts[i];
                }
            }
            else
            {
                out_matrix(i, i) = 1.0;
            }
        }
    }
};

// -----------------------------------------------------------------------------
// 2. Multi-Horizon Propagator Kernel (Matrix Exponentiation: W^k)
// Mathematically damps 1-minute noise by evolving forward k steps ahead.
// -----------------------------------------------------------------------------
template <std::size_t Bins, std::size_t Horizon = 5>
class MultiStepPropagatorKernel : public ITransitionKernel<Bins>
{
public:
    void build_kernel(double dx, double x_max, std::span<const DerivedFeatures> data, FlatMatrix<double, Bins>& out_matrix) override
    {
        // 1. Build base 1-step transition matrix W_1
        FlatMatrix<double, Bins> W_1{};
        EmpiricalKernel<Bins>    base_builder;
        base_builder.build_kernel(dx, x_max, data, W_1);

        // 2. Compute multi-step transition operator W_k = W_1 ^ Horizon
        FlatMatrix<double, Bins> W_curr = W_1;
        FlatMatrix<double, Bins> W_next{};

        for (std::size_t step = 1; step < Horizon; ++step)
        {
            W_next.fill(0.0);
            for (std::size_t i = 0; i < Bins; ++i)
            {
                for (std::size_t k = 0; k < Bins; ++k)
                {
                    const double w_ik = W_curr(i, k);
                    if (w_ik <= 1e-12)
                        continue;

                    for (std::size_t j = 0; j < Bins; ++j)
                    {
                        W_next(i, j) += w_ik * W_1(k, j);
                    }
                }
            }
            W_curr = W_next;
        }

        out_matrix = W_curr;
    }
};

// -----------------------------------------------------------------------------
// 3. Inertial Anti-Churn Kernel (Diagonal Stabilization)
// Adds a state-retention prior lambda * I to stop micro-second signal flipping.
// -----------------------------------------------------------------------------
template <std::size_t Bins>
class InertialAntiChurnKernel : public ITransitionKernel<Bins>
{
public:
    explicit InertialAntiChurnKernel(double inertia_lambda = 0.40) : inertia_lambda_(inertia_lambda) {}

    void build_kernel(double dx, double x_max, std::span<const DerivedFeatures> data, FlatMatrix<double, Bins>& out_matrix) override
    {
        EmpiricalKernel<Bins> base_builder;
        base_builder.build_kernel(dx, x_max, data, out_matrix);

        // Blend matrix with Identity matrix to enforce state persistence
        for (std::size_t i = 0; i < Bins; ++i)
        {
            for (std::size_t j = 0; j < Bins; ++j)
            {
                double target    = (i == j) ? 1.0 : 0.0;
                out_matrix(i, j) = (1.0 - inertia_lambda_) * out_matrix(i, j) + inertia_lambda_ * target;
            }
        }
    }

private:
    double inertia_lambda_;
};

// -----------------------------------------------------------------------------
// 4. Trend-Filtered Kinetic Kernel
// Evaluates transition probabilities conditional on rolling volatility & drift.
// -----------------------------------------------------------------------------
template <std::size_t Bins>
class TrendFilteredKineticKernel : public ITransitionKernel<Bins>
{
public:
    explicit TrendFilteredKineticKernel(double damping = 0.25) : damping_(damping) {}

    void build_kernel(double dx, double x_max, std::span<const DerivedFeatures> data, FlatMatrix<double, Bins>& out_matrix) override
    {
        out_matrix.fill(0.0);
        double counts[Bins]{0.0};

        auto state = [x_max, dx](double x) -> std::size_t
        {
            double clamped = std::clamp(x, -x_max, x_max);
            return std::min(static_cast<std::size_t>((clamped + x_max) / dx), Bins - 1);
        };

        if (data.size() >= 2)
        {
            for (std::size_t t = 0; t + 1 < data.size(); ++t)
            {
                // Filter out micro-noise: only register transitions if move exceeds rolling vol
                if (std::abs(data[t].log_return) < damping_ * data[t].rolling_volatility)
                    continue;

                std::size_t i = state(data[t].log_return);
                std::size_t j = state(data[t + 1].log_return);
                out_matrix(i, j) += 1.0;
                counts[i] += 1.0;
            }
        }

        for (std::size_t i = 0; i < Bins; ++i)
        {
            if (counts[i] > 0.0)
            {
                for (std::size_t j = 0; j < Bins; ++j)
                {
                    out_matrix(i, j) /= counts[i];
                }
            }
            else
            {
                out_matrix(i, i) = 1.0;
            }
        }
    }

private:
    double damping_;
};
