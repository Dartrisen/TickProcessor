#pragma once

#include "quant/models/interfaces.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <span>

// -----------------------------------------------------------------------------
// 1. Baseline Empirical Kernel (Maximum Likelihood with Laplace Regularization)
// -----------------------------------------------------------------------------
template <std::size_t Bins>
class EmpiricalKernel : public ITransitionKernel<Bins>
{
public:
    explicit EmpiricalKernel(double laplace_prior = 1e-4) : laplace_prior_(laplace_prior) {}

    void build_kernel(double dx, double x_max, std::span<const DerivedFeatures> data, FlatMatrix<double, Bins>& out_matrix) override
    {
        out_matrix.fill(0.0);

        // Initialize counts with a subtle Laplace prior to prevent zero-probability locking
        std::array<double, Bins> row_counts{};
        for (std::size_t i = 0; i < Bins; ++i)
        {
            for (std::size_t j = 0; j < Bins; ++j)
            {
                out_matrix(i, j) = laplace_prior_;
                row_counts[i] += laplace_prior_;
            }
        }

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

        // Normalize rows to form valid probability distributions
        for (std::size_t i = 0; i < Bins; ++i)
        {
            for (std::size_t j = 0; j < Bins; ++j)
            {
                out_matrix(i, j) /= row_counts[i];
            }
        }
    }

private:
    double laplace_prior_;
};

// -----------------------------------------------------------------------------
// 2. Spatial-Smoothed Empirical Kernel (1D Gaussian Spatial KDE Blur)
// Smoothes target return bins (columns) without blurring time-step transitions.
// -----------------------------------------------------------------------------
template <std::size_t Bins>
class SpatialSmoothedKernel : public ITransitionKernel<Bins>
{
public:
    explicit SpatialSmoothedKernel(double spatial_bandwidth_bins = 0.60) : bandwidth_bins_(spatial_bandwidth_bins) {}

    void build_kernel(double dx, double x_max, std::span<const DerivedFeatures> data, FlatMatrix<double, Bins>& out_matrix) override
    {
        // 1. Build raw empirical transition counts
        EmpiricalKernel<Bins>    raw_builder(0.0);
        FlatMatrix<double, Bins> raw_W{};
        raw_builder.build_kernel(dx, x_max, data, raw_W);

        // 2. Apply 1D spatial Gaussian kernel smoothing across target bins (columns j)
        double inv_2h2 = 1.0 / (2.0 * bandwidth_bins_ * bandwidth_bins_);

        for (std::size_t i = 0; i < Bins; ++i)
        {
            double                   row_sum = 0.0;
            std::array<double, Bins> smoothed_row{};

            for (std::size_t j = 0; j < Bins; ++j)
            {
                double sum_prob   = 0.0;
                double sum_weight = 0.0;

                for (std::size_t k = 0; k < Bins; ++k)
                {
                    double diff = static_cast<double>(j) - static_cast<double>(k);
                    double w    = std::exp(-diff * diff * inv_2h2);
                    sum_prob += raw_W(i, k) * w;
                    sum_weight += w;
                }

                smoothed_row[j] = sum_prob / sum_weight;
                row_sum += smoothed_row[j];
            }

            // Re-normalize row sum to 1.0
            for (std::size_t j = 0; j < Bins; ++j)
            {
                out_matrix(i, j) = (row_sum > 0.0) ? (smoothed_row[j] / row_sum) : (i == j ? 1.0 : 0.0);
            }
        }
    }

private:
    double bandwidth_bins_;
};
