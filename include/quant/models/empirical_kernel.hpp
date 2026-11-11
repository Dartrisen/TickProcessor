#pragma once

#include "quant/models/interfaces.hpp"

#include <algorithm>
#include <cmath>

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
                out_matrix(i, i) = 1.0; // Identity transition if state is unobserved
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
