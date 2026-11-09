#pragma once

#include "market_features.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

// ============================================================
// Abstract Model Interface
// ============================================================
class IMarketModel
{
public:
    virtual ~IMarketModel()                                                       = default;
    virtual void                fit(const std::vector<DerivedFeatures>& features) = 0;
    virtual std::vector<double> step(const std::vector<double>& f) const          = 0;
};

// ============================================================
// Pluggable Kernel Strategy Interface for W(z' -> z)
// ============================================================
class ITransitionKernel
{
public:
    virtual ~ITransitionKernel()                                                                                                                    = default;
    virtual void build_kernel(std::size_t bins, double dx, double x_max, const std::vector<DerivedFeatures>& data, std::vector<double>& out_matrix) = 0;
};

// --- Kernel Implementation 1: Empirical Frequency Kernel ---
class EmpiricalKernel : public ITransitionKernel
{
public:
    void build_kernel(std::size_t bins, double dx, double x_max, const std::vector<DerivedFeatures>& data, std::vector<double>& out_matrix) override
    {
        out_matrix.assign(bins * bins, 0.0);
        std::vector<double> counts(bins, 0.0);

        auto state = [bins, x_max, dx](double x) -> std::size_t
        {
            double clamped = std::clamp(x, -x_max, x_max);
            return std::min(static_cast<std::size_t>((clamped + x_max) / dx), bins - 1);
        };

        for (std::size_t t = 0; t + 1 < data.size(); ++t)
        {
            std::size_t i = state(data[t].log_return);
            std::size_t j = state(data[t + 1].log_return);
            out_matrix[i * bins + j] += 1.0;
            counts[i] += 1.0;
        }

        // Normalize state transition probabilities W(i -> j)
        for (std::size_t i = 0; i < bins; ++i)
        {
            if (counts[i] > 0.0)
            {
                for (std::size_t j = 0; j < bins; ++j)
                {
                    out_matrix[i * bins + j] /= counts[i];
                }
            }
        }
    }
};

// --- Kernel Implementation 2: Gaussian Kinetic Drift-Diffusion Kernel ---
class GaussianDiffusionKernel : public ITransitionKernel
{
public:
    explicit GaussianDiffusionKernel(double drift = 0.0, double diffusion = 0.01) : drift_(drift), diffusion_(diffusion) {}

    void build_kernel(std::size_t bins, double dx, double x_max, const std::vector<DerivedFeatures>& /*data*/, std::vector<double>& out_matrix) override
    {
        out_matrix.assign(bins * bins, 0.0);

        for (std::size_t i = 0; i < bins; ++i)
        {
            double x_i     = -x_max + (i + 0.5) * dx;
            double row_sum = 0.0;

            for (std::size_t j = 0; j < bins; ++j)
            {
                double x_j               = -x_max + (j + 0.5) * dx;
                double diff              = (x_j - x_i - drift_);
                double prob              = std::exp(-0.5 * (diff * diff) / (diffusion_ * diffusion_));
                out_matrix[i * bins + j] = prob;
                row_sum += prob;
            }

            for (std::size_t j = 0; j < bins; ++j)
            {
                out_matrix[i * bins + j] /= row_sum;
            }
        }
    }

private:
    double drift_;
    double diffusion_;
};

// ============================================================
// Kinetic Model Engine
// ============================================================
class KineticModel : public IMarketModel
{
public:
    using Distribution = std::vector<double>;

    KineticModel(std::size_t bins, double x_max, std::unique_ptr<ITransitionKernel> kernel = std::make_unique<EmpiricalKernel>())
            : bins_(bins), x_max_(x_max), dx_(2.0 * x_max / bins), kernel_(std::move(kernel))
    {
        if (bins < 2 || x_max <= 0.0)
        {
            throw std::invalid_argument("Invalid kinetic model resolution parameters");
        }
        transition_.resize(bins_ * bins_, 0.0);
    }

    void set_kernel(std::unique_ptr<ITransitionKernel> kernel)
    {
        kernel_ = std::move(kernel);
        fitted_ = false;
    }

    void fit(const std::vector<DerivedFeatures>& features) override
    {
        if (features.size() < 2)
        {
            throw std::invalid_argument("Insufficient observations for model calibration");
        }
        kernel_->build_kernel(bins_, dx_, x_max_, features, transition_);
        fitted_ = true;
    }

    // Swappable overload accepting raw log returns
    void fit(const std::vector<double>& log_returns)
    {
        std::vector<DerivedFeatures> feats;
        feats.reserve(log_returns.size());
        for (double r : log_returns)
        {
            feats.push_back(DerivedFeatures{.log_return = r});
        }
        fit(feats);
    }

    [[nodiscard]] Distribution step(const Distribution& f) const override
    {
        return evolve(f);
    }

    [[nodiscard]] Distribution evolve(const Distribution& f) const
    {
        if (!fitted_)
            throw std::runtime_error("Kinetic model non-calibrated");
        if (f.size() != bins_)
            throw std::invalid_argument("Distribution dimension mismatch");

        Distribution next(bins_, 0.0);
        // Optimized 1D continuous array iteration for dynamic memory cache efficiency
        for (std::size_t i = 0; i < bins_; ++i)
        {
            const std::size_t row_offset = i * bins_;
            const double      f_i        = f[i];
            if (f_i == 0.0)
                continue;

            for (std::size_t j = 0; j < bins_; ++j)
            {
                next[j] += f_i * transition_[row_offset + j];
            }
        }
        return next;
    }

    [[nodiscard]] Distribution initial_distribution() const
    {
        return Distribution(bins_, 1.0 / static_cast<double>(bins_));
    }

    [[nodiscard]] std::size_t bins() const noexcept
    {
        return bins_;
    }

private:
    std::size_t                        bins_;
    double                             x_max_;
    double                             dx_;
    std::vector<double>                transition_; // Flat 1D Matrix [bins * bins]
    std::unique_ptr<ITransitionKernel> kernel_;
    bool                               fitted_{false};
};
