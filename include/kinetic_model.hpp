#pragma once

#include "market_features.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

struct QuantForecast
{
    double current_price{0.0};
    double expected_price{0.0};
    double mode_price{0.0};
    double price_ci_lower_95{0.0};
    double price_ci_upper_95{0.0};
    double expected_return{0.0};
    double expected_volatility{0.0};
    double prob_positive_return{0.0};
    double var_95{0.0};
    double cvar_95{0.0};
    double distribution_entropy{0.0};
};

class IMarketModel
{
public:
    virtual ~IMarketModel()                                                       = default;
    virtual void                fit(const std::vector<DerivedFeatures>& features) = 0;
    virtual std::vector<double> step(const std::vector<double>& f) const          = 0;
};

class ITransitionKernel
{
public:
    virtual ~ITransitionKernel()                                                                                                                    = default;
    virtual void build_kernel(std::size_t bins, double dx, double x_max, const std::vector<DerivedFeatures>& data, std::vector<double>& out_matrix) = 0;
};

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

        for (std::size_t i = 0; i < bins; ++i)
        {
            if (counts[i] > 0.0)
            {
                for (std::size_t j = 0; j < bins; ++j)
                {
                    out_matrix[i * bins + j] /= counts[i];
                }
            }
            else
            {
                out_matrix[i * bins + i] = 1.0; // Diffuse to self if unobserved
            }
        }
    }
};

class GaussianDiffusionKernel : public ITransitionKernel
{
public:
    explicit GaussianDiffusionKernel(double drift = 0.0, double diffusion = 0.005) : drift_(drift), diffusion_(diffusion) {}

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

    // --- Modern Quantitative Analysis Tools ---

    [[nodiscard]] double bin_center(std::size_t index) const noexcept
    {
        return -x_max_ + (static_cast<double>(index) + 0.5) * dx_;
    }

    [[nodiscard]] std::size_t state_index(double log_return) const noexcept
    {
        double clamped = std::clamp(log_return, -x_max_, x_max_);
        return std::min(static_cast<std::size_t>((clamped + x_max_) / dx_), bins_ - 1);
    }

    [[nodiscard]] Distribution state_delta(std::size_t state_idx) const
    {
        Distribution f(bins_, 0.0);
        if (state_idx < bins_)
            f[state_idx] = 1.0;
        return f;
    }

    [[nodiscard]] Distribution stationary_distribution(std::size_t max_iters = 1000, double tol = 1e-9) const
    {
        Distribution dist = initial_distribution();
        for (std::size_t iter = 0; iter < max_iters; ++iter)
        {
            Distribution next = evolve(dist);
            double       diff = 0.0;
            for (std::size_t i = 0; i < bins_; ++i)
            {
                diff += std::abs(next[i] - dist[i]);
            }
            dist = std::move(next);
            if (diff < tol)
                break;
        }
        return dist;
    }

    [[nodiscard]] QuantForecast analyze_and_predict(double current_price, const Distribution& dist) const
    {
        QuantForecast forecast;
        forecast.current_price = current_price;

        double mean_r   = 0.0;
        double var_r    = 0.0;
        double prob_pos = 0.0;
        double entropy  = 0.0;

        std::size_t max_idx  = 0;
        double      max_prob = -1.0;

        for (std::size_t i = 0; i < bins_; ++i)
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

        for (std::size_t i = 0; i < bins_; ++i)
        {
            double diff = bin_center(i) - mean_r;
            var_r += dist[i] * diff * diff;
        }

        forecast.expected_return      = mean_r;
        forecast.expected_volatility  = std::sqrt(var_r);
        forecast.prob_positive_return = prob_pos;
        forecast.distribution_entropy = entropy;

        // Price expectations
        forecast.expected_price = current_price * std::exp(mean_r);
        forecast.mode_price     = current_price * std::exp(bin_center(max_idx));

        // Quantile Risk Metrics (VaR, CVaR, & Confidence Intervals)
        double cum_prob   = 0.0;
        double var_95_val = bin_center(0);
        double cvar_sum   = 0.0;
        double cvar_prob  = 0.0;

        for (std::size_t i = 0; i < bins_; ++i)
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
            forecast.price_ci_upper_95 = current_price * std::exp(bin_center(bins_ - 1));
        }

        forecast.cvar_95 = (cvar_prob > 0.0) ? (cvar_sum / cvar_prob) : var_95_val;

        return forecast;
    }

    [[nodiscard]] Distribution initial_distribution() const
    {
        return Distribution(bins_, 1.0 / static_cast<double>(bins_));
    }

    [[nodiscard]] std::size_t bins() const noexcept
    {
        return bins_;
    }

    [[nodiscard]] double x_max() const noexcept
    {
        return x_max_;
    }

    [[nodiscard]] const std::vector<double>& transition_matrix() const noexcept
    {
        return transition_;
    }

private:
    std::size_t                        bins_;
    double                             x_max_;
    double                             dx_;
    std::vector<double>                transition_;
    std::unique_ptr<ITransitionKernel> kernel_;
    bool                               fitted_{false};
};
