#pragma once

#include "quant/core/matrix.hpp"
#include "quant/core/types.hpp"

#include <cstddef>
#include <span>

enum class PositionSide
{
    Flat,
    Long,
    Short
};

struct QuantForecast
{
    std::size_t horizon_steps{1};
    double      entry_gap_return{0.0};

    double current_price{0.0};
    double expected_price{0.0};
    double mode_price{0.0};
    double price_ci_lower_95{0.0};
    double price_ci_upper_95{0.0};
    double expected_return{0.0};
    double expected_volatility{0.0};
    double prob_positive_return{0.0};

    // Legacy scalar fields (populated with long-side metrics for backward compatibility)
    double var_95{0.0};
    double cvar_95{0.0};

    // Side-specific unit tail loss magnitudes (positive values)
    double long_var_95{0.0};   // Downside 95% VaR (loss for long exposure)
    double long_cvar_95{0.0};  // Downside 95% CVaR (expected tail loss for long exposure)
    double short_var_95{0.0};  // Upside 95% VaR (loss for short exposure)
    double short_cvar_95{0.0}; // Upside 95% CVaR (expected tail loss for short exposure)

    double distribution_entropy{0.0};

    [[nodiscard]] constexpr double side_var_95(PositionSide side) const noexcept
    {
        switch (side)
        {
            case PositionSide::Long:
                return long_var_95;
            case PositionSide::Short:
                return short_var_95;
            case PositionSide::Flat:
            default:
                return 0.0;
        }
    }

    [[nodiscard]] constexpr double side_cvar_95(PositionSide side) const noexcept
    {
        switch (side)
        {
            case PositionSide::Long:
                return long_cvar_95;
            case PositionSide::Short:
                return short_cvar_95;
            case PositionSide::Flat:
            default:
                return 0.0;
        }
    }
};

template <std::size_t Bins>
class ITransitionKernel
{
public:
    virtual ~ITransitionKernel()                                                                                                    = default;
    virtual void build_kernel(double dx, double x_max, std::span<const DerivedFeatures> data, FlatMatrix<double, Bins>& out_matrix) = 0;
};
