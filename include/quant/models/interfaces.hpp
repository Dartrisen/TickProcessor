#pragma once

#include "quant/core/matrix.hpp"
#include "quant/core/types.hpp"

#include <span>

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

template <std::size_t Bins>
class ITransitionKernel
{
public:
    virtual ~ITransitionKernel()                                                                                                    = default;
    virtual void build_kernel(double dx, double x_max, std::span<const DerivedFeatures> data, FlatMatrix<double, Bins>& out_matrix) = 0;
};
