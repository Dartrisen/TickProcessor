#pragma once

#include "dataset.hpp"

#include <cmath>
#include <vector>

struct DerivedFeatures
{
    double log_return{0.0};
    double simple_return{0.0};
    double rolling_volatility{0.0};
    double volume_ratio{0.0};
};

class MarketFeatures
{
public:
    static std::vector<double> log_returns(const MarketDataset& data)
    {
        if (data.size() < 2)
            return {};

        std::vector<double> returns;
        returns.reserve(data.size() - 1);

        for (std::size_t i = 1; i < data.size(); ++i)
        {
            returns.push_back(std::log(data[i].close / data[i - 1].close));
        }
        return returns;
    }

    static std::vector<double> simple_returns(const MarketDataset& data)
    {
        if (data.size() < 2)
            return {};

        std::vector<double> returns;
        returns.reserve(data.size() - 1);

        for (std::size_t i = 1; i < data.size(); ++i)
        {
            returns.push_back((data[i].close / data[i - 1].close) - 1.0);
        }
        return returns;
    }
};
