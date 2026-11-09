#pragma once

#include "tick.hpp"

#include <span>
#include <utility>
#include <vector>

class MarketDataset
{
public:
    using Container = std::vector<Tick>;

    void add(Tick bar)
    {
        bars_.push_back(std::move(bar));
    }

    [[nodiscard]] std::size_t size() const noexcept
    {
        return bars_.size();
    }
    [[nodiscard]] bool empty() const noexcept
    {
        return bars_.empty();
    }

    [[nodiscard]] const Tick& operator[](std::size_t i) const
    {
        return bars_.at(i);
    }
    [[nodiscard]] std::span<const Tick> bars() const noexcept
    {
        return bars_;
    }

    void reserve(std::size_t capacity)
    {
        bars_.reserve(capacity);
    }

private:
    Container bars_;
};
