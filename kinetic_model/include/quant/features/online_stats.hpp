#pragma once

#include <cmath>
#include <cstddef>

template <std::size_t WindowSize>
class RollingWindowStats
{
public:
    void push(double val) noexcept
    {
        if (count_ < WindowSize)
        {
            buffer_[count_++] = val;
            sum_ += val;
        }
        else
        {
            sum_ -= buffer_[head_];
            buffer_[head_] = val;
            sum_ += val;
            head_ = (head_ + 1) % WindowSize;
        }
    }

    [[nodiscard]] double mean() const noexcept
    {
        return count_ > 0 ? (sum_ / count_) : 0.0;
    }

    [[nodiscard]] double variance() const noexcept
    {
        if (count_ < 2)
            return 0.0;

        const double m           = mean();
        double       sq_diff_sum = 0.0;
        for (std::size_t i = 0; i < count_; ++i)
        {
            double diff = buffer_[i] - m;
            sq_diff_sum += diff * diff;
        }
        return sq_diff_sum / static_cast<double>(count_);
    }

    [[nodiscard]] double stddev() const noexcept
    {
        return std::sqrt(variance());
    }

private:
    double      buffer_[WindowSize]{};
    std::size_t head_{0};
    std::size_t count_{0};
    double      sum_{0.0};
};
