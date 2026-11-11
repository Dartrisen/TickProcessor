#pragma once
#include <array>
#include <cstddef>
#include <span>

template <typename T, std::size_t Bins>
class FlatMatrix
{
public:
    static constexpr std::size_t TotalSize = Bins * Bins;

    constexpr T& operator()(std::size_t i, std::size_t j) noexcept
    {
        return data_[i * Bins + j];
    }
    constexpr const T& operator()(std::size_t i, std::size_t j) const noexcept
    {
        return data_[i * Bins + j];
    }

    [[nodiscard]] constexpr std::span<const T> row(std::size_t i) const noexcept
    {
        return std::span<const T>(&data_[i * Bins], Bins);
    }

    void fill(T val) noexcept
    {
        data_.fill(val);
    }

private:
    alignas(64) std::array<T, TotalSize> data_{}; // 64-byte aligned for AVX2/AVX-512 SIMD
};
