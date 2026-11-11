#pragma once

#include <array>
#include <atomic>
#include <cstddef>

template <typename T, std::size_t Capacity>
class SPSCRingBuffer
{
public:
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");

    SPSCRingBuffer() : head_(0), tail_(0) {}

    bool push(const T& item) noexcept
    {
        const auto current_tail = tail_.load(std::memory_order_relaxed);
        const auto current_head = head_.load(std::memory_order_acquire);

        if ((current_tail - current_head) >= Capacity)
        {
            return false; // Queue full
        }

        buffer_[current_tail & Mask] = item;
        tail_.store(current_tail + 1, std::memory_order_release);
        return true;
    }

    bool pop(T& item) noexcept
    {
        const auto current_head = head_.load(std::memory_order_relaxed);
        const auto current_tail = tail_.load(std::memory_order_acquire);

        if (current_head == current_tail)
        {
            return false; // Queue empty
        }

        item = buffer_[current_head & Mask];
        head_.store(current_head + 1, std::memory_order_release);
        return true;
    }

    [[nodiscard]] std::size_t size() const noexcept
    {
        return tail_.load(std::memory_order_relaxed) - head_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] bool empty() const noexcept
    {
        return size() == 0;
    }

private:
    static constexpr std::size_t Mask = Capacity - 1;
    alignas(64) std::array<T, Capacity> buffer_{};
    alignas(64) std::atomic<std::size_t> head_{0};
    alignas(64) std::atomic<std::size_t> tail_{0};
};
