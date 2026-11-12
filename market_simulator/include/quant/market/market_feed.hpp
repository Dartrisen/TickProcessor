#pragma once

#include "quant/core/types.hpp"
#include "spsc_queue.hpp"

#include <atomic>
#include <concepts>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>

namespace detail
{
template <template <typename, std::size_t> class QueueT, typename T, std::size_t Cap>
auto resolve_spsc_queue() -> QueueT<T, Cap>;

template <template <typename> class QueueT, typename T, std::size_t Cap>
auto resolve_spsc_queue() -> QueueT<T>;
} // namespace detail

// Unified feed interface for seamless switching between CSV replay and a live broker stream.
class IMarketDataFeed
{
public:
    virtual ~IMarketDataFeed() = default;

    // Pulls the next available OHLCV bar/tick.
    // Returns true if `out_tick` was populated; false if no tick is ready or the feed ended.
    [[nodiscard]] virtual bool next(Tick& out_tick) = 0;

    // True when a finite historical dataset has finished replaying (always false for live feeds).
    [[nodiscard]] virtual bool is_exhausted() const noexcept = 0;

    // Optional initial warmup bars for in-sample kernel calibration prior to live/OOS execution.
    [[nodiscard]] virtual std::span<const Tick> warmup_bars() const noexcept
    {
        return {};
    }

    [[nodiscard]] virtual std::string_view source_name() const noexcept = 0;
};

// Lock-free SPSC feed adapter for real-time broker connections.
// Broker network/callback thread pushes via push_tick(); strategy thread polls via next().
template <std::size_t Capacity = 4096>
class LiveRingBufferFeed final : public IMarketDataFeed
{
    using Queue = decltype(detail::resolve_spsc_queue<SPSCQueue, Tick, Capacity>());

public:
    explicit LiveRingBufferFeed(std::string source = "LIVE_BROKER") : source_(std::move(source)), queue_(make_queue()) {}

    // Non-blocking push for real-time broker threads
    [[nodiscard]] bool push_tick(const Tick& tick) noexcept
    {
        bool pushed = false;
        if constexpr (std::is_same_v<decltype(queue_.push(tick)), bool>)
        {
            pushed = queue_.push(tick);
        }
        else
        {
            queue_.push(tick);
            pushed = true;
        }

        if (pushed)
        {
            in_flight_.fetch_add(1, std::memory_order_release);
        }
        return pushed;
    }

    // Backpressure-aware push for CSV replay threads (never drops bars when queue is full)
    void push_blocking(const Tick& tick) noexcept
    {
        while (!push_tick(tick))
        {
            std::this_thread::yield();
        }
    }

    void mark_exhausted() noexcept
    {
        exhausted_.store(true, std::memory_order_release);
    }

    [[nodiscard]] bool next(Tick& out_tick) noexcept override
    {
        bool popped = false;
        if constexpr (requires {
                          { queue_.pop(out_tick) } -> std::convertible_to<bool>;
                      })
        {
            popped = queue_.pop(out_tick);
        }
        else
        {
            auto item = queue_.pop();
            if (item.has_value())
            {
                out_tick = std::move(*item);
                popped   = true;
            }
        }

        if (popped)
        {
            in_flight_.fetch_sub(1, std::memory_order_acq_rel);
        }
        return popped;
    }

    // Only exhausted once the producer finishes AND all in-flight ticks are drained
    [[nodiscard]] bool is_exhausted() const noexcept override
    {
        return exhausted_.load(std::memory_order_acquire) && pending() == 0;
    }

    [[nodiscard]] std::string_view source_name() const noexcept override
    {
        return source_;
    }

    [[nodiscard]] std::size_t pending() const noexcept
    {
        return in_flight_.load(std::memory_order_acquire);
    }

private:
    static Queue make_queue()
    {
        if constexpr (std::is_constructible_v<Queue, std::size_t>)
        {
            return Queue(Capacity);
        }
        else
        {
            return Queue{};
        }
    }

    std::string              source_;
    Queue                    queue_;
    std::atomic<std::size_t> in_flight_{0};
    std::atomic<bool>        exhausted_{false};
};
