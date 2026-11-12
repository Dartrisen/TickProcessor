// market_simulator/include/quant/market/csv_replay_feed.hpp
#pragma once

#include "quant/market/csv_reader.hpp"
#include "quant/market/market_feed.hpp"

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>

template <std::size_t Capacity = 4096>
class CsvReplayFeed final : public IMarketDataFeed
{
public:
    explicit CsvReplayFeed(const std::string& csv_path, double train_split_ratio = 0.70)
            : dataset_(CsvReader::load(csv_path)), train_size_(compute_train_size(dataset_.size(), train_split_ratio)), ring_feed_(dataset_[0].source),
              producer_(
                      [this](std::stop_token stoken)
                      {
                          for (const auto& bar : dataset_.bars().subspan(train_size_))
                          {
                              if (stoken.stop_requested())
                                  break;
                              ring_feed_.push_blocking(bar, stoken);
                          }
                          ring_feed_.mark_exhausted();
                      })
    {
    }

    [[nodiscard]] bool next(Tick& out_tick) noexcept override
    {
        return ring_feed_.next(out_tick);
    }
    [[nodiscard]] bool is_exhausted() const noexcept override
    {
        return ring_feed_.is_exhausted();
    }
    [[nodiscard]] std::string_view source_name() const noexcept override
    {
        return ring_feed_.source_name();
    }
    [[nodiscard]] std::span<const Tick> warmup_bars() const noexcept override
    {
        return dataset_.bars().subspan(0, train_size_);
    }

    [[nodiscard]] std::size_t total_bars() const noexcept
    {
        return dataset_.size();
    }
    [[nodiscard]] std::size_t warmup_size() const noexcept
    {
        return train_size_;
    }
    [[nodiscard]] std::size_t replay_size() const noexcept
    {
        return dataset_.size() - train_size_;
    }

private:
    static std::size_t compute_train_size(std::size_t total, double ratio)
    {
        if (total < 10)
            throw std::runtime_error("Insufficient market bars loaded for simulation.");
        return std::max<std::size_t>(2, static_cast<std::size_t>(static_cast<double>(total) * ratio));
    }

    MarketDataset                dataset_;
    std::size_t                  train_size_;
    LiveRingBufferFeed<Capacity> ring_feed_;
    std::jthread                 producer_; // Declared last so it joins before ring_feed_/dataset_ destroy
};
