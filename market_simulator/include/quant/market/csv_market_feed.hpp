#pragma once

#include "quant/market/csv_reader.hpp"
#include "quant/market/market_feed.hpp"

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

class CsvMarketFeed final : public IMarketDataFeed
{
public:
    explicit CsvMarketFeed(const std::string& csv_path, double train_split_ratio = 0.70) : path_(csv_path)
    {
        try
        {
            dataset_ = CsvReader::load(csv_path);
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error("Failed to load market data from '" + csv_path + "': " + e.what());
        }

        if (dataset_.size() < 10)
        {
            throw std::runtime_error("Insufficient market bars loaded for simulation.");
        }

        train_size_ = std::max<std::size_t>(2, static_cast<std::size_t>(static_cast<double>(dataset_.size()) * train_split_ratio));
        cursor_     = train_size_;
    }

    [[nodiscard]] bool next(Tick& out_tick) noexcept override
    {
        if (cursor_ >= dataset_.size())
        {
            return false;
        }
        out_tick = dataset_[cursor_++];
        return true;
    }

    [[nodiscard]] bool is_exhausted() const noexcept override
    {
        return cursor_ >= dataset_.size();
    }

    [[nodiscard]] std::span<const Tick> warmup_bars() const noexcept override
    {
        return dataset_.bars().subspan(0, train_size_);
    }

    [[nodiscard]] std::string_view source_name() const noexcept override
    {
        return dataset_.empty() ? std::string_view{path_} : std::string_view{dataset_[0].source};
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

    void reset() noexcept
    {
        cursor_ = train_size_;
    }

private:
    std::string   path_;
    MarketDataset dataset_;
    std::size_t   train_size_{0};
    std::size_t   cursor_{0};
};
