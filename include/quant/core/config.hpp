#pragma once

#include "quant/risk/risk_manager.hpp"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

struct EngineConfig
{
    // Model & Feature parameters
    std::size_t bins{16};
    double      x_max{0.015};
    std::size_t volatility_window{20};
    double      train_split_ratio{0.70};

    // Risk Engine parameters
    RiskLimits risk_limits{};

    // Exchange Simulation parameters
    double initial_capital{10000.0};
    double allocation_pct{0.95};
    double commission_rate{0.0001};
    double fixed_fee{0.00};
    double slippage_bps{0.0001};

    // Ingestion & Infrastructure
    std::string data_path{"/Users/cdg546/Downloads/data/SPY_clean_1min.csv"};
    std::size_t async_calibration_ms{500};
};

class ConfigLoader
{
public:
    static EngineConfig load(const std::string& env_filepath = "quant.env")
    {
        EngineConfig  cfg{};
        std::ifstream file(env_filepath);
        if (!file.is_open())
        {
            std::cout << "\033[1;33m[!] ConfigLoader: '" << env_filepath << "' not found. Falling back to default hyper-parameters.\033[0m\n";
            return cfg;
        }

        std::unordered_map<std::string, std::string> kv;
        std::string                                  line;

        while (std::getline(file, line))
        {
            auto trimmed = trim(line);
            if (trimmed.empty() || trimmed.front() == '#')
                continue;

            auto pos = trimmed.find('=');
            if (pos != std::string_view::npos)
            {
                std::string key   = std::string(trim(trimmed.substr(0, pos)));
                std::string value = std::string(trim(trimmed.substr(pos + 1)));

                // Strip inline comments if present
                auto comment_pos = value.find('#');
                if (comment_pos != std::string::npos)
                {
                    value = std::string(trim(value.substr(0, comment_pos)));
                }

                kv[key] = value;
            }
        }

        if (kv.contains("RISK_MIN_ENTROPY"))
        {
            throw std::runtime_error("Deprecated config key 'RISK_MIN_ENTROPY' found in " + env_filepath + ". Replace it with 'RISK_MAX_ENTROPY'.");
        }

        // Parse key-value map into EngineConfig struct
        if (auto it = kv.find("MODEL_BINS"); it != kv.end())
            cfg.bins = std::stoull(it->second);
        if (auto it = kv.find("MODEL_X_MAX"); it != kv.end())
            cfg.x_max = std::stod(it->second);
        if (auto it = kv.find("FEATURE_VOLATILITY_WINDOW"); it != kv.end())
            cfg.volatility_window = std::stoull(it->second);
        if (auto it = kv.find("TRAIN_SPLIT_RATIO"); it != kv.end())
            cfg.train_split_ratio = std::stod(it->second);

        if (auto it = kv.find("RISK_MAX_VAR_95"); it != kv.end())
            cfg.risk_limits.max_var_95 = std::stod(it->second);
        if (auto it = kv.find("RISK_MAX_ENTROPY"); it != kv.end())
            cfg.risk_limits.max_entropy = std::stod(it->second);
        if (auto it = kv.find("RISK_MIN_EXPECTED_RETURN"); it != kv.end())
            cfg.risk_limits.min_expected_return = std::stod(it->second);
        if (auto it = kv.find("RISK_MIN_BULLISH_PROB"); it != kv.end())
            cfg.risk_limits.min_bullish_prob = std::stod(it->second);
        if (auto it = kv.find("RISK_MAX_BEARISH_PROB"); it != kv.end())
            cfg.risk_limits.max_bearish_prob = std::stod(it->second);

        if (auto it = kv.find("SIM_INITIAL_CAPITAL"); it != kv.end())
            cfg.initial_capital = std::stod(it->second);
        if (auto it = kv.find("SIM_ALLOCATION_PCT"); it != kv.end())
            cfg.allocation_pct = std::stod(it->second);
        if (auto it = kv.find("SIM_COMMISSION_RATE"); it != kv.end())
            cfg.commission_rate = std::stod(it->second);
        if (auto it = kv.find("SIM_FIXED_FEE"); it != kv.end())
            cfg.fixed_fee = std::stod(it->second);
        if (auto it = kv.find("SIM_SLIPPAGE_BPS"); it != kv.end())
            cfg.slippage_bps = std::stod(it->second);

        if (auto it = kv.find("DATA_PATH"); it != kv.end())
            cfg.data_path = it->second;
        if (auto it = kv.find("ASYNC_CALIBRATION_MS"); it != kv.end())
            cfg.async_calibration_ms = std::stoull(it->second);

        // Validate compile-time template buckets and parameter bounds
        if (cfg.bins != 8 && cfg.bins != 12 && cfg.bins != 16 && cfg.bins != 24 && cfg.bins != 32 && cfg.bins != 64)
        {
            throw std::runtime_error("MODEL_BINS=" + std::to_string(cfg.bins) + " is not in the compiled template dispatch table. Supported values: 8, 12, 16, 24, 32, 64.");
        }

        if (cfg.volatility_window != 10 && cfg.volatility_window != 14 && cfg.volatility_window != 20 && cfg.volatility_window != 30 && cfg.volatility_window != 50 &&
            cfg.volatility_window != 60)
        {
            throw std::runtime_error("FEATURE_VOLATILITY_WINDOW=" + std::to_string(cfg.volatility_window) +
                                     " is not in the compiled template dispatch table. Supported values: 10, 14, 20, 30, 50, 60.");
        }

        if (cfg.x_max <= 0.0)
            throw std::runtime_error("MODEL_X_MAX must be positive.");

        if (cfg.train_split_ratio <= 0.0 || cfg.train_split_ratio >= 1.0)
            throw std::runtime_error("TRAIN_SPLIT_RATIO must be strictly between 0.0 and 1.0.");

        if (cfg.allocation_pct <= 0.0 || cfg.allocation_pct > 1.0)
            throw std::runtime_error("SIM_ALLOCATION_PCT must be in (0, 1].");

        if (cfg.risk_limits.max_var_95 < 0.0)
            throw std::runtime_error("RISK_MAX_VAR_95 must be non-negative.");

        if (cfg.risk_limits.max_entropy < 0.0)
            throw std::runtime_error("RISK_MAX_ENTROPY must be non-negative.");

        std::cout << "\033[1;32m[+] ConfigLoader: Loaded runtime parameters from '" << env_filepath << "' (BINS=" << cfg.bins << ", VOL_WIN=" << cfg.volatility_window
                  << ").\033[0m\n";
        return cfg;
    }

private:
    static std::string_view trim(std::string_view s)
    {
        constexpr std::string_view ws    = " \t\r\n";
        auto                       first = s.find_first_not_of(ws);
        if (first == std::string_view::npos)
            return {};
        auto last = s.find_last_not_of(ws);
        return s.substr(first, last - first + 1);
    }
};
