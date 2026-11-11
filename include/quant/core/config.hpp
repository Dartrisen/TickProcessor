#pragma once

#include "quant/risk/risk_manager.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>

#ifndef DEFAULT_CONFIG_PATH
#define DEFAULT_CONFIG_PATH "quant.env"
#endif

struct EngineConfig
{
    // model parameters
    std::size_t bins{16};
    double      x_max{0.015};
    std::size_t volatility_window{20};

    // risk engine parameters
    RiskLimits risk_limits{};

    // exchange simulation parameters
    double initial_capital{100000.0};
    double allocation_pct{0.95};
    double commission_rate{0.0001};
    double fixed_fee{0.00};
    double slippage_bps{0.0001};

    // ingestion & infrastructure
    std::string data_path{"data/SPY_clean_1min.csv"};
    std::size_t async_calibration_ms{500};
};

class ConfigLoader
{
public:
    static EngineConfig load(const std::string& env_filepath = "quant.env")
    {
        EngineConfig          cfg{};
        std::filesystem::path target_path = resolve_config_path(env_filepath);

        std::ifstream file(target_path);
        if (!file.is_open())
        {
            std::cout << "\033[1;33m[!] ConfigLoader: '" << target_path.string() << "' not found. Falling back to defaults.\033[0m\n";
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

                auto comment_pos = value.find('#');
                if (comment_pos != std::string::npos)
                {
                    value = std::string(trim(value.substr(0, comment_pos)));
                }

                kv[key] = value;
            }
        }

        // map key-values into engine config struct
        if (auto it = kv.find("MODEL_BINS"); it != kv.end())
            cfg.bins = std::stoull(it->second);
        if (auto it = kv.find("MODEL_X_MAX"); it != kv.end())
            cfg.x_max = std::stod(it->second);
        if (auto it = kv.find("FEATURE_VOLATILITY_WINDOW"); it != kv.end())
            cfg.volatility_window = std::stoull(it->second);

        if (auto it = kv.find("RISK_MAX_VAR_95"); it != kv.end())
            cfg.risk_limits.max_var_95 = std::stod(it->second);
        if (auto it = kv.find("RISK_MIN_ENTROPY"); it != kv.end())
            cfg.risk_limits.min_entropy = std::stod(it->second);
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

        std::cout << "\033[1;32m[+] ConfigLoader: Loaded runtime parameters from '" << target_path.string() << "'.\033[0m\n";
        return cfg;
    }

private:
    static std::filesystem::path resolve_config_path(const std::string& requested_path)
    {
        // check environment variable override
        if (const char* env_path = std::getenv("QUANT_ENV_PATH"))
        {
            std::filesystem::path p(env_path);
            if (std::filesystem::exists(p))
                return std::filesystem::canonical(p);
        }

        // check explicit path parameter or local working directory
        std::filesystem::path direct_path(requested_path);
        if (std::filesystem::exists(direct_path))
        {
            return std::filesystem::canonical(direct_path);
        }

        return direct_path;
    }

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
