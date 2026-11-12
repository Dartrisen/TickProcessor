#pragma once

#include "quant/core/dispatch_table.hpp"
#include "quant/risk/risk_manager.hpp"

#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

struct EngineConfig
{
    // Model & Feature parameters
    KernelType  kernel_type{KernelType::Empirical};
    std::size_t bins{16};
    double      x_max{0.015};
    std::size_t volatility_window{20};
    double      train_split_ratio{0.70};
    std::size_t forecast_horizon{3};
    std::size_t recalibration_bars{390};
    std::size_t min_hold_bars{1};
    std::size_t max_hold_bars{0}; // 0 = disabled (let positions run until exit signal or risk limit)
    bool        recheck_at_open{true};
    bool        dynamic_sizing{true};
    double      stop_loss_pct{0.0};
    double      take_profit_pct{0.0};

    // Risk Engine parameters
    RiskLimits risk_limits{};

    // Exchange Simulation parameters
    double initial_capital{10000.0};
    double allocation_pct{0.95};
    double commission_rate{0.0001};
    double fixed_fee{0.00};
    double slippage_bps{0.0001};
    double bars_per_year{98280.0};

    // Ingestion & Infrastructure
    std::string data_path{"data/SPY_clean_1min.csv"};
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

        if (auto it = kv.find("MODEL_KERNEL"); it != kv.end())
        {
            const auto parsed = DispatchLimits::parse_kernel(it->second);
            if (!parsed.has_value())
            {
                throw std::runtime_error("Unsupported MODEL_KERNEL='" + it->second + "'. Supported values: empirical, gaussian, dirichlet, phase_space.");
            }
            cfg.kernel_type = *parsed;
        }

        if (auto it = kv.find("MODEL_BINS"); it != kv.end())
            cfg.bins = std::stoull(it->second);
        if (auto it = kv.find("MODEL_X_MAX"); it != kv.end())
            cfg.x_max = std::stod(it->second);
        if (auto it = kv.find("FEATURE_VOLATILITY_WINDOW"); it != kv.end())
            cfg.volatility_window = std::stoull(it->second);
        if (auto it = kv.find("TRAIN_SPLIT_RATIO"); it != kv.end())
            cfg.train_split_ratio = std::stod(it->second);
        if (auto it = kv.find("MODEL_FORECAST_HORIZON"); it != kv.end())
            cfg.forecast_horizon = std::stoull(it->second);
        if (auto it = kv.find("MODEL_RECALIBRATION_BARS"); it != kv.end())
            cfg.recalibration_bars = std::stoull(it->second);
        if (auto it = kv.find("RISK_MIN_HOLD_BARS"); it != kv.end())
            cfg.min_hold_bars = std::stoull(it->second);
        if (auto it = kv.find("RISK_MAX_HOLD_BARS"); it != kv.end())
            cfg.max_hold_bars = std::stoull(it->second);
        if (auto it = kv.find("RISK_RECHECK_AT_OPEN"); it != kv.end())
            cfg.recheck_at_open = (it->second == "1" || it->second == "true" || it->second == "TRUE" || it->second == "yes");
        if (auto it = kv.find("RISK_DYNAMIC_SIZING"); it != kv.end())
            cfg.dynamic_sizing = (it->second == "1" || it->second == "true" || it->second == "TRUE" || it->second == "yes");
        if (auto it = kv.find("RISK_EXIT_EXPECTED_RETURN"); it != kv.end())
            cfg.risk_limits.exit_expected_return = std::stod(it->second);
        if (auto it = kv.find("SIM_STOP_LOSS_PCT"); it != kv.end())
            cfg.stop_loss_pct = std::stod(it->second);
        if (auto it = kv.find("SIM_TAKE_PROFIT_PCT"); it != kv.end())
            cfg.take_profit_pct = std::stod(it->second);

        if (cfg.forecast_horizon == 0 || cfg.forecast_horizon > 10)
            throw std::runtime_error("MODEL_FORECAST_HORIZON must be between 1 and 10.");

        if (auto it = kv.find("RISK_MAX_VAR_95"); it != kv.end())
            cfg.risk_limits.max_var_95 = std::stod(it->second);
        if (auto it = kv.find("RISK_MAX_CVAR_95"); it != kv.end())
            cfg.risk_limits.max_cvar_95 = std::stod(it->second);
        if (auto it = kv.find("RISK_SHORT_MAX_VAR_95"); it != kv.end())
            cfg.risk_limits.short_max_var_95 = std::stod(it->second);
        if (auto it = kv.find("RISK_SHORT_MAX_CVAR_95"); it != kv.end())
            cfg.risk_limits.short_max_cvar_95 = std::stod(it->second);
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
        if (auto it = kv.find("SIM_BARS_PER_YEAR"); it != kv.end())
            cfg.bars_per_year = std::stod(it->second);

        if (auto it = kv.find("DATA_PATH"); it != kv.end())
            cfg.data_path = it->second;

        if (!DispatchLimits::is_supported_bin(cfg.bins))
        {
            throw std::runtime_error("MODEL_BINS=" + std::to_string(cfg.bins) + " is not in the compiled template dispatch table. Supported values: 8, 12, 16, 24, 32, 64.");
        }

        if (!DispatchLimits::is_supported_window(cfg.volatility_window))
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

        if (cfg.bars_per_year <= 0.0)
            throw std::runtime_error("SIM_BARS_PER_YEAR must be positive.");

        if (cfg.risk_limits.max_var_95 < 0.0)
            throw std::runtime_error("RISK_MAX_VAR_95 must be non-negative.");

        if (cfg.risk_limits.max_entropy < 0.0)
            throw std::runtime_error("RISK_MAX_ENTROPY must be non-negative.");

        std::cout << "\033[1;32m[+] ConfigLoader: Loaded runtime parameters from '" << env_filepath << "' (KERNEL=" << DispatchLimits::kernel_name(cfg.kernel_type)
                  << ", BINS=" << cfg.bins << ", VOL_WIN=" << cfg.volatility_window << ").\033[0m\n";
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
