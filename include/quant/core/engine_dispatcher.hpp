#pragma once

#include "quant/core/config.hpp"
#include "quant/core/dispatch_table.hpp"
#include "quant/models/kernels.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>

class EngineDispatcher
{
public:
    template <typename Fn>
    [[nodiscard]] static bool try_dispatch(const EngineConfig& cfg, Fn&& fn)
    {
        using BinIndices = std::make_index_sequence<DispatchLimits::Bins.size()>;
        switch (cfg.kernel_type)
        {
            case KernelType::Empirical:
                return dispatch_bins<EmpiricalKernel>(cfg, fn, BinIndices{});
            case KernelType::Gaussian:
                return dispatch_bins<GaussianDiffusionKernel>(cfg, fn, BinIndices{});
            case KernelType::Dirichlet:
                return dispatch_bins<AdaptiveDirichletKernel>(cfg, fn, BinIndices{});
            case KernelType::PhaseSpace:
                return dispatch_bins<FactorizedPhaseSpaceKernel>(cfg, fn, BinIndices{});
        }
        return false;
    }

    template <typename Fn>
    static void dispatch(const EngineConfig& cfg, Fn&& fn)
    {
        if (!try_dispatch(cfg, std::forward<Fn>(fn)))
        {
            throw std::runtime_error("Unsupported MODEL_KERNEL, MODEL_BINS, or FEATURE_VOLATILITY_WINDOW.");
        }
    }

private:
    template <template <std::size_t> class KernelT, std::size_t Bins, typename Fn, std::size_t... WinIdx>
    [[nodiscard]] static bool dispatch_window(const EngineConfig& cfg, Fn& fn, std::index_sequence<WinIdx...>)
    {
        return ((cfg.volatility_window == DispatchLimits::Windows[WinIdx] ? (fn.template operator()<KernelT, Bins, DispatchLimits::Windows[WinIdx]>(cfg), true) : false) || ...);
    }

    template <template <std::size_t> class KernelT, typename Fn, std::size_t... BinIdx>
    [[nodiscard]] static bool dispatch_bins(const EngineConfig& cfg, Fn& fn, std::index_sequence<BinIdx...>)
    {
        using WindowIndices = std::make_index_sequence<DispatchLimits::Windows.size()>;
        return ((cfg.bins == DispatchLimits::Bins[BinIdx] ? dispatch_window<KernelT, DispatchLimits::Bins[BinIdx]>(cfg, fn, WindowIndices{}) : false) || ...);
    }
};
