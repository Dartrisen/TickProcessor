#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <string_view>

enum class KernelType
{
    Empirical,
    Gaussian,
    Dirichlet,
    PhaseSpace
};

struct DispatchLimits
{
    static constexpr std::array<std::size_t, 6> Bins    = {8, 12, 16, 24, 32, 64};
    static constexpr std::array<std::size_t, 6> Windows = {10, 14, 20, 30, 50, 60};

    [[nodiscard]] static constexpr bool is_supported_bin(std::size_t b) noexcept
    {
        return std::find(Bins.begin(), Bins.end(), b) != Bins.end();
    }

    [[nodiscard]] static constexpr bool is_supported_window(std::size_t w) noexcept
    {
        return std::find(Windows.begin(), Windows.end(), w) != Windows.end();
    }

    [[nodiscard]] static constexpr std::optional<KernelType> parse_kernel(std::string_view name) noexcept
    {
        auto iequals = [](std::string_view a, std::string_view b) constexpr noexcept
        {
            if (a.size() != b.size())
                return false;
            for (std::size_t i = 0; i < a.size(); ++i)
            {
                const char ca = (a[i] >= 'A' && a[i] <= 'Z') ? static_cast<char>(a[i] + ('a' - 'A')) : a[i];
                const char cb = (b[i] >= 'A' && b[i] <= 'Z') ? static_cast<char>(b[i] + ('a' - 'A')) : b[i];
                if (ca != cb)
                    return false;
            }
            return true;
        };

        if (iequals(name, "empirical"))
            return KernelType::Empirical;
        if (iequals(name, "gaussian"))
            return KernelType::Gaussian;
        if (iequals(name, "dirichlet"))
            return KernelType::Dirichlet;
        if (iequals(name, "phase_space"))
            return KernelType::PhaseSpace;
        return std::nullopt;
    }

    [[nodiscard]] static constexpr std::string_view kernel_name(KernelType type) noexcept
    {
        switch (type)
        {
            case KernelType::Empirical:
                return "empirical";
            case KernelType::Gaussian:
                return "gaussian";
            case KernelType::Dirichlet:
                return "dirichlet";
            case KernelType::PhaseSpace:
                return "phase_space";
        }
        return "unknown";
    }
};
