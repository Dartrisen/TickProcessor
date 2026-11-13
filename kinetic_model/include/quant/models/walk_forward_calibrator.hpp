#pragma once

#include "quant/core/matrix.hpp"
#include "quant/core/types.hpp"
#include "quant/models/interfaces.hpp"
#include "quant/models/kernels.hpp"
#include "quant/models/kinetic_model.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

template <std::size_t Bins, template <std::size_t> class KernelT = EmpiricalKernel>
class WalkForwardCalibrator
{
    static_assert(std::derived_from<KernelT<Bins>, ITransitionKernel<Bins>>, "KernelT<Bins> must inherit from ITransitionKernel<Bins>");

public:
    static constexpr std::size_t DefaultRecalibrationBars = 390;

    explicit WalkForwardCalibrator(std::size_t recalibration_interval = DefaultRecalibrationBars, KernelT<Bins> builder = KernelT<Bins>{})
            : recalibration_interval_(recalibration_interval), builder_(std::move(builder))
    {
    }

    void initialize(HighPerformanceKineticEngine<Bins>& engine, std::vector<DerivedFeatures> initial_features)
    {
        rolling_features_   = std::move(initial_features);
        calibration_window_ = rolling_features_.size();
        bars_since_recal_   = 0;

        rebuild_and_swap(engine);
    }

    void on_new_feature(HighPerformanceKineticEngine<Bins>& engine, const DerivedFeatures& feat)
    {
        rolling_features_.push_back(feat);

        if (recalibration_interval_ > 0 && ++bars_since_recal_ >= recalibration_interval_ && rolling_features_.size() >= calibration_window_)
        {
            rebuild_and_swap(engine);
            bars_since_recal_ = 0;
        }
    }

    [[nodiscard]] const FlatMatrix<double, Bins>& active_kernel() const noexcept
    {
        return buffers_[write_index_ ^ 1];
    }

private:
    void rebuild_and_swap(HighPerformanceKineticEngine<Bins>& engine)
    {
        const std::span<const DerivedFeatures> trailing_slice(rolling_features_.data() + (rolling_features_.size() - calibration_window_), calibration_window_);

        auto& next_buffer = buffers_[write_index_];
        builder_.build_kernel(engine.dx(), engine.x_max(), trailing_slice, next_buffer);
        engine.update_kernel(&next_buffer);
        write_index_ ^= 1;
    }

    std::size_t                             recalibration_interval_;
    std::array<FlatMatrix<double, Bins>, 2> buffers_{};
    std::size_t                             write_index_{0};
    KernelT<Bins>                           builder_;
    std::vector<DerivedFeatures>            rolling_features_{};
    std::size_t                             calibration_window_{0};
    std::size_t                             bars_since_recal_{0};
};
