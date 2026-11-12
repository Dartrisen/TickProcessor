#pragma once

#include "quant/core/config.hpp"
#include "quant/core/types.hpp"
#include "quant/execution/execution_gateway.hpp"
#include "quant/features/feature_engine.hpp"
#include "quant/market/market_feed.hpp"
#include "quant/models/kinetic_model.hpp"
#include "quant/models/walk_forward_calibrator.hpp"
#include "quant/risk/risk_manager.hpp"
#include "quant/session/session_observer.hpp"

#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

template <std::size_t BINS, std::size_t VOL_WINDOW, template <std::size_t> class KernelT = EmpiricalKernel>
class KineticTradingSession
{
public:
    KineticTradingSession(const EngineConfig& cfg, IExecutionGateway& broker, ISessionObserver& observer = default_observer(), KernelT<BINS> kernel_builder = {})
            : cfg_(cfg), broker_(broker), observer_(observer), engine_(cfg.x_max), calibrator_(cfg.recalibration_bars, std::move(kernel_builder)), risk_engine_(cfg.risk_limits),
              current_dist_(engine_.initial_uniform_distribution())
    {
    }

    void warmup(std::span<const Tick> warmup_bars)
    {
        std::vector<DerivedFeatures> train_features;
        train_features.reserve(warmup_bars.size());

        DerivedFeatures feat{};
        for (const auto& bar : warmup_bars)
        {
            if (feature_engine_.process(bar, feat))
            {
                train_features.push_back(feat);
            }
        }

        calibrator_.initialize(engine_, std::move(train_features));
    }

    void run(IMarketDataFeed& feed)
    {
        observer_.on_session_start();

        bool   has_first_bar = false;
        double start_price   = 0.0;
        Tick   last_tick{};
        Tick   tick{};

        while (!feed.is_exhausted())
        {
            if (!feed.next(tick))
            {
                std::this_thread::yield();
                continue;
            }

            if (!has_first_bar)
            {
                start_price   = tick.close;
                has_first_bar = true;
            }
            last_tick = tick;

            on_bar_open(tick);
            on_intra_bar(tick);
            on_bar_close(tick);
        }

        if (has_first_bar && has_open_position())
        {
            flatten_position(last_tick.datetime, last_tick.close, "END-OF-DATA FLATTEN ");
            (void) broker_.on_bar_update(last_tick, 0.0, 0.0);
        }

        observer_.on_session_end(
                SessionSummary{
                        .initial_capital = cfg_.initial_capital,
                        .start_price     = start_price,
                        .final_price     = last_tick.close,
                        .sharpe_ratio    = broker_.sharpe_ratio(),
                },
                broker_.account());
    }

private:
    using KineticEngine = HighPerformanceKineticEngine<BINS>;
    using StateVector   = typename KineticEngine::StateVector;

    struct PendingOrder
    {
        OrderSignal signal{OrderSignal::Hold};
        double      signal_close_price{0.0};
        double      allocation_pct{0.0};
        StateVector step1_dist{};
    };

    static NullSessionObserver& default_observer() noexcept
    {
        static NullSessionObserver instance{};
        return instance;
    }

    [[nodiscard]] bool has_open_position() const noexcept
    {
        return broker_.account().position_shares != 0.0;
    }

    [[nodiscard]] PortfolioState current_portfolio(double mark_price) const noexcept
    {
        const auto& acc = broker_.account();
        return PortfolioState{
                .equity                = acc.equity(mark_price),
                .position_shares       = acc.position_shares,
                .current_price         = mark_price,
                .target_allocation_pct = cfg_.allocation_pct,
                .allow_pyramiding      = false,
        };
    }

    void flatten_position(FixedTimestamp timestamp, double price, std::string_view label)
    {
        if (!has_open_position())
            return;

        const auto fill = broker_.close_position(timestamp, price);
        observer_.on_close_fill(label, fill);
        bars_held_ = 0;
    }

    void on_bar_open(const Tick& tick)
    {
        if (pending_flatten_ && has_open_position())
        {
            flatten_position(tick.datetime, tick.open, "RISK FLATTEN EXIT   ");
            pending_flatten_ = false;
            pending_order_.reset();
            return;
        }

        if (!pending_order_.has_value())
            return;

        if (pending_order_->signal == OrderSignal::Close)
        {
            flatten_position(tick.datetime, tick.open, "EXECUTED EXIT/CLOSE ");
        }
        else
        {
            double exec_allocation = pending_order_->allocation_pct;
            if (revalidate_entry_at_open(tick, exec_allocation))
            {
                submit_entry_order(tick, exec_allocation);
            }
        }

        pending_order_.reset();
        pending_flatten_ = false;
    }

    [[nodiscard]] bool revalidate_entry_at_open(const Tick& tick, double& out_allocation)
    {
        if (!cfg_.recheck_at_open || pending_order_->signal_close_price <= 0.0 || tick.open <= 0.0)
            return true;

        const double         entry_gap     = std::log(tick.open / pending_order_->signal_close_price);
        const QuantForecast  open_forecast = engine_.analyze_and_predict_horizon(tick.open, pending_order_->step1_dist, cfg_.forecast_horizon, entry_gap);
        const SignalDecision open_decision = risk_engine_.evaluate(open_forecast, current_portfolio(tick.open));

        if (open_decision.signal != pending_order_->signal)
        {
            if (open_decision.should_flatten)
            {
                flatten_position(tick.datetime, tick.open, "RISK FLATTEN EXIT   ");
            }
            return false;
        }

        if (cfg_.dynamic_sizing)
        {
            out_allocation = open_decision.max_allowed_allocation_pct;
        }
        return true;
    }

    void submit_entry_order(const Tick& tick, double allocation)
    {
        const double      shares_before = broker_.account().position_shares;
        const std::size_t fills_before  = broker_.fills().size();

        (void) broker_.execute_order(pending_order_->signal, tick.datetime, tick.open, allocation);

        for (std::size_t idx = fills_before; idx < broker_.fills().size(); ++idx)
        {
            observer_.on_execution_fill(broker_.fills()[idx]);
        }

        const double shares_after = broker_.account().position_shares;
        if (shares_after == 0.0 || (shares_before <= 0.0 && shares_after > 0.0) || (shares_before >= 0.0 && shares_after < 0.0))
        {
            bars_held_ = 0;
        }
    }

    void on_intra_bar(const Tick& tick)
    {
        if (!has_open_position())
        {
            bars_held_ = 0;
            (void) broker_.on_bar_update(tick, 0.0, 0.0);
            return;
        }

        ++bars_held_;
        const auto bracket_fill = broker_.on_bar_update(tick, cfg_.stop_loss_pct, cfg_.take_profit_pct);
        if (bracket_fill.closed_position)
        {
            observer_.on_close_fill("INTRA-BAR BRACKET   ", bracket_fill);
            bars_held_ = 0;
        }
    }

    void on_bar_close(const Tick& tick)
    {
        DerivedFeatures feat{};
        if (!feature_engine_.process(tick, feat))
            return;

        current_dist_ = engine_.state_delta(engine_.state_index(feat.log_return));
        engine_.evolve(current_dist_, next_dist_);

        const QuantForecast  forecast = engine_.analyze_and_predict_horizon(tick.close, next_dist_, cfg_.forecast_horizon, 0.0);
        const SignalDecision decision = risk_engine_.evaluate(forecast, current_portfolio(tick.close));

        schedule_next_action(tick, decision);
        calibrator_.on_new_feature(engine_, feat);
    }

    void schedule_next_action(const Tick& tick, const SignalDecision& decision)
    {
        const bool is_open     = has_open_position();
        const bool in_min_hold = is_open && (bars_held_ < cfg_.min_hold_bars);
        const bool time_stop   = is_open && (cfg_.max_hold_bars > 0) && (bars_held_ >= cfg_.max_hold_bars);

        if (decision.should_flatten)
        {
            pending_flatten_ = true;
            pending_order_.reset();
        }
        else if (in_min_hold)
        {
            pending_order_.reset();
        }
        else if (decision.signal == OrderSignal::Buy || decision.signal == OrderSignal::Sell)
        {
            const double alloc = cfg_.dynamic_sizing ? decision.max_allowed_allocation_pct : cfg_.allocation_pct;
            pending_order_     = PendingOrder{decision.signal, tick.close, alloc, next_dist_};
            pending_flatten_   = false;
        }
        else if (is_open && (decision.signal == OrderSignal::Close || time_stop))
        {
            pending_order_   = PendingOrder{OrderSignal::Close, tick.close, 0.0, next_dist_};
            pending_flatten_ = false;
        }
    }

    const EngineConfig&                  cfg_;
    IExecutionGateway&                   broker_;
    ISessionObserver&                    observer_;
    KineticEngine                        engine_;
    WalkForwardCalibrator<BINS, KernelT> calibrator_;
    OnlineFeatureEngine<VOL_WINDOW>      feature_engine_{};
    RiskEngine                           risk_engine_;
    StateVector                          current_dist_{};
    StateVector                          next_dist_{};
    std::optional<PendingOrder>          pending_order_{};
    bool                                 pending_flatten_{false};
    std::size_t                          bars_held_{0};
};
