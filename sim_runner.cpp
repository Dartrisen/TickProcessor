#include "quant/core/config.hpp"
#include "quant/core/matrix.hpp"
#include "quant/core/types.hpp"
#include "quant/execution/execution_gateway.hpp"
#include "quant/execution/simulated_exchange.hpp"
#include "quant/features/feature_engine.hpp"
#include "quant/market/csv_replay_feed.hpp"
#include "quant/models/kernels.hpp"
#include "quant/models/kinetic_model.hpp"
#include "quant/risk/risk_manager.hpp"
#include "quant/telemetry/console_reporter.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

template <std::size_t BINS, std::size_t VOL_WINDOW>
class KineticTradingSession
{
public:
    KineticTradingSession(const EngineConfig& cfg, IExecutionGateway& broker)
            : cfg_(cfg), broker_(broker), engine_(cfg.x_max), kernel_(std::make_unique<FlatMatrix<double, BINS>>()), risk_engine_(cfg.risk_limits),
              current_dist_(engine_.initial_uniform_distribution())
    {
    }

    void warmup(std::span<const Tick> warmup_bars)
    {
        train_features_.reserve(warmup_bars.size());
        DerivedFeatures feat{};

        for (const auto& bar : warmup_bars)
        {
            if (feature_engine_.process(bar, feat))
                train_features_.push_back(feat);
        }

        kernel_builder_.build_kernel(engine_.dx(), engine_.x_max(), train_features_, *kernel_);
        engine_.update_kernel(kernel_.get());

        rolling_features_   = train_features_;
        calibration_window_ = train_features_.size();
    }

    void run(IMarketDataFeed& feed)
    {
        QuantConsoleReporter::render_header("ORDER GATEWAY & TRADE EXECUTION STREAM");

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

        if (has_first_bar && broker_.account().position_shares != 0.0)
        {
            const auto fill = broker_.close_position(last_tick.datetime, last_tick.close);
            QuantConsoleReporter::print_close_fill("END-OF-DATA FLATTEN ", fill);
            (void) broker_.on_bar_update(last_tick, 0.0, 0.0);
        }

        QuantConsoleReporter::print_account_performance(cfg_.initial_capital, last_tick.close, start_price, broker_.sharpe_ratio(), broker_.account());
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

    void on_bar_open(const Tick& tick)
    {
        if (pending_flatten_ && broker_.account().position_shares != 0.0)
        {
            const auto fill = broker_.close_position(tick.datetime, tick.open);
            QuantConsoleReporter::print_close_fill("RISK FLATTEN EXIT   ", fill);
            bars_held_       = 0;
            pending_flatten_ = false;
            pending_order_.reset();
            return;
        }

        if (!pending_order_.has_value())
            return;

        if (pending_order_->signal == OrderSignal::Close)
        {
            if (broker_.account().position_shares != 0.0)
            {
                const auto fill = broker_.close_position(tick.datetime, tick.open);
                QuantConsoleReporter::print_close_fill("EXECUTED EXIT/CLOSE ", fill);
                bars_held_ = 0;
            }
        }
        else
        {
            execute_validated_entry(tick);
        }

        pending_order_.reset();
        pending_flatten_ = false;
    }

    void execute_validated_entry(const Tick& tick)
    {
        double exec_allocation = pending_order_->allocation_pct;

        if (cfg_.recheck_at_open && pending_order_->signal_close_price > 0.0 && tick.open > 0.0)
        {
            const double        entry_gap     = std::log(tick.open / pending_order_->signal_close_price);
            const QuantForecast open_forecast = engine_.analyze_and_predict_horizon(tick.open, pending_order_->step1_dist, cfg_.forecast_horizon, entry_gap);

            const auto&          acc = broker_.account();
            const PortfolioState open_portfolio{
                    .equity                = acc.equity(tick.open),
                    .position_shares       = acc.position_shares,
                    .current_price         = tick.open,
                    .target_allocation_pct = cfg_.allocation_pct,
                    .allow_pyramiding      = false,
            };

            const SignalDecision open_decision = risk_engine_.evaluate(open_forecast, open_portfolio);
            if (open_decision.signal != pending_order_->signal)
            {
                if (open_decision.should_flatten && acc.position_shares != 0.0)
                {
                    const auto fill = broker_.close_position(tick.datetime, tick.open);
                    QuantConsoleReporter::print_close_fill("RISK FLATTEN EXIT   ", fill);
                    bars_held_ = 0;
                }
                return;
            }

            if (cfg_.dynamic_sizing)
                exec_allocation = open_decision.max_allowed_allocation_pct;
        }

        const double      shares_before = broker_.account().position_shares;
        const std::size_t fills_before  = broker_.fills().size();
        (void) broker_.execute_order(pending_order_->signal, tick.datetime, tick.open, exec_allocation);

        for (std::size_t idx = fills_before; idx < broker_.fills().size(); ++idx)
        {
            QuantConsoleReporter::print_execution_fill(broker_.fills()[idx]);
        }

        const double shares_after = broker_.account().position_shares;
        if (shares_after == 0.0 || (shares_before <= 0.0 && shares_after > 0.0) || (shares_before >= 0.0 && shares_after < 0.0))
        {
            bars_held_ = 0;
        }
    }

    void on_intra_bar(const Tick& tick)
    {
        if (broker_.account().position_shares != 0.0)
        {
            ++bars_held_;
            const auto bracket_fill = broker_.on_bar_update(tick, cfg_.stop_loss_pct, cfg_.take_profit_pct);
            if (bracket_fill.closed_position)
            {
                QuantConsoleReporter::print_close_fill("INTRA-BAR BRACKET   ", bracket_fill);
                bars_held_ = 0;
            }
        }
        else
        {
            bars_held_ = 0;
            (void) broker_.on_bar_update(tick, 0.0, 0.0);
        }
    }

    void on_bar_close(const Tick& tick)
    {
        DerivedFeatures feat{};
        if (!feature_engine_.process(tick, feat))
            return;

        current_dist_ = engine_.state_delta(engine_.state_index(feat.log_return));
        engine_.evolve(current_dist_, next_dist_);

        const auto&          acc = broker_.account();
        const PortfolioState portfolio{
                .equity                = acc.equity(tick.close),
                .position_shares       = acc.position_shares,
                .current_price         = tick.close,
                .target_allocation_pct = cfg_.allocation_pct,
                .allow_pyramiding      = false,
        };

        const QuantForecast  forecast = engine_.analyze_and_predict_horizon(tick.close, next_dist_, cfg_.forecast_horizon, 0.0);
        const SignalDecision decision = risk_engine_.evaluate(forecast, portfolio);

        const bool is_open     = (acc.position_shares != 0.0);
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

        rolling_features_.push_back(feat);
        if (++bars_since_recal_ >= 390 && rolling_features_.size() >= calibration_window_)
        {
            const std::span<const DerivedFeatures> slice(rolling_features_.data() + (rolling_features_.size() - calibration_window_), calibration_window_);
            kernel_builder_.build_kernel(engine_.dx(), engine_.x_max(), slice, *kernel_);
            engine_.update_kernel(kernel_.get());
            bars_since_recal_ = 0;
        }
    }

    const EngineConfig&                       cfg_;
    IExecutionGateway&                        broker_;
    KineticEngine                             engine_;
    std::unique_ptr<FlatMatrix<double, BINS>> kernel_;
    EmpiricalKernel<BINS>                     kernel_builder_;
    OnlineFeatureEngine<VOL_WINDOW>           feature_engine_;
    RiskEngine                                risk_engine_;
    StateVector                               current_dist_{};
    StateVector                               next_dist_{};
    std::vector<DerivedFeatures>              train_features_;
    std::vector<DerivedFeatures>              rolling_features_;
    std::size_t                               calibration_window_{0};
    std::size_t                               bars_since_recal_{0};
    std::optional<PendingOrder>               pending_order_;
    bool                                      pending_flatten_{false};
    std::size_t                               bars_held_{0};
};

template <std::size_t BINS, std::size_t VOL_WINDOW>
void run_simulation(const EngineConfig& cfg)
{
    const FeeStructure fees{.commission_rate = cfg.commission_rate, .fixed_fee_per_trade = cfg.fixed_fee, .slippage_bps = cfg.slippage_bps};
    SimulatedExchange  broker(cfg.initial_capital, fees);

    std::cout << "[+] Connecting to market feed: " << QuantConsoleReporter::CYAN << cfg.data_path << QuantConsoleReporter::RESET << " ...\n";
    CsvReplayFeed<4096> feed(cfg.data_path, cfg.train_split_ratio);

    std::cout << "[+] Loaded " << QuantConsoleReporter::BOLD << feed.total_bars() << QuantConsoleReporter::RESET << " total ticks (Source: " << feed.source_name()
              << " | In-Sample Calibration: " << feed.warmup_size() << " | Out-of-Sample Test: " << feed.replay_size() << ").\n";

    KineticTradingSession<BINS, VOL_WINDOW> session(cfg, broker);
    session.warmup(feed.warmup_bars());
    session.run(feed);
}

template <std::size_t BINS>
void dispatch_vol_window(const EngineConfig& cfg)
{
    switch (cfg.volatility_window)
    {
        case 10:
            return run_simulation<BINS, 10>(cfg);
        case 14:
            return run_simulation<BINS, 14>(cfg);
        case 20:
            return run_simulation<BINS, 20>(cfg);
        case 30:
            return run_simulation<BINS, 30>(cfg);
        case 50:
            return run_simulation<BINS, 50>(cfg);
        case 60:
            return run_simulation<BINS, 60>(cfg);
        default:
            throw std::runtime_error("Unsupported FEATURE_VOLATILITY_WINDOW.");
    }
}

void dispatch_simulation(const EngineConfig& cfg)
{
    switch (cfg.bins)
    {
        case 8:
            return dispatch_vol_window<8>(cfg);
        case 12:
            return dispatch_vol_window<12>(cfg);
        case 16:
            return dispatch_vol_window<16>(cfg);
        case 24:
            return dispatch_vol_window<24>(cfg);
        case 32:
            return dispatch_vol_window<32>(cfg);
        case 64:
            return dispatch_vol_window<64>(cfg);
        default:
            throw std::runtime_error("Unsupported MODEL_BINS.");
    }
}

int main()
{
    try
    {
        QuantConsoleReporter::render_header("LIVE API BACKTEST & PAPER TRADING SIMULATION ENGINE");
        const EngineConfig cfg = ConfigLoader::load("quant.env");
        dispatch_simulation(cfg);
    }
    catch (const std::exception& e)
    {
        std::cerr << QuantConsoleReporter::RED << "Fatal Simulation Error: " << e.what() << QuantConsoleReporter::RESET << "\n";
        return 1;
    }

    return 0;
}
