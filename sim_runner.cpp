#include "quant/core/config.hpp"
#include "quant/core/matrix.hpp"
#include "quant/core/types.hpp"
#include "quant/execution/execution_gateway.hpp"
#include "quant/execution/simulated_exchange.hpp"
#include "quant/features/feature_engine.hpp"
#include "quant/market/csv_reader.hpp"
#include "quant/market/market_feed.hpp"
#include "quant/models/kernels.hpp"
#include "quant/models/kinetic_model.hpp"
#include "quant/risk/risk_manager.hpp"
#include "quant/telemetry/console_reporter.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

template <std::size_t BINS, std::size_t VOL_WINDOW>
void run_simulation(const EngineConfig& cfg)
{
    using Matrix        = FlatMatrix<double, BINS>;
    using KineticEngine = HighPerformanceKineticEngine<BINS>;

    // 1. Initialize Execution Gateway (swap SimulatedExchange for a live broker adapter later)
    const FeeStructure fees{
            .commission_rate     = cfg.commission_rate,
            .fixed_fee_per_trade = cfg.fixed_fee,
            .slippage_bps        = cfg.slippage_bps,
    };
    const double                       initial_cash = cfg.initial_capital;
    std::unique_ptr<IExecutionGateway> broker       = std::make_unique<SimulatedExchange>(initial_cash, fees);

    // 2. Initialize Market Data Feed (swap CsvMarketFeed for LiveRingBufferFeed<4096> later)
    std::cout << "[+] Connecting to market feed: " << QuantConsoleReporter::CYAN << cfg.data_path << QuantConsoleReporter::RESET << " ...\n";
    // Load CSV dataset and split warmup vs. out-of-sample replay[cite: 1]
    const MarketDataset dataset    = CsvReader::load(cfg.data_path);
    const std::size_t   total_bars = dataset.size();
    if (total_bars < 10)
    {
        throw std::runtime_error("Insufficient market bars loaded for simulation.");
    }

    const std::size_t train_size  = std::max<std::size_t>(2, static_cast<std::size_t>(static_cast<double>(total_bars) * cfg.train_split_ratio));
    const std::size_t replay_size = total_bars - train_size;
    const auto        warmup_bars = dataset.bars().subspan(0, train_size);
    const auto        oos_bars    = dataset.bars().subspan(train_size);

    auto ring_feed = std::make_unique<LiveRingBufferFeed<4096>>(dataset[0].source);

    // Producer thread: simulates a live broker pushing ticks into the SPSC ring buffer.
    // Later, swap this thread with your real broker API callback calling ring_feed->push_tick(tick).
    std::jthread csv_producer(
            [feed = ring_feed.get(), oos_bars]()
            {
                for (const auto& bar : oos_bars)
                {
                    feed->push_blocking(bar);
                }
                feed->mark_exhausted();
            });

    IMarketDataFeed&             market_feed = *ring_feed;
    std::vector<DerivedFeatures> train_features;
    train_features.reserve(warmup_bars.size());

    OnlineFeatureEngine<VOL_WINDOW> feature_engine;
    DerivedFeatures                 feat{};

    for (const auto& bar : warmup_bars)
    {
        if (feature_engine.process(bar, feat))
        {
            train_features.push_back(feat);
        }
    }

    std::cout << "[+] Loaded " << QuantConsoleReporter::BOLD << total_bars << QuantConsoleReporter::RESET << " total ticks (Source: " << market_feed.source_name()
              << " | In-Sample Calibration: " << train_size << " | Out-of-Sample Test: " << replay_size << ").\n";

    KineticEngine engine(cfg.x_max);
    auto          kernel = std::make_unique<Matrix>();

    EmpiricalKernel<BINS> kernel_builder;
    kernel_builder.build_kernel(engine.dx(), engine.x_max(), train_features, *kernel);
    engine.update_kernel(kernel.get());

    RiskEngine                          risk_engine(cfg.risk_limits);
    typename KineticEngine::StateVector current_distribution = engine.initial_uniform_distribution();
    typename KineticEngine::StateVector next_distribution{};

    QuantConsoleReporter::render_header("ORDER GATEWAY & TRADE EXECUTION STREAM");

    std::vector<DerivedFeatures> rolling_features   = train_features;
    const std::size_t            calibration_window = train_features.size();
    std::size_t                  bars_since_recal   = 0;

    struct PendingOrder
    {
        OrderSignal                         signal{OrderSignal::Hold};
        double                              signal_close_price{0.0};
        double                              allocation_pct{0.0};
        typename KineticEngine::StateVector step1_dist{};
    };

    std::optional<PendingOrder> pending_order;
    bool                        pending_flatten{false};
    std::size_t                 bars_held{0};

    auto log_close_fill = [](const char* label, const ExecutionFill& fill)
    {
        if (!fill.closed_position)
            return;

        const char* pnl_color = fill.realized_pnl >= 0.0 ? QuantConsoleReporter::GREEN : QuantConsoleReporter::RED;
        std::cout << QuantConsoleReporter::GRAY << "[" << fill.timestamp << "] " << QuantConsoleReporter::RESET << QuantConsoleReporter::YELLOW << QuantConsoleReporter::BOLD
                  << label << QuantConsoleReporter::RESET << " | Shares: " << fill.shares << " @ $" << std::fixed << std::setprecision(2) << fill.fill_price << " | Value: $"
                  << fill.trade_value << " | Fee: $" << fill.fee_paid << " | Realized PnL: " << pnl_color << QuantConsoleReporter::BOLD << "$" << fill.realized_pnl
                  << QuantConsoleReporter::RESET << "\n";
    };

    bool   has_first_oos_bar = false;
    double start_oos_price   = 0.0;
    Tick   last_tick{};
    Tick   tick{};

    // 4. Unified Event Loop (works identically for CSV replay or live ring-buffer tick stream)
    while (!market_feed.is_exhausted())
    {
        if (!market_feed.next(tick))
        {
            std::this_thread::yield();
            continue;
        }

        if (!has_first_oos_bar)
        {
            start_oos_price   = tick.close;
            has_first_oos_bar = true;
        }
        last_tick = tick;

        // Stage 1 (Bar Open): Execute protective flattens or validated open orders
        if (pending_flatten && broker->account().position_shares != 0.0)
        {
            const auto fill = broker->close_position(tick.datetime, tick.open);
            log_close_fill("RISK FLATTEN EXIT   ", fill);
            bars_held       = 0;
            pending_flatten = false;
            pending_order.reset();
        }
        else if (pending_order.has_value())
        {
            if (pending_order->signal == OrderSignal::Close)
            {
                if (broker->account().position_shares != 0.0)
                {
                    const auto fill = broker->close_position(tick.datetime, tick.open);
                    log_close_fill("EXECUTED EXIT/CLOSE ", fill);
                    bars_held = 0;
                }
            }
            else
            {
                bool   execute_at_open = true;
                double exec_allocation = pending_order->allocation_pct;

                if (cfg.recheck_at_open && pending_order->signal_close_price > 0.0 && tick.open > 0.0)
                {
                    const double        entry_gap     = std::log(tick.open / pending_order->signal_close_price);
                    const QuantForecast open_forecast = engine.analyze_and_predict_horizon(tick.open, pending_order->step1_dist, cfg.forecast_horizon, entry_gap);

                    const auto&          acc = broker->account();
                    const PortfolioState open_portfolio{
                            .equity                = acc.equity(tick.open),
                            .position_shares       = acc.position_shares,
                            .current_price         = tick.open,
                            .target_allocation_pct = cfg.allocation_pct,
                            .allow_pyramiding      = false,
                    };

                    const SignalDecision open_decision = risk_engine.evaluate(open_forecast, open_portfolio);
                    if (open_decision.signal != pending_order->signal)
                    {
                        execute_at_open = false;
                        if (open_decision.should_flatten && acc.position_shares != 0.0)
                        {
                            const auto fill = broker->close_position(tick.datetime, tick.open);
                            log_close_fill("RISK FLATTEN EXIT   ", fill);
                            bars_held = 0;
                        }
                    }
                    else if (cfg.dynamic_sizing)
                    {
                        exec_allocation = open_decision.max_allowed_allocation_pct;
                    }
                }

                if (execute_at_open)
                {
                    const double      shares_before = broker->account().position_shares;
                    const std::size_t fills_before  = broker->fills().size();
                    (void) broker->execute_order(pending_order->signal, tick.datetime, tick.open, exec_allocation);

                    for (std::size_t f_idx = fills_before; f_idx < broker->fills().size(); ++f_idx)
                    {
                        const auto& fill   = broker->fills()[f_idx];
                        const bool  is_buy = fill.side == OrderSignal::Buy;
                        const char* color  = is_buy ? QuantConsoleReporter::GREEN : QuantConsoleReporter::RED;

                        std::cout << QuantConsoleReporter::GRAY << "[" << fill.timestamp << "] " << QuantConsoleReporter::RESET << color << QuantConsoleReporter::BOLD
                                  << (is_buy ? "EXECUTED BUY/COVER " : "EXECUTED SELL/SHORT") << QuantConsoleReporter::RESET << " | Shares: " << fill.shares << " @ $" << std::fixed
                                  << std::setprecision(2) << fill.fill_price << " | Value: $" << fill.trade_value << " | Fee: $" << fill.fee_paid;

                        if (fill.closed_position)
                        {
                            const char* pnl_color = fill.realized_pnl >= 0.0 ? QuantConsoleReporter::GREEN : QuantConsoleReporter::RED;
                            std::cout << " | Realized PnL: " << pnl_color << QuantConsoleReporter::BOLD << "$" << fill.realized_pnl << QuantConsoleReporter::RESET;
                        }
                        std::cout << "\n";
                    }

                    const double shares_after = broker->account().position_shares;
                    if (shares_after == 0.0 || (shares_before <= 0.0 && shares_after > 0.0) || (shares_before >= 0.0 && shares_after < 0.0))
                    {
                        bars_held = 0;
                    }
                }
            }

            pending_order.reset();
            pending_flatten = false;
        }

        // Stage 2 (Intra-Bar & Mark-to-Market): Venue bracket evaluation & equity update
        if (broker->account().position_shares != 0.0)
        {
            ++bars_held;
            const auto bracket_fill = broker->on_bar_update(tick, cfg.stop_loss_pct, cfg.take_profit_pct);
            if (bracket_fill.closed_position)
            {
                log_close_fill("INTRA-BAR BRACKET   ", bracket_fill);
                bars_held = 0;
            }
        }
        else
        {
            bars_held = 0;
            (void) broker->on_bar_update(tick, 0.0, 0.0);
        }

        // Stage 3 (Bar Close): Causal walk-forward update & signal generation
        if (feature_engine.process(tick, feat))
        {
            const std::size_t state_idx = engine.state_index(feat.log_return);

            current_distribution = engine.state_delta(state_idx);
            engine.evolve(current_distribution, next_distribution);

            const auto&          acc = broker->account();
            const PortfolioState portfolio{
                    .equity                = acc.equity(tick.close),
                    .position_shares       = acc.position_shares,
                    .current_price         = tick.close,
                    .target_allocation_pct = cfg.allocation_pct,
                    .allow_pyramiding      = false,
            };

            const QuantForecast  forecast = engine.analyze_and_predict_horizon(tick.close, next_distribution, cfg.forecast_horizon, 0.0);
            const SignalDecision decision = risk_engine.evaluate(forecast, portfolio);

            const bool is_open_position  = (acc.position_shares != 0.0);
            const bool inside_min_hold   = is_open_position && (bars_held < cfg.min_hold_bars);
            const bool reached_time_stop = is_open_position && (cfg.max_hold_bars > 0) && (bars_held >= cfg.max_hold_bars);

            if (decision.should_flatten)
            {
                pending_flatten = true;
                pending_order.reset();
            }
            else if (inside_min_hold)
            {
                pending_order.reset();
            }
            else if (decision.signal == OrderSignal::Buy || decision.signal == OrderSignal::Sell)
            {
                const double target_alloc = cfg.dynamic_sizing ? decision.max_allowed_allocation_pct : cfg.allocation_pct;
                pending_order             = PendingOrder{
                        .signal             = decision.signal,
                        .signal_close_price = tick.close,
                        .allocation_pct     = target_alloc,
                        .step1_dist         = next_distribution,
                };
                pending_flatten = false;
            }
            else if (is_open_position && (decision.signal == OrderSignal::Close || reached_time_stop))
            {
                pending_order = PendingOrder{
                        .signal             = OrderSignal::Close,
                        .signal_close_price = tick.close,
                        .allocation_pct     = 0.0,
                        .step1_dist         = next_distribution,
                };
                pending_flatten = false;
            }

            // Append completed bar to rolling window and periodically recalibrate kernel (strictly causal)
            rolling_features.push_back(feat);
            if (++bars_since_recal >= 390 && rolling_features.size() >= calibration_window)
            {
                const std::span<const DerivedFeatures> trailing_slice(rolling_features.data() + (rolling_features.size() - calibration_window), calibration_window);
                kernel_builder.build_kernel(engine.dx(), engine.x_max(), trailing_slice, *kernel);
                engine.update_kernel(kernel.get());
                bars_since_recal = 0;
            }
        }
    }

    // Liquidate any remaining open position at the final known close for accounting.
    if (has_first_oos_bar && broker->account().position_shares != 0.0)
    {
        const auto fill = broker->close_position(last_tick.datetime, last_tick.close);
        log_close_fill("END-OF-DATA FLATTEN ", fill);
        (void) broker->on_bar_update(last_tick, 0.0, 0.0);
    }

    // Output Portfolio PnL & Performance Metrics
    const auto&  acc           = broker->account();
    const double final_price   = last_tick.close;
    const double buy_hold_pct  = (start_oos_price > 0.0) ? (((final_price / start_oos_price) - 1.0) * 100.0) : 0.0;
    const double final_equity  = acc.equity(final_price);
    const double total_pnl     = final_equity - initial_cash;
    const double return_pct    = (total_pnl / initial_cash) * 100.0;
    const double sharpe        = broker->sharpe_ratio();
    const double win_rate      = (acc.total_trades > 0) ? (static_cast<double>(acc.winning_trades) / acc.total_trades * 100.0) : 0.0;
    const double profit_factor = acc.profit_factor();
    const double payoff_ratio  = acc.payoff_ratio();
    const double expectancy    = acc.expectancy_per_trade();

    QuantConsoleReporter::render_header("SIMULATED ACCOUNT PERFORMANCE & P&L STATEMENT");
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Initial Starting Capital     : $" << initial_cash << "\n";
    std::cout << "  Final Account Equity         : $" << final_equity << "\n";
    std::cout << "  Net Realized Profit / Loss   : " << (total_pnl >= 0 ? QuantConsoleReporter::GREEN : QuantConsoleReporter::RED) << QuantConsoleReporter::BOLD << "$" << total_pnl
              << " (" << return_pct << "%)" << QuantConsoleReporter::RESET << "\n";
    std::cout << "  Out-of-Sample Buy & Hold     : " << (buy_hold_pct >= 0 ? QuantConsoleReporter::GREEN : QuantConsoleReporter::RED) << buy_hold_pct << "%"
              << QuantConsoleReporter::RESET << " (Benchmark)\n";
    std::cout << "  Total Exchange Fees Paid     : " << QuantConsoleReporter::YELLOW << "$" << acc.total_fees_paid << QuantConsoleReporter::RESET << " (Slippage & Commission)\n";
    std::cout << "  Max Equity Drawdown          : " << QuantConsoleReporter::RED << (acc.max_drawdown * 100.0) << "%" << QuantConsoleReporter::RESET << "\n";
    std::cout << "  Completed Round-Trip Trades  : " << acc.total_trades << " (Wins: " << acc.winning_trades << " | Losses: " << acc.losing_trades << ")\n";
    std::cout << "  Strategy Win Rate            : " << (win_rate >= 50.0 ? QuantConsoleReporter::GREEN : QuantConsoleReporter::YELLOW) << win_rate << "%"
              << QuantConsoleReporter::RESET << "\n";
    std::cout << "  Avg Win / Avg Loss (Payoff)  : " << QuantConsoleReporter::GREEN << "$" << acc.avg_win() << QuantConsoleReporter::RESET << " / " << QuantConsoleReporter::RED
              << "$" << acc.avg_loss() << QuantConsoleReporter::RESET << " (" << std::setprecision(2) << payoff_ratio << ":1)\n";
    std::cout << "  Profit Factor (Gross W/L)    : " << (profit_factor >= 1.2 ? QuantConsoleReporter::GREEN : QuantConsoleReporter::YELLOW) << std::setprecision(2) << profit_factor
              << QuantConsoleReporter::RESET << "\n";
    std::cout << "  Net Expectancy per Trade     : " << (expectancy >= 0.0 ? QuantConsoleReporter::GREEN : QuantConsoleReporter::RED) << "$" << std::setprecision(2) << expectancy
              << QuantConsoleReporter::RESET << "\n";
    std::cout << "  Annualized Sharpe Ratio      : " << (sharpe >= 1.0 ? QuantConsoleReporter::GREEN : QuantConsoleReporter::YELLOW) << std::setprecision(3) << sharpe
              << QuantConsoleReporter::RESET << "\n";
    std::cout << QuantConsoleReporter::CYAN << "========================================================================================================\n"
              << QuantConsoleReporter::RESET;
}

template <std::size_t BINS>
void dispatch_vol_window(const EngineConfig& cfg)
{
    switch (cfg.volatility_window)
    {
        case 10:
            run_simulation<BINS, 10>(cfg);
            break;
        case 14:
            run_simulation<BINS, 14>(cfg);
            break;
        case 20:
            run_simulation<BINS, 20>(cfg);
            break;
        case 30:
            run_simulation<BINS, 30>(cfg);
            break;
        case 50:
            run_simulation<BINS, 50>(cfg);
            break;
        case 60:
            run_simulation<BINS, 60>(cfg);
            break;
        default:
            throw std::runtime_error("Unsupported FEATURE_VOLATILITY_WINDOW.");
    }
}

void dispatch_simulation(const EngineConfig& cfg)
{
    switch (cfg.bins)
    {
        case 8:
            dispatch_vol_window<8>(cfg);
            break;
        case 12:
            dispatch_vol_window<12>(cfg);
            break;
        case 16:
            dispatch_vol_window<16>(cfg);
            break;
        case 24:
            dispatch_vol_window<24>(cfg);
            break;
        case 32:
            dispatch_vol_window<32>(cfg);
            break;
        case 64:
            dispatch_vol_window<64>(cfg);
            break;
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
