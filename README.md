# Kinetic Quant Trading Framework

![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)
![Build](https://img.shields.io/badge/Build-Bazel-green.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

A modular C++20 quantitative research, signal analysis, and paper-trading simulation framework. The engine discretizes log-returns onto a bounded state grid, evolves forward probability densities using Markov transition kernels ($f_{t+1} = W \cdot f_t$), and gates order execution through a direction-aware pre-trade risk filter.

---

## Architecture Overview

To keep the bar-processing loop deterministic and avoid per-tick heap allocations, data ingestion is decoupled from strategy evaluation via a lock-free Single-Producer Single-Consumer (SPSC) ring buffer (`LiveRingBufferFeed<4096>`) transporting trivially copyable `Tick` structs (`FixedTimestamp` + OHLCV fields).

```
                     [ CSV MARKET DATA (Producer jthread) ]
                                       │
                                       ▼  (Lock-Free SPSC Ring Buffer)
                   ┌───────────────────────────────────────┐
                   │       LiveRingBufferFeed<4096>        │
                   └───────────────────┬───────────────────┘
                                       │
                                       ▼
                   ┌───────────────────────────────────────┐
                   │       KineticTradingSession           │
                   │  1. Bar Open : Gap recheck & fills    │
                   │  2. Intra-Bar: SL / TP bracket check  │
                   │  3. Bar Close: Features & forecast    │
                   └───────────────────┬───────────────────┘
                                       │
                  ┌────────────────────┴────────────────────┐
                  ▼                                         ▼
  ┌───────────────────────────────┐         ┌───────────────────────────────┐
  │ HighPerformanceKineticEngine  │         │     WalkForwardCalibrator     │
  │ - Stack-allocated FlatMatrix  │◄────────┤ - Rolling feature window      │
  │ - Multi-step PDF rollout      │ Atomic  │ - Double-buffered W(i -> j)   │
  │ - VaR, CVaR, Shannon entropy  │  Swap   │ - Rebuilds every N bars       │
  └───────────────┬───────────────┘         └───────────────────────────────┘
                  │
                  ▼
  ┌───────────────────────────────┐
  │          RiskEngine           │
  │ - Direction-aware VaR / CVaR  │
  │ - Max Shannon entropy cap     │
  │ - Alpha hurdle & sizing       │
  └───────────────┬───────────────┘
                  │
                  ▼
  ┌───────────────────────────────┐
  │      IExecutionGateway        │
  │  (SimulatedExchange + Fees)   │
  └───────────────────────────────┘
```

---

## Key Capabilities

### 1. Compile-Time Template Dispatch (`EngineDispatcher`)
Runtime settings in `quant.env` map cleanly to `constexpr`-sized stack structures without virtual dispatch in the hot loop. Supported compile-time configurations:
- **Transition Kernels (`MODEL_KERNEL`):** `empirical`, `gaussian`, `dirichlet`, `phase_space`
- **State Grid Bins (`MODEL_BINS`):** `8, 12, 16, 24, 32, 64`
- **Volatility Windows (`FEATURE_VOLATILITY_WINDOW`):** `10, 14, 20, 30, 50, 60`

### 2. Double-Buffered Walk-Forward Calibration
- `HighPerformanceKineticEngine<Bins>` stores transition kernels in 64-byte aligned contiguous `FlatMatrix<double, Bins>` arrays and reads the active matrix pointer via `std::memory_order_acquire`.
- `WalkForwardCalibrator<Bins, KernelT>` maintains a trailing window of `DerivedFeatures`, alternates writes between two pre-allocated `FlatMatrix` buffers every `MODEL_RECALIBRATION_BARS` bars, and publishes updates atomically.

### 3. Lookahead-Free Simulation & Execution (`SimulatedExchange`)
- **Bar-Open Execution:** Signals generated at bar $t$ close are queued and executed at bar $t+1$ open. When `RISK_RECHECK_AT_OPEN=true`, the engine subtracts the overnight/inter-bar gap $\ln(O_{t+1} / C_t)$ from the forecast and re-runs risk checks before filling.
- **Intra-Bar Brackets:** Evaluates optional protective stop-loss (`SIM_STOP_LOSS_PCT`) and take-profit (`SIM_TAKE_PROFIT_PCT`) levels against bar high/low ranges.
- **Cost & Metric Accounting:** Models slippage (`SIM_SLIPPAGE_BPS`), percentage commissions (`SIM_COMMISSION_RATE`), flat per-trade fees (`SIM_FIXED_FEE`), long/short reversals, max drawdown, and $O(1)$ online Welford Sharpe ratio estimation.

### 4. Direction-Aware Pre-Trade Risk Management (`RiskEngine`)
Every candidate entry, reversal, and open position is evaluated against:
- **Asymmetric Tail Risk:** Checks 95% VaR and CVaR against the left tail for long exposure and the right tail for short exposure (with optional `RISK_SHORT_MAX_VAR_95` / `RISK_SHORT_MAX_CVAR_95` overrides), scaled by post-trade portfolio exposure.
- **Uncertainty Ceiling:** Vetoes entries and flattens open positions if forecast Shannon entropy exceeds `RISK_MAX_ENTROPY`.
- **Alpha & Hold Controls:** Enforces expected return hurdles (`RISK_MIN_EXPECTED_RETURN`), alpha-fade exits (`RISK_EXIT_EXPECTED_RETURN`), directional probability bounds, minimum/maximum hold durations, and dynamic exposure scaling (`RISK_DYNAMIC_SIZING`).

---

## Repository Structure

```
.
├── BUILD.bazel                         # Bazel targets for sim_runner & signal_inspector
├── MODULE.bazel                        # Bazel module dependencies
├── quant.env                           # Runtime model, risk, and simulation configuration
├── sim_runner.cpp                      # Paper-trading & backtest simulation executable
├── signal_inspector.cpp                # Diagnostic signal & distribution inspection tool
├── include/quant/
│   ├── core/
│   │   ├── config.hpp                  # .env parser and parameter validator
│   │   ├── dispatch_table.hpp          # Supported kernel, bin, and window definitions
│   │   └── engine_dispatcher.hpp       # Compile-time template dispatcher
│   ├── session/
│   │   ├── session_observer.hpp        # Observer interface (ISessionObserver / NullSessionObserver)
│   │   └── trading_session.hpp         # Bar-lifecycle orchestration (open, intra-bar, close)
│   └── telemetry/
│       └── console_reporter.hpp        # ANSI console dashboard, trade log, and ASCII density plots
├── kinetic_model/include/quant/        # Core math, types, online features, kernels, & calibrator
├── market_simulator/include/quant/     # CSV reader & lock-free SPSC ring buffer market feed
└── execution_simulator/include/quant/  # IExecutionGateway interface & SimulatedExchange
```

---

## Building & Running

### Prerequisites
- **Compiler:** GCC 11+, Clang 13+, or MSVC with C++20 support
- **Build System:** Bazel 6.0+

### Build All Targets
```bash
bazel build //... -c opt
```

### Run the Paper-Trading Simulation (`sim_runner`)
Loads `quant.env`, calibrates on the initial `TRAIN_SPLIT_RATIO` slice, replays out-of-sample bars through `SimulatedExchange`, and prints trade fills and P&L statistics:
```bash
bazel run //:sim_runner -c opt
```

### Run the Signal & Risk Inspector (`signal_inspector`)
Replays out-of-sample bars without an exchange broker to inspect transition matrix weights, signal decisions, risk veto counts, and terminal state-density distributions:
```bash
bazel run //:signal_inspector -c opt
```

---

## Configuration (`quant.env`)

All simulation parameters are configured in `quant.env`:

| Section | Key Parameters | Purpose |
| :--- | :--- | :--- |
| **Ingestion & Calibration** | `DATA_PATH`, `TRAIN_SPLIT_RATIO`, `MODEL_RECALIBRATION_BARS` | Dataset path, in-sample split, and walk-forward rebuild cadence |
| **Kinetic Model** | `MODEL_KERNEL`, `MODEL_BINS`, `MODEL_X_MAX`, `MODEL_FORECAST_HORIZON`, `FEATURE_VOLATILITY_WINDOW` | Kernel estimator, grid resolution, return bounds, horizon, and vol lookback |
| **Alpha Hurdles** | `RISK_MIN_EXPECTED_RETURN`, `RISK_EXIT_EXPECTED_RETURN`, `RISK_MIN_BULLISH_PROB`, `RISK_MAX_BEARISH_PROB` | Entry/exit expected return floors and directional probability cutoffs |
| **Tail Risk & Entropy** | `RISK_MAX_VAR_95`, `RISK_MAX_CVAR_95`, `RISK_SHORT_MAX_*`, `RISK_MAX_ENTROPY` | Exposure-scaled 95% VaR/CVaR caps and maximum forecast uncertainty (bits) |
| **Execution Lifecycle** | `RISK_MIN_HOLD_BARS`, `RISK_MAX_HOLD_BARS`, `RISK_RECHECK_AT_OPEN`, `RISK_DYNAMIC_SIZING` | Hold constraints, open-gap revalidation, and dynamic position sizing |
| **Simulated Exchange** | `SIM_INITIAL_CAPITAL`, `SIM_ALLOCATION_PCT`, `SIM_STOP_LOSS_PCT`, `SIM_TAKE_PROFIT_PCT`, `SIM_COMMISSION_RATE`, `SIM_SLIPPAGE_BPS`, `SIM_BARS_PER_YEAR` | Starting capital, allocation, intra-bar brackets, transaction costs, and Sharpe annualization |

---

## Component Summary

* **`FlatMatrix<T, Bins>`:** 64-byte aligned inline storage (`alignas(64) std::array<T, Bins * Bins>`) for cache-friendly dense matrix-vector propagation.
* **`LiveRingBufferFeed<Capacity>`:** Lock-free SPSC circular buffer transporting trivially copyable `Tick` values between the feed producer thread and the session consumer loop.
* **`HighPerformanceKineticEngine<Bins>`:** Evolves discrete return probability densities across $H$ steps and computes expected return, volatility, confidence intervals, direction-aware VaR/CVaR, and Shannon entropy.
* **`RiskEngine`:** Evaluates candidate and active positions against portfolio tail-risk limits ($\text{VaR}(95\%) \le \text{MaxVaR}$ and $\text{CVaR}(95\%) \le \text{MaxCVaR}$), entropy ceilings ($H \le H_{\max}$ via `RISK_MAX_ENTROPY`), and expected-return hurdles.

---

## License
This project is licensed under the MIT License.
