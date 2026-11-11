# High-Performance Kinetic Quant Trading Framework

![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)
![Build](https://img.shields.io/badge/Build-Bazel-green.svg)
![Architecture](https://img.shields.io/badge/Architecture-Dual--Speed_Hot%2FCold-orange.svg)
![Allocations](https://img.shields.io/badge/Memory-Zero--Heap_Hot_Path-brightgreen.svg)

A production-grade, low-latency quantitative market prediction and execution framework written in C++20. The system leverages state-space Fokker-Planck kinetic equations for short-term probability density evolution, featuring strict dual-speed separation between high-frequency execution (hot path) and asynchronous matrix calibration (warm/cold path).

---

## 🏛 Architecture Overview

Traditional trading pipelines often suffer from heap jitter, dynamic memory allocations (`std::vector::resize`, `std::function` callbacks), and CPU cache misses when running complex matrix operations alongside tick ingestion.

This framework enforces a **Dual-Speed Separation Architecture**:
- **Hot Path (Real-Time Execution):** $O(N^2)$ dense matrix-vector probability density propagation ($f_{t+1} = W \cdot f_t$) executing on stack-allocated, SIMD-aligned structures with zero dynamic heap allocations.
- **Warm Path (Async Calibration):** Background worker thread that periodically recalibrates dynamic transition kernels ($W_{i \to j}$) and updates the active hot path matrix via release-acquire atomic pointer swaps.
- **Cold Path (Ingestion Engine):** Historical CSV reader and batch feature extraction.

```
                      [ LIVE FEEDS / CSV INGESTION ]
                                    │
                                    ▼  (Lock-Free SPSC Ring Buffer)
                   ┌──────────────────────────────────┐
                   │    Data Layer (Ingestion Engine) │
                   └────────────────┬─────────────────┘
                                    │
                                    ▼  (Zero-Allocation Hot Path)
                   ┌──────────────────────────────────┐
                   │   Online Feature Engine (O(1))   │
                   └────────────────┬─────────────────┘
                                    │
               ┌────────────────────┴────────────────────┐
               ▼                                         ▼
  ┌──────────────────────────────┐          ┌───────────────────────────┐
  │   Model Inference (Hot Path) │          │  Calibration Engine       │
  │   - Step PDF propagation     │          │  (Async Background Thread)│
  │   - O(N^2) dense matrix-vec  │          │  - Calibrate gamma, sigma │
  │   - Double-buffered Kernel   │◄─────────┤  - Rebuild W(i->j) Kernel │
  └────────────┬─────────────────┘  Update  └───────────────────────────┘
               │                    Kernel
               ▼
  ┌──────────────────────────────┐
  │ Pre-Trade Risk & Signals     │
  │ - VaR / CVaR / Entropy Filter│
  │ - Order Sizing & Signal Output│
  └────────────┬─────────────────┘
               │
               ▼
    [ ORDER EXECUTION / TELEMETRY ]
```

---

## Key Architectural Pillars

### 1. Dual-Speed Double-Buffered Kernel Pipeline
- **Hot Path:** Evaluates forward probability densities using a contiguous 64-byte aligned `FlatMatrix<double, Bins>`. Matrix multiplications are stack-bound and SIMD-friendly (AVX2/AVX-512).
- **Warm Path:** Rebuilds transition kernels asynchronously every $K$ ticks or $M$ seconds using empirical histograms or Gaussian parameterizations. Kernel updates are swapped atomically using `std::atomic<const Matrix*>`.

### 2. Zero-Allocation Feature & Ring Buffer Pipeline
- Lock-free **Single-Producer Single-Consumer (SPSC)** circular ring buffer for tick ingestion.
- Eliminates heap-allocated `std::function` callbacks and unbounded containers in favor of fixed-capacity ring buffers and stack-allocated Welford statistics.

### 3. Integrated Pre-Trade Risk Management
- Real-time pre-trade risk filters evaluating Value-at-Risk (95% VaR), Expected Shortfall (CVaR), and Kinetic Entropy (predictive uncertainty).
- Automatic trade vetoes on high-uncertainty regime shifts or risk limit breaches.

### 4. Rich Telemetry & ANSI Console Formatting
- Asynchronous console reporting with full ANSI color coding, dynamic risk-event tracking, and live ASCII state-density visualization.

---

## 📁 Repository Structure

```
.
├── BUILD.bazel                   # Main binary target configuration
├── MODULE.bazel                  # Bazel module definitions
├── quant.cpp                     # Application entry point & pipeline runtime
└── include/
    └── quant/
        ├── BUILD.bazel           # C++ library dependency targets
        ├── core/                 # Core low-level abstractions & primitives
        │   ├── matrix.hpp        # 64-byte aligned stack matrix (FlatMatrix)
        │   ├── ring_buffer.hpp   # Lock-free SPSC circular queue
        │   └── types.hpp         # POD structures (Tick, DerivedFeatures)
        ├── data/                 # Ingestion & data loading
        │   └── csv_reader.hpp    # Historical CSV market batch reader
        ├── features/             # Incremental feature engineering
        │   ├── feature_engine.hpp# Hot path online feature processor
        │   └── online_stats.hpp  # Zero-allocation rolling variance & stats
        ├── models/               # Kinetic Fokker-Planck transition models
        │   ├── empirical_kernel.hpp # Empirical & Gaussian kernel algorithms
        │   ├── interfaces.hpp    # Model interfaces & QuantForecast struct
        │   └── kinetic_model.hpp # High-performance kinetic evolution engine
        ├── risk/                 # Pre-trade risk manager
        │   └── risk_manager.hpp  # VaR, CVaR, & Entropy filter decision maker
        └── telemetry/            # Diagnostic & visualization outputs
            └── console_reporter.hpp # ANSI terminal dashboard & ASCII plots
```

---

## 🚀 Building & Running

### Prerequisites
- **C++ Compiler:** GCC 11+, Clang 13+, or MSVC supporting `-std=c++20`
- **Build System:** [Bazel](https://bazel.build/) 6.0+

### Build the Project
Compile the framework with full SIMD vectorization and release optimizations:

```bash
bazel build //:quant -c opt
```

### Run the Engine
Execute the live pipeline binary:

```bash
bazel run //:quant -c opt
```

---

## 📊 Sample Output Telemetry

Upon running, the engine displays historical ingestion metrics, initial dynamic transition matrices, live execution signals with risk filter evaluations, and final state-density distributions:

```
========================================================================================================
  LIVE SIGNAL ANALYSIS & PRE-TRADE RISK STREAM
========================================================================================================
[15:31:00] [SIGNAL: BUY]  Price: $442.1500 | E[r]: +0.0125% | P(r>0): 58.30% | VaR(95): 0.4200%
  └─► Reason: BUY PASSED: E[r] (+0.0125%) > Hurdle & P(r>0) (58.30%) >= 52.00%
[15:32:00] [SIGNAL: HOLD] Price: $442.1000 | E[r]: -0.0010% | P(r>0): 49.10% | VaR(95): 0.3800%
  └─► Reason: HOLD: Alpha expectation neutral (E[r]=-0.0010%)
[15:33:00] [SIGNAL: HOLD] Price: $441.9000 | E[r]: -0.0150% | P(r>0): 42.10% | VaR(95): 5.2000%
  └─► Reason: VETO: VaR(95%) 5.2000% exceeds max limit 5.0000%

========================================================================================================
  PRE-TRADE RISK & EXECUTION TELEMETRY SUMMARY
========================================================================================================
  Total Hot Path Ticks Evaluated : 390
  Buy Signals Executed [BUY]    : 42
  Sell Signals Executed [SELL]  : 18
  Hold Decisions [HOLD]         : 330
  Risk Limits Vetoed Trades     : 5
========================================================================================================
```

---

## 🔧 Component Overview

### `FlatMatrix<T, Bins>`
Provides aligned inline storage (`alignas(64) std::array<T, Bins * Bins>`) ensuring matrix data fits seamlessly into L1 CPU cache lines without pointer indirection.

### `SPSCRingBuffer<T, Capacity>`
A single-producer single-consumer queue utilizing cache-line padding (`alignas(64)`) to eliminate false sharing between enqueue and dequeue operations.

### `HighPerformanceKineticEngine<Bins>`
Performs zero-allocation state vector propagation using matrix multiplication. The active kernel matrix pointer is fetched on each step using `std::memory_order_acquire`.

### `RiskEngine`
Monitors portfolio parameters and mathematical forecasts. Every candidate trade must pass:
1. $\text{VaR}_{95} \le \text{MaxVaRThreshold}$
2. $H(\text{distribution}) \ge \text{MinEntropyFloor}$
3. Alpha hurdle expectations ($E[r]$ and directional probability bounds).

---

## 📄 License
This project is open-source software licensed under the MIT License.
