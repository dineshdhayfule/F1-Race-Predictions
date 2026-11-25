<div align="center">

# ApexGrid

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Test Suite](https://img.shields.io/badge/tests-183%20passed%20(100%25)-brightgreen.svg)](tests/)
[![Architecture](https://img.shields.io/badge/architecture-Multi--Model%20Ensemble-orange.svg)](src/models/)
[![Simulation](https://img.shields.io/badge/Monte%20Carlo-25%2C000%20runs-purple.svg)](src/simulation/)

</div>

> **Predicting the Race Beyond the Grid.**  
> *An ML-powered motorsport intelligence, multi-model ensemble, and Monte Carlo simulation platform for Formula 1 Grand Prix outcome forecasting.*

---

## Overview

**ApexGrid** is a production-grade machine learning and stochastic simulation platform engineered for Formula 1 performance analytics and race outcome forecasting. Built to eliminate hardcoded circuit heuristics, ApexGrid combines multi-stage telemetry ingestion, non-linear tire degradation modeling, pit stop strategy simulation, environmental weather dynamics, safety car hazard forecasting, and multi-model ensembling into a unified predictive pipeline.

ApexGrid models the probabilistic distribution of finishing positions, win probabilities, podium chances, and points finishes across all 23 championship rounds (from Round 1 Australia through Round 23 Abu Dhabi, including Lusail / Qatar) without future-session data leakage.

---

## Key Features

- **Validated Multi-Model Ensemble Architecture:** Blends Random Forest (40%), XGBoost (30%), and L2-regularized Ridge Regression (30%) calibrated via chronological walk-forward validation to minimize out-of-sample error and prevent tree-based overfitting. Also supports a fast 2-way hybrid ensemble (60% RF / 40% XGB) and custom weight specifications.
- **Circuit-Aware Starting-Grid Blending:** Calibrates starting-grid position weighting dynamically according to circuit topology:
  - *Street circuits* (Monaco, Baku, Singapore): 50% Grid / 50% ML ($w_{\text{grid}} = 0.50$, $w_{\text{ml}} = 0.50$) for high positional conservatism.
  - *Permanent circuits* (Silverstone, Spa, Monza, Interlagos): 35% Grid / 65% ML ($w_{\text{grid}} = 0.35$, $w_{\text{ml}} = 0.65$).
  - *Temporary circuits* (Melbourne, Montreal): 45% Grid / 55% ML ($w_{\text{grid}} = 0.45$, $w_{\text{ml}} = 0.55$).
  - *Active Weekend-Only Mode:* When executing without prior-season historical form (`--weekend-only`), the pipeline uses the 25-feature telemetry matrix with the full Random Forest / XGBoost / Ridge ensemble and applies this exact circuit-aware grid blending seamlessly.
- **Strict Stage-Aware Temporal Boundaries:** Enforces chronological cutoffs preventing target and future-session data leakage across seven weekend stages: `pre_fp1`, `after_fp1`, `after_fp2`, `after_fp3`, `after_sprint_qualifying`, `after_sprint_race`, and `after_qualifying`.
- **Pre-Qualifying Provisional Forecasts & Fallback Blending:** Generates out-of-sample ML qualifying predictions (`provisional_grid_position`, rendered as `~P1`, `~P2`, etc.) when official qualifying hasn't occurred. If no starting grid data exists, grid blending is cleanly bypassed ($w_{\text{grid}} = 0.0, w_{\text{ml}} = 1.0$) rather than forcing an artificial constant midfield rank.
- **Dual-View Probability & ExpectedFinish Presentation:** 
  - *Most Likely Podium Banner:* Showcases the top 3 drivers ordered by `Win%` descending, `Podium%` descending, and `ExpectedFinish` ascending (tie-breaker).
  - *Full Field Table:* Ranks drivers strictly by DNF-adjusted `ExpectedFinish` (mean simulated finish across all Monte Carlo iterations).
- **25,000-Iteration Monte Carlo Engine:** Simulates 25,000 stochastic race outcomes per entrant using out-of-fold residual-calibrated pace variance ($\sigma$), empirical DNF hazard modeling with deterministic $+1000.0$ non-finisher penalties, field compression dynamics, and driver-specific wet skill differentials.
- **Five Specialized Physical & Strategic Subsystems:**
  - *Race Pace Intelligence:* Fuel-corrected stint pace extraction ($0.035\text{s}/\text{lap}$ burn-off) and clean-air pace modeling.
  - *Tire Degradation Modeling:* Non-linear exponential degradation wear curves ($\Delta t = \alpha \cdot L^\beta$) across Soft, Medium, Hard, Intermediate, and Wet compounds.
  - *Pit Stop Strategy Simulation:* Multi-stint optimization under track-specific pit-loss seconds and strict FIA Article 30.5 dry compound regulations.
  - *Race Control Hazard Modeling:* Venue-specific disruption modeling (Safety Car, VSC, and Red Flag probabilities).
  - *Weather Intelligence:* Live ambient and track temperature evaluation with stochastic rainfall scenario sampling.
- **Multi-Source Starting Grid Resolution:** Three-tier grid resolution pipeline supporting official FIA Starting Grid Manifests (`data/grids/`), OpenF1 live feeds, and programmatic FIA Article 42 penalty cascade re-alignment.
- **Safe Data Management & Boundary Guard:** Integrated cleanup utility with strict root directory containment, dry-run preview mode, and protected file preservation (`outputs/`, `data/grids/`, `config/`).
- **183-Test Verified Regression Suite:** Comprehensive automated test coverage (183 passing, 0 failing) across 21 test modules ensuring 100% pass rates with exit code 0.

---

## Architecture & Methodology

ApexGrid is engineered with rigorous mathematical and machine learning guarantees to ensure models reflect genuine predictive signal rather than subtle data leakage or overfitting.

### 1. Stage-Gated Starting-Grid Blending & Temporal Isolation
Starting grid positions heavily determine Sunday race outcomes, but using official qualifying data prior to qualifying introduces catastrophic lookahead bias. ApexGrid resolves this with stage-aware gating:
* **Post-Qualifying (`after_qualifying`, `before_race`):** Official FIA starting grid positions (`ActualStartingGrid`, incorporating grid penalties and pit-lane starts) are blended with the pure ML pace predictions using circuit-aware topology weights:
  $$\text{Score}_{\text{blended}} = w_{\text{ml}} \cdot \text{Score}_{\text{ML}} + w_{\text{grid}} \cdot \text{ActualStartingGrid}$$
* **Pre-Qualifying (`pre_fp1` through `after_fp3`):** `ActualStartingGrid` and qualifying times are strictly masked to `NaN`. The feature pipeline generates an out-of-sample provisional qualifying forecast (`provisional_grid_position`, displayed as `~P<num>` in the CLI). In `src/models/predict.py`:
  * If a valid `provisional_grid_position` exists, it blends with the ML pace score using circuit topology weights.
  * If no legitimate grid or provisional position exists, grid blending is cleanly disabled ($w_{\text{grid}} = 0.0, w_{\text{ml}} = 1.0$), ensuring predictions rely exclusively on practice telemetry, fuel-corrected long runs, and historical EWMA form.

### 2. Dual-View Outcome Presentation
Monte Carlo race simulations produce full finishing distributions rather than point predictions. The CLI presents two distinct, complementary views:
* **Most Likely Podium Banner:** Highlights the 3 drivers with the highest probability of winning or finishing on the podium. Selected via:
  1. `WinProbability` descending
  2. `Top3Probability` descending
  3. `ExpectedFinish` ascending (tie-breaker)
* **Top 10 / Full Field Table:** Ranks all drivers by `PredictedPosition` ($1 \dots N$), sorted strictly by `ExpectedFinish` ascending. Simulated DNF runs receive a $+1000.0$ penalty so non-finishers rank behind classified finishers. Drivers with high win upside but elevated DNF hazard (e.g. aggressive frontrunners) appear prominently in the podium banner while their mean expected finish reflects their full risk profile.

### 3. Methodology & Validation Highlights
* **Chronological Walk-Forward Validation:** Shuffled k-fold splits are strictly prohibited. Models evaluate via expanding-window walk-forward validation: for each validation round $r > \text{min\_train\_rounds}$, the models train exclusively on rounds $\{1 \dots r-1\}$ and test on out-of-sample round $r$. If any fold fails, the validation transitions to `"BLOCKED"` and all residual arrays are cleared to prevent partial residual contamination.
* **Decoupled Two-Stage Modeling:** ML regressors (Random Forest, XGBoost, Ridge) train exclusively on classified finishers (`PaceEligible == 1`, where race status is `"FINISHED"`). True on-track pace is kept uncorrupted by mechanical retirements or lap-1 incidents. DNF hazards are modeled separately via binomial draws in Monte Carlo.
* **Residual-Calibrated Pace Volatility ($\sigma$):** Monte Carlo simulation variance is empirically calibrated from walk-forward validation residuals using Median Absolute Deviation (MAD):
  $$\sigma = \max\left(1.5,\, 1.4826 \cdot \text{MAD}(\text{residuals})\right)$$
  Under wet-weather sampling, volatility scales by $1.35\times$ and DNF hazard scales by $1.40\times$.
* **Exponentially Weighted Form ($\alpha = 0.30$):** Driver and constructor rolling form metrics use an EWMA decay factor of $\alpha = 0.30$, heavily weighting the most recent 3–5 rounds to swiftly reflect mid-season aerodynamic upgrades while maintaining baseline stability.

---

## Tech Stack

| Component | Technologies |
| :--- | :--- |
| **Language & Runtime** | Python 3.10, 3.11, 3.12, 3.13 |
| **Machine Learning** | Scikit-Learn (Random Forest, Ridge Regression, Preprocessing), XGBoost (Gradient Boosted Trees) |
| **Scientific Computing** | NumPy, SciPy, Pandas |
| **Telemetry & Live Data** | FastF1, OpenF1 REST API |
| **CLI & Terminal UI** | Typer, Rich, ANSI Terminal Formatting |
| **Testing & Verification** | Pytest, Pytest-Cov |

---

## Getting Started

### Prerequisites

- **Python:** Version 3.10, 3.11, 3.12, or 3.13 installed.
- **Package Manager:** `pip` (v23.0+ recommended).
- **Version Control:** `git` installed.

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/dineshdhayfule/F1-Race-Predictions.git
   cd F1-Race-Predictions
   ```

2. **Create and activate a virtual environment:**
   - *Windows (PowerShell):*

     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```

   - *Linux / macOS:*

     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Configuration

ApexGrid uses JSON configuration files in the [`config/`](config/) directory:

- [`config/model_config.json`](config/model_config.json): Estimator hyperparameters, feature sets, circuit blending weights, model selection modes, default capabilities, and feature pruning toggles.
- [`config/tracks.json`](config/tracks.json): Canonical mappings for all 23 championship rounds (including Lusail / Qatar), track lengths, circuit types, and championship calendars.
- [`config/strategy_config.json`](config/strategy_config.json): Compound parameters, pit lane loss times, and 15 track-specific circuit priors.
- [`config/settings.json`](config/settings.json): Cache locations, logging thresholds, and output directories.

---

## Usage Examples

### 1. Interactive Guided Menu (Recommended)

Launch the interactive terminal interface:

```bash
python main.py
# or using the Typer entrypoint:
apexgrid
```

Or access historical statistics and analytics directly:

```bash
python main.py history
# or:
apexgrid history
```

The menu provides eight guided workflows organized into three logical operational groups:

#### Core Pipeline
1. **Predict a Race:** Interactive wizard for circuit selection, weekend stage, model selection, and forecast execution.
2. **Load / Collect Data:** Ingest FastF1 practice telemetry or full-season championship rounds.
3. **Evaluate a Prediction:** Score forecast accuracy against official race classifications.

#### Analytics & History
4. **View Previous Predictions & Reports:** Inspect generated prediction tables and Markdown race reports.
5. **Historical Statistics & Analytics (1950–2026):** 11 modular motorsport intelligence views spanning session results (FP1–FP3, SQ, Sprint, Quali, Race), driver career metrics, teammate H2H, qualifying performance, race history, circuit records, season standings, constructor & engine dominance, all-time records & milestones, historical reliability & attrition, and regulatory era comparisons.

#### System & Admin
6. **System Health & Data Readiness:** Run diagnostic checks and data availability audits.
7. **Configuration & Supported Circuits:** View circuit parameters, track types, and model settings.
8. **Data Management & Cache Deletion:** Safely preview and prune disposable FastF1 cache and telemetry files.

---

### 2. Command Line Interface (CLI)

Both `python main.py <command>` and the installed `apexgrid <command>` CLI entrypoint support all operations:

#### Run Race Predictions

```bash
# Validated Multi-Model Ensemble (40% RF / 30% XGB / 30% Ridge)
python main.py predict --track sao_paulo --year 2026 --stage pre_fp1 --model ensemble

# Default 2-Way Hybrid Ensemble (60% RF / 40% XGB)
python main.py predict --track sepang --year 2026 --stage after_qualifying --model hybrid --top 10

# Pure Ridge Regression (100% Ridge)
python main.py predict --track monza --year 2026 --stage after_qualifying --model ridge

# Pure Random Forest (100% RF)
python main.py predict --track silverstone --year 2026 --stage after_fp2 --model rf

# Pure XGBoost (100% XGB)
python main.py predict --track spa --year 2026 --stage after_fp3 --model xgb

# Custom Hybrid Ensemble Weights (70% RF / 30% XGB)
python main.py predict --track baku --year 2026 --stage after_qualifying --model hybrid --rf-weight 70 --xgb-weight 30

# Pure Active Weekend Telemetry ML Ensemble (Zero prior season historical form)
python main.py predict --track las_vegas --year 2026 --stage after_qualifying --weekend-only

# Enable Enhanced Physics Capabilities (Race Pace, Tire Deg, Weather, Strategy)
python main.py predict --track monaco --year 2026 --stage after_qualifying --enhanced
```

#### Chronological Walk-Forward Model Evaluation

```bash
# Run walk-forward validation across completed rounds for an event
python main.py evaluate --track sepang --year 2026
```

#### Anti-Leakage & Dataset Validation

```bash
# Verify stage cutoff isolation and feature availability
python main.py validate --track baku --year 2026 --stage after_qualifying
```

#### Safe Data Management

```bash
# Preview disposable FastF1 cache without deleting
python main.py clean --target cache --preview

# Clean disposable cache files with force flag
python main.py clean --target cache --force
```

---

## Testing & Quality Assurance

ApexGrid maintains **183 automated tests** across 21 test modules, guaranteeing 100% pass rates:

```bash
# Run the entire test suite
pytest

# Run podium probability selection and provisional grid formatting tests
pytest tests/test_podium_and_grid_display.py

# Run remediation integrity tests (F-01, F-02, F-05, F-07, F-08 verification)
pytest tests/test_remediation_integrity.py

# Run chronological walk-forward validation tests
pytest tests/test_walk_forward.py

# Run dynamic roster and practice driver substitution tests
pytest tests/test_dynamic_roster.py

# Run stage isolation, core features, and anti-leakage tests
pytest tests/test_core_features.py

# Run historical intelligence expansion and weekend-only ML pipeline tests
pytest tests/test_historical_intelligence_v2.py tests/test_weekend_only_pipeline.py

# Run terminal UI and CLI interaction tests
pytest tests/test_terminal_ui.py tests/test_cli_menu.py
```

---

## Contributing

1. **Fork the Repository:** Create a personal fork on GitHub.
2. **Create a Feature Branch:**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Implement Changes & Tests:** Add unit/integration tests in [`tests/`](tests/) matching existing conventions.
4. **Run Verification:** Confirm zero test regressions (`pytest` must exit with code 0).
5. **Submit a Pull Request:** Open a PR against `main` with detailed verification logs.

---

## License

This project is licensed under the [MIT License](LICENSE).
