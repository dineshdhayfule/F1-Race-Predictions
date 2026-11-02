# APEXGRID

> **Predicting the Race Beyond the Grid.**  
> *An ML-powered motorsport intelligence and race prediction platform.*

---

## Overview

**APEXGRID** is a modern, configuration-driven machine learning and Monte Carlo simulation platform engineered for Formula 1 performance analytics and Grand Prix outcome forecasting. Designed from first principles to eliminate hardcoded circuit rules, APEXGRID unifies multi-stage telemetry ingestion, non-linear tire degradation modeling, pit-stop strategy simulation, weather intelligence, safety-car hazard modeling, and stochastic race simulations into an integrated motorsport prediction pipeline.

APEXGRID bridges the gap between raw track telemetry and probabilistic race forecasts, predicting finishing distributions, win probabilities, podium chances, and points finishes across the entire 23-circuit championship calendar.

---

## Core Capabilities

```
                                  +-------------------------------------------------------+
                                  |                       APEXGRID                        |
                                  |         Predicting the Race Beyond the Grid           |
                                  +-------------------------------------------------------+
                                                              |
           +--------------------------+-----------------------+-----------------------+--------------------------+
           |                          |                       |                       |                          |
           v                          v                       v                       v                          v
+--------------------+   +----------------------+   +-------------------+   +--------------------+   +-----------------------+
|  Stage-Aware Form  |   | Race Pace & Deg      |   | Pit Stop Strategy |   | Safety Car / VSC   |   | Weather Intelligence  |
|  & Telemetry Engine|   | Nonlinear Physics    |   | Stint Simulation  |   | Hazard Modeling    |   | Driver Wet Bonuses    |
+--------------------+   +----------------------+   +-------------------+   +--------------------+   +-----------------------+
           |                          |                       |                       |                          |
           +--------------------------+-----------------------+-----------------------+--------------------------+
                                                              |
                                                              v
                                              +-------------------------------+
                                              | 62-Feature Feature Matrix     |
                                              +-------------------------------+
                                                              |
                                                              v
                                              +-------------------------------+
                                              | ML Ensemble (RF + XGBoost)    |
                                              | Walk-Forward Cross-Validation |
                                              +-------------------------------+
                                                              |
                                                              v
                                              +-------------------------------+
                                              | Monte Carlo Simulator         |
                                              | 25,000 Stochastic Iterations  |
                                              +-------------------------------+
                                                              |
                                                              v
                                              +-------------------------------+
                                              | Probabilistic Race Forecasts  |
                                              | (Win %, Top 3, Top 5, Points) |
                                              +-------------------------------+
```

### 1. Config-Driven Circuit Architecture & Canonical Resolution
- All circuit layouts, physical attributes, historical rounds, sprint weekend configurations, and known penalties are declared in [`config/tracks.json`](config/tracks.json).
- **Zero per-circuit Python branching:** Adding or updating a circuit requires no Python changes (`if track == 'monaco':` conditionals are strictly forbidden).
- **Centralized Canonical Resolution (`src/utils/circuit_mapping.py`):** Canonical resolver mapping circuit IDs (`austin`, `sao_paulo`, `baku`, `sepang`, etc.), common aliases (`cota`, `interlagos`, `suzuka`, `montreal`), and official event titles across all 23 championship rounds.

### 2. Stage-Aware Data Architecture & Strict Temporal Anti-Leakage
Chronological information boundaries enforce realistic data availability across all weekend stages, eliminating target and future-session data leakage:
- `pre_fp1`: Zero weekend telemetry observed; relies strictly on prior rolling form and historical circuit priors.
- `after_fp1`: Practice 1 telemetry observed.
- `after_fp2`: Practice 1 + Practice 2 telemetry observed (Conventional weekends).
- `after_fp3`: Practice 1 + Practice 2 + Practice 3 telemetry observed (Conventional weekends).
- `after_sprint_qualifying`: FP1 + Sprint Shootout observed; Sprint Race results, Sunday Qualifying, and Sunday Starting Grid are **strictly masked**.
- `after_sprint_race`: FP1 + Sprint Shootout + Sprint Race observed; Sunday Qualifying and Sunday Starting Grid remain **strictly masked**.
- `after_qualifying`: Grand Prix Qualifying results and starting-grid penalties observed; Race results remain strictly inaccessible.
- `post_race`: Post-race evaluation and validation against official classification.

### 3. Integrated Physics & Strategic Simulation Subsystems
APEXGRID incorporates five specialized analytical models:
1. **Race Pace Intelligence (`src/features/race_pace.py`):** Filters in/out laps and short runs, corrects for fuel-load lap time decay ($0.035\text{s}/\text{lap}$), and derives clean-air long-run pace differentials.
2. **Tire Degradation Modeling (`src/features/tire_degradation.py`):** Models non-linear exponential tire wear ($\Delta t = \alpha \cdot L^\beta$) across Soft, Medium, Hard, Intermediate, and Wet compounds, calculating thermal degradation slopes and cliff laps.
3. **Pit Stop Strategy Simulator (`src/simulation/strategy_simulator.py`):** Evaluates 1-stop, 2-stop, and 3-stop strategies using track-specific pit-loss seconds from [`config/strategy_config.json`](config/strategy_config.json) and strictly enforces FIA Article 30.5 (two distinct dry-weather compounds mandatory in dry races).
4. **Safety Car / VSC / Red Flag Probability Model (`src/simulation/race_control_model.py`):** Computes empirical disruption likelihoods (e.g. Monaco 75% SC, Jeddah 80% SC) and simulates field compression during neutralized laps.
5. **Weather-Aware Race Predictions (`src/features/weather_features.py`):** Evaluates ambient temperature, track temperature, humidity, wind velocity, and rainfall probability, applying empirical wet-weather driver skill bonuses under rain conditions.

### 4. Multi-Source Starting Grid Resolution & Penalty Staging
Distinguishes `QualifyingPosition` from `ActualStartingGrid` using a three-tier starting grid resolution hierarchy:
1. **Official FIA Grid Manifests:** Ingests finalized starting grids from `data/grids/{year}_round_{round}_starting_grid.json` referencing official FIA Stewards' documents.
2. **OpenF1 API Ingestion:** Ingests live starting grids published via OpenF1.
3. **Deterministic FIA Penalty Resolver (Fallback):** Programmatically resolves grid drops conforming to FIA Sporting Regulations (Article 42):
   - Cumulative grid place penalties (+3, +5, +10, +15 places).
   - Back-of-the-grid penalties (+30 / power unit changes).
   - Unaffected car re-alignment cascade.
   - Pit lane start separation.

### 5. 25,000-Iteration Monte Carlo Simulation & Entrant Integrity
- Simulates 25,000 stochastic race outcomes per entrant using walk-forward residual-calibrated pace variance ($\sigma$).
- **Strict Entrant Integrity:** Non-starting entrants (`DNS` or `DSQ`) are excluded from simulated track laps to preserve physical realism, but their entrant records are explicitly preserved in the final output with `PredictedPosition = np.nan`, `ExpectedFinish = np.nan`, and `0.0%` win/podium probabilities, ensuring the full 22-car grid is retained without row truncation.
- Outputs include:
  - Expected finishing position (mean) and median finish.
  - Win probability ($\sum = 100.0\%$).
  - Podium probability (Top 3, $\sum = 300.0\%$).
  - Top 5 probability and Points probability (Top 10).
  - P5–P95 percentile confidence intervals.

### 6. Selectable ML Models & Custom Hybrid Weighting
- **Random Forest (`rf`):** 100% Random Forest regressor.
- **XGBoost (`xgb`):** 100% XGBoost regressor (400 estimators, early stopping).
- **Hybrid Ensemble (`hybrid`):** Blended ensemble with configurable weights (default: 60% RF / 40% XGBoost).
- **Custom Hybrid Blends:** Specify custom weight splits (e.g. 70% RF / 30% XGBoost, 80% RF / 20% XGBoost).
- **Interactive Model Selection:** Supplying bare `--model` opens an interactive menu to configure and preview weights with sum-to-100% validation.
- Walk-forward validation dynamically evaluates the selected model configuration, calibrating Monte Carlo uncertainty residuals directly to the chosen ensemble.

### 7. Safe Data Management & Deletion Engine
- Integrated data cleanup utility (`src/data/data_manager.py`) with strict path containment preventing directory traversal.
- **Protected File Preservation:** Source code, configurations (`config/`), automated tests (`tests/`), documentation, starting grid manifests (`data/grids/`), and git-tracked baseline output fixtures (`outputs/`) are permanently protected from deletion.
- **Safety Safeguards:** Dry-run preview mode, non-interactive force flags, and typed phrase confirmation safeguards (`DELETE DATA NOW`) prevent accidental data loss.

---

## Machine Learning & Simulation Architecture

```
+----------------------------------------------------------------------------------------------------+
|                                      DATA INGESTION LAYER                                          |
|  FastF1 Live Telemetry  |  OpenF1 Starting Grids  |  FIA Document Manifests  |  Historical CSVs    |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                 FEATURE ENGINEERING (62 FEATURES)                                  |
|  - Starting Grid Dynamics (grid_norm, expected_grid_finish, grid_vs_quali_delta)                   |
|  - Qualifying Pace Progression (quali_pos_norm, quali_gap_pct, q1/q2/q3 gaps)                      |
|  - Practice Telemetry (fp1/fp2/fp3 lap times and progression deltas)                               |
|  - Form & Momentum (Race_EWMA_Finish [alpha=0.3], Constructor_RollingMeanFinish)                   |
|  - Teammate Differentials (teammate_grid_delta, rolling_h2h_win_ratio)                             |
|  - Sprint Weekend Metrics (sprint_quali_pos, sprint_finish_pos, sprint_pos_delta)                  |
|  - Race Pace Intelligence (long_run_pace_gap_pct, race_pace_consistency)                           |
|  - Tire Degradation (soft/medium/hard deg slopes, compound pace deltas)                            |
|  - Strategy Simulation (optimal_strategy_stops, strategy_time_loss_sec)                            |
|  - Race Control & Weather (p_safety_car, p_vsc, air_temp, track_temp, rain_probability)            |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                    TEMPORAL STAGE-AWARE MASKING                                    |
|  Masks future features (Sunday qualifying, grid, sprint results) based on session timeline cutoff |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                               CHRONOLOGICAL WALK-FORWARD VALIDATION                                |
|  Folds 4..N-1 | Residual-Calibrated Uncertainty Sigma | Out-of-Fold Model Selection & Weighting    |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                              25,000-ITERATION MONTE CARLO SIMULATION                               |
|  Stochastic Lap Simulation | Empirical DNF Hazard | Field Compression | Entrant Integrity Retention|
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                      REPORTING & FORECAST LAYER                                    |
|  Prediction Tables (CSV) | Markdown Executive Reports | Interactive CLI Tables | Visual Overviews  |
+----------------------------------------------------------------------------------------------------+
```

---

## Technology Stack

- **Core Runtime:** Python 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.13)
- **Machine Learning:** `scikit-learn`, `xgboost`
- **Data Engineering & Analytics:** `pandas`, `numpy`, `scipy`
- **Motorsport Telemetry:** `fastf1`, `openf1`
- **Testing & Quality Assurance:** `pytest`, `pytest-cov`, `pytest-json-ctrf`
- **Interface & Utilities:** Standard Python CLI (`argparse`), ANSI Color Terminal UI

---

## Repository Structure

```text
f1-race-prediction/
├── config/
│   ├── model_config.json          # ML models, hyperparameters, and feature definitions
│   ├── settings.json              # Runtime directories, FastF1 cache, and logging configuration
│   ├── strategy_config.json        # Tire compounds, circuit priors, and pit strategy parameters
│   └── tracks.json                # All 23 championship circuits, attributes, and penalties
├── data/
│   ├── grids/                     # Authoritative FIA starting grid JSON manifests
│   └── raw/                       # Cached FastF1 telemetry and historical season results
├── outputs/                       # Baseline reference fixtures and generated forecast reports
│   ├── baku/                      # Baku Grand Prix baseline reference fixtures
│   └── sepang/                    # Sepang Grand Prix baseline reference fixtures
├── reports/                       # Formal engineering audits and verification documentation
│   ├── APEXGRID_REBRANDING_REPORT.md
│   ├── FINAL_INDEPENDENT_VERIFICATION.md
│   └── PRIORITY_REMEDIATION_REPORT.md
├── src/
│   ├── cli/                       # Terminal UI utilities and interactive guided menu
│   │   ├── menu.py                # 7-workflow interactive menu controller
│   │   └── terminal_ui.py         # ANSI colors, banners, and table formatters
│   ├── data/                      # Ingestion, validation, and safe data management
│   │   ├── collect_season.py      # Full season race and qualifying collector
│   │   ├── collect_track.py       # Single-venue weekend telemetry collector
│   │   ├── data_loader.py         # Multi-source session loader and stage router
│   │   ├── data_manager.py        # Safe deletion engine and path boundary guard
│   │   └── data_validator.py      # Schema verification and anti-leakage audits
│   ├── features/                  # 62-feature engineering system
│   │   ├── build_features.py      # Feature engineering and temporal stage masking
│   │   ├── grid_features.py       # Grid penalty resolution and provenance formatting
│   │   ├── race_pace.py           # Long-run fuel-corrected stint pace intelligence
│   │   ├── tire_degradation.py    # Non-linear compound tire degradation modeling
│   │   └── weather_features.py    # Weather intelligence and wet-weather skill modeling
│   ├── models/                    # Model training, evaluation, and inference
│   │   ├── evaluate.py            # Chronological walk-forward cross-validation
│   │   ├── predict.py             # Model inference and prediction generation
│   │   └── train.py               # Random Forest and XGBoost model fitting
│   ├── pipeline/                  # End-to-end execution orchestrator
│   │   └── race_prediction_pipeline.py
│   ├── simulation/                # Strategic and probabilistic simulation
│   │   ├── monte_carlo.py         # 25,000-iteration Monte Carlo race simulator
│   │   ├── race_control_model.py  # Safety Car, VSC, and Red Flag disruption model
│   │   └── strategy_simulator.py  # Multi-stint pit stop strategy simulator
│   └── utils/                     # Shared utilities
│       ├── circuit_mapping.py     # Centralized canonical circuit-key resolver
│       ├── driver_mapping.py      # Driver normalization and course vehicle filter
│       └── logging_utils.py       # Standardized logging
├── tests/                         # 256 automated unit, integration, and regression tests
├── main.py                        # APEXGRID master CLI entrypoint
├── pytest.ini                     # Pytest configuration
├── requirements.txt               # Production dependencies
└── README.md                      # Project documentation
```

---

## Installation & Quick Start

### 1. Clone & Set Up Environment

```bash
# Clone the repository
git clone https://github.com/dineshdhayfule/F1-Race-Predictions.git
cd F1-Race-Predictions

# Create and activate virtual environment
# Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage Guide

### 1. Interactive Terminal Interface (Recommended)

Launch the guided terminal interface:

```bash
python main.py
```
*(or explicitly: `python main.py menu`)*

The menu provides seven guided workflows:
1. **Predict a Race:** Guided wizard for circuit selection, timeline stage selection, model configuration, and simulation.
2. **Load / Collect Data:** Ingest FastF1 practice telemetry or full season rounds with automated schema audits.
3. **View Previous Predictions & Reports:** Browse generated output folders, inspect prediction tables, and view Markdown reports.
4. **Evaluate a Prediction:** Score forecast accuracy against race results with duplicate and mismatch safeguards.
5. **System Health & Data Readiness:** Run automated diagnostic audits, dependency checks, and stage readiness validations.
6. **Configuration & Supported Circuits:** Inspect model hyperparameters, track parameters, and strategy configurations.
7. **Data Management (Clean / Reset Data):** Safely preview and clean disposable cache, prediction, or telemetry files.

---

### 2. Command Line Interface (CLI)

#### Race Predictions

```bash
# Default Hybrid Ensemble (60% RF / 40% XGBoost)
python main.py predict --track mexico --year 2026 --stage after_qualifying --top 10

# Pure Random Forest (100% RF)
python main.py predict --track mexico --year 2026 --stage after_qualifying --model rf

# Pure XGBoost (100% XGB)
python main.py predict --track mexico --year 2026 --stage after_qualifying --model xgb

# Custom Hybrid Blending (70% RF / 30% XGBoost)
python main.py predict --track mexico --year 2026 --stage after_qualifying --model hybrid --rf-weight 70 --xgb-weight 30

# Interactive Model Selection
python main.py predict --track mexico --year 2026 --stage after_qualifying --model
```

#### Chronological Evaluation & Diagnostics

```bash
# Evaluate walk-forward model performance on a completed event
python main.py evaluate --track sepang --year 2026

# Validate dataset integrity and anti-leakage cutoffs
python main.py validate --track baku --year 2026 --stage after_qualifying
```

#### Data Collection

```bash
# Collect full season results for a specific round (e.g. Round 18)
python -m src.data.collect_season --year 2026 --rounds 18

# Collect weekend practice and qualifying telemetry for a circuit
python -m src.data.collect_track --track austin --year 2026
```

#### Safe Data Management

```bash
# Dry-run preview of disposable data
python main.py clean --preview

# Clean FastF1 cache with non-interactive confirmation
python main.py clean --target cache --force
```

---

## Testing & Quality Assurance

APEXGRID maintains a comprehensive suite of **256 automated tests** across 29 test files:

```bash
# Run the entire test suite
pytest

# Run targeted priority verification suites
pytest tests/test_sprint_leakage_remediation.py tests/test_circuit_mapping_remediation.py tests/test_monte_carlo_integrity.py

# Run five enhanced capabilities verification
pytest tests/test_five_capabilities.py
```

Audit documentation and verification reports:
- [`reports/APEXGRID_REBRANDING_REPORT.md`](reports/APEXGRID_REBRANDING_REPORT.md): Rebranding execution and repository audit report.
- [`reports/PRIORITY_REMEDIATION_REPORT.md`](reports/PRIORITY_REMEDIATION_REPORT.md): Core remediation report (sprint anti-leakage, canonical resolution, entrant integrity, COTA ingestion).
- [`reports/FINAL_INDEPENDENT_VERIFICATION.md`](reports/FINAL_INDEPENDENT_VERIFICATION.md): Independent verification and regression audit.

---

## License

This project is licensed under the [MIT License](LICENSE).
