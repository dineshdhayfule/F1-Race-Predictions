# ApexGrid

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Test Suite](https://img.shields.io/badge/tests-300%20passed%20(100%25)-brightgreen.svg)](tests/)
[![Architecture](https://img.shields.io/badge/architecture-Multi--Model%20Ensemble-orange.svg)](src/models/)
[![Simulation](https://img.shields.io/badge/Monte%20Carlo-25%2C000%20runs-purple.svg)](src/simulation/)

> **Predicting the Race Beyond the Grid.**  
> *An ML-powered motorsport intelligence, multi-model ensemble, and Monte Carlo simulation platform for Formula 1 Grand Prix outcome forecasting.*

---

## Overview

**ApexGrid** is a production-grade machine learning and stochastic simulation platform engineered for Formula 1 performance analytics and race outcome forecasting. Built to eliminate hardcoded circuit heuristics, ApexGrid combines multi-stage telemetry ingestion, non-linear tire degradation modeling, pit stop strategy simulation, environmental weather dynamics, safety car hazard forecasting, and multi-model ensembling into a unified predictive pipeline.

ApexGrid models the probabilistic distribution of finishing positions, win probabilities, podium chances, and points finishes across all 23 championship rounds without future-session data leakage.

---

## Key Features

- **Validated Multi-Model Ensemble Architecture:** Blends Random Forest (40%), XGBoost (30%), and L2-regularized Ridge Regression (30%) calibrated via chronological walk-forward validation to minimize out-of-sample error and prevent tree-based overfitting.
- **Circuit-Aware Starting-Grid Blending:** Calibrates starting-grid position weighting dynamically according to circuit topology:
  - *Street circuits* (Monaco, Baku, Singapore): 50% Grid / 50% ML ($w_{\text{grid}} = 0.50$, $w_{\text{ml}} = 0.50$) for high positional conservatism.
  - *Permanent circuits* (Silverstone, Spa, Monza, Interlagos): 35% Grid / 65% ML ($w_{\text{grid}} = 0.35$, $w_{\text{ml}} = 0.65$).
  - *Temporary circuits* (Melbourne, Montreal): 45% Grid / 55% ML ($w_{\text{grid}} = 0.45$, $w_{\text{ml}} = 0.55$).
- **Strict Stage-Aware Temporal Boundaries:** Enforces chronological cutoffs preventing target and future-session data leakage across seven weekend stages: `pre_fp1`, `after_fp1`, `after_fp2`, `after_fp3`, `after_sprint_qualifying`, `after_sprint_race`, and `after_qualifying`.
- **25,000-Iteration Monte Carlo Engine:** Simulates 25,000 stochastic race outcomes per entrant using out-of-fold residual-calibrated pace variance ($\sigma$), empirical DNF hazard modeling, field compression dynamics, and driver-specific wet skill differentials.
- **Five Specialized Physical & Strategic Subsystems:**
  - *Race Pace Intelligence:* Fuel-corrected stint pace extraction ($0.035\text{s}/\text{lap}$ burn-off) and clean-air pace modeling.
  - *Tire Degradation Modeling:* Non-linear exponential degradation wear curves ($\Delta t = \alpha \cdot L^\beta$) across Soft, Medium, Hard, Intermediate, and Wet compounds.
  - *Pit Stop Strategy Simulation:* Multi-stint optimization under track-specific pit-loss seconds and strict FIA Article 30.5 dry compound regulations.
  - *Race Control Hazard Modeling:* Empirical venue disruption modeling (Safety Car, VSC, and Red Flag probabilities).
  - *Weather Intelligence:* Live ambient and track temperature evaluation with stochastic rainfall scenario sampling.
- **Multi-Source Starting Grid Resolution:** Three-tier grid resolution pipeline supporting official FIA Starting Grid Manifests (`data/grids/`), OpenF1 live feeds, and programmatic FIA Article 42 penalty cascade re-alignment.
- **Safe Data Management & Boundary Guard:** Integrated cleanup utility with strict root directory containment, dry-run preview mode, and protected file preservation (`outputs/`, `data/grids/`, `config/`).
- **300-Test Verified Regression Suite:** Comprehensive automated test coverage (299 passing, 1 skipped in non-interactive environments) ensuring 100% pass rates with exit code 0.

---

## Tech Stack

| Component | Technologies |
| :--- | :--- |
| **Language & Runtime** | Python 3.10, 3.11, 3.12, 3.13 |
| **Machine Learning** | Scikit-Learn (Random Forest, Ridge Regression, Preprocessing), XGBoost (Gradient Boosted Trees) |
| **Scientific Computing** | NumPy, SciPy, Pandas |
| **Telemetry & Live Data** | FastF1, OpenF1 REST API |
| **Testing & Verification** | Pytest, Pytest-Cov, Pytest-JSON-CTRF |
| **CLI & Terminal UI** | Argparse, ANSI Terminal Formatting |

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
- [`config/model_config.json`](config/model_config.json): Estimator hyperparameters, feature sets, circuit blending weights, and pruning toggles.
- [`config/tracks.json`](config/tracks.json): Canonical circuit mappings, track lengths, circuit types, and championship calendars.
- [`config/strategy_config.json`](config/strategy_config.json): Compound parameters, pit lane loss times, and degradation priors.
- [`config/settings.json`](config/settings.json): Cache locations, logging thresholds, and output directories.

---

## Usage Examples

### 1. Interactive Guided Menu (Recommended)

Launch the interactive terminal interface:

```bash
python main.py
```

The menu provides seven guided workflows:
1. **Predict a Race:** Interactive wizard for circuit selection, weekend stage, model selection, and forecast execution.
2. **Load / Collect Data:** Ingest FastF1 practice telemetry or full-season championship rounds.
3. **View Previous Predictions & Reports:** Inspect generated prediction tables and Markdown race reports.
4. **Evaluate a Prediction:** Score forecast accuracy against official race classifications.
5. **System Health & Data Readiness:** Run diagnostic checks and data availability audits.
6. **Configuration & Supported Circuits:** View circuit parameters, track types, and model settings.
7. **Data Management:** Safely preview and prune disposable FastF1 cache and telemetry files.

---

### 2. Command Line Interface (CLI)

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

ApexGrid maintains **300 automated tests** across 33 test modules.

```bash
# Run the entire test suite
pytest

# Run targeted model selection and Ridge weight reporting tests
pytest tests/test_model_selection.py

# Run Phase 4 production implementation tests
pytest tests/test_phase4_production.py

# Run sandbox parity and output isolation tests
pytest tests/test_parity_regression.py

# Run anti-leakage audit tests
pytest tests/test_leakage.py tests/test_sprint_leakage_remediation.py
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
