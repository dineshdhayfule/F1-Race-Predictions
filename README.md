# F1 Unified Race Prediction Engine

A production-grade, config-driven machine learning and Monte Carlo simulation platform for predicting Formula 1 Grand Prix race results across the calendar without circuit-specific Python code.

---

## Key Capabilities & Architectural Highlights

### 1. Config-Driven Circuit Extensibility

- Circuit parameters, physical characteristics, historical availability, Sprint configuration, and known grid penalties are defined in `config/tracks.json`.
- Adding or configuring a circuit requires **zero track-specific Python code**.
- Supports both normal and Sprint weekends.

### 2. Strict Zero-Leakage Architecture

- Chronological walk-forward training using only data from rounds before the target round.
- Target race results are strictly excluded from pre-race feature construction.
- Stage-aware session loading prevents future sessions from entering earlier predictions.
- Supports:
  - `pre_fp1`
  - `after_fp1`
  - `after_fp2`
  - `after_fp3`
  - `after_sprint_qualifying`
  - `after_sprint_race`
  - `after_qualifying`
- Automated leakage audits verify chronological integrity.

### 3. FIA-Aware Starting Grid Construction

The system distinguishes between:

- `QualifyingPosition`
- `ActualStartingGrid`

Grid penalties are applied before race prediction, including:

- Grid drops
- Multiple penalties
- Back-of-grid penalties
- Reordering of unaffected drivers

Example Sepang 2026 validation:

- Hadjar: Q3 +5 penalty → P8
- Colapinto: +15 penalty → P21
- Lindblad: +30 penalty → P22

### 4. 37-Feature Prediction System

The current feature vector combines:

- Starting grid position and normalization
- Grid-to-race conversion
- Qualifying performance
- Q1/Q2/Q3 pace gaps
- FP1/FP2/FP3 performance
- Driver rolling form
- Driver DNF rate
- Constructor rolling form
- Sprint qualifying performance
- Sprint race performance
- Sprint position changes
- Teammate qualifying gap
- Teammate grid difference
- Driver-vs-teammate head-to-head performance

The current production configuration contains **37 features**.

### 5. Random Forest + XGBoost

The engine supports:

- Random Forest
- XGBoost
- RF/XGB ensemble configurations
- Chronological walk-forward validation
- Deterministic prediction
- Model comparison using:
  - MAE
  - MedAE
  - RMSE
  - Spearman rank correlation

Model selection is based on chronological out-of-fold performance rather than random train/test splitting.

### 6. 25,000-Iteration Monte Carlo Simulation

The prediction engine runs up to **25,000 simulations** using:

- Residual-calibrated uncertainty
- Empirical DNF risk
- Incident shock modeling
- Starting-grid information
- Predicted finishing distributions

Outputs include:

- Expected finish
- Win probability
- Top 3 probability
- Top 5 probability
- Points probability
- Prediction percentiles

---

## Directory Structure

```text
f1-race-prediction/
├── config/
│   ├── tracks.json
│   ├── model_config.json
│   └── settings.json
│
├── data/
│   ├── raw/
│   │   ├── season/
│   │   └── tracks/
│   └── processed/
│
├── outputs/
│   ├── sepang/
│   └── baku/
│
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   └── ...
│   ├── features/
│   │   ├── build_features.py
│   │   ├── weekend_features.py
│   │   ├── teammate_features.py
│   │   ├── grid_conversion.py
│   │   └── ...
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   ├── simulation/
│   │   └── monte_carlo.py
│   ├── pipeline/
│   │   └── ...
│   └── utils/
│
├── tests/
│
├── main.py
└── pytest.ini
