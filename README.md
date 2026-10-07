# F1 Unified Race Prediction Engine

A production-grade, config-driven machine learning and simulation platform for Formula 1 Grand Prix race predictions.

Designed from the ground up to replace fragmented single-track pipelines with a single, modular engine capable of predicting **any circuit on the Formula 1 calendar** without writing track-specific code.

---

## Key Capabilities & Architectural Highlights

1. **Config-Driven Circuit Extensibility**
   - All circuit parameters, physical traits, overtaking metrics, historical availability, and known grid penalties are defined in [`config/tracks.json`](file:///d:/f1%20race%20pred/f1-race-prediction/config/tracks.json).
   - Adding a new race (e.g., Monza, Silverstone, Singapore) requires **zero python code edits**—simply add an entry to `config/tracks.json`.

2. **Strict Zero-Leakage Architecture**
   - **Chronological Form Isolation**: Rolling form (EWMA finishing position, EWMA qualifying position, points rate, DNF hazard) is computed strictly on rounds prior to the target round (`Round < target_round`).
   - **Target Isolation**: Target race outcomes (`ActualFinish`, `Position`, `Points`, `Status`) are strictly forbidden from pre-race target feature sets and validated via automated anti-leakage audits.
   - **Session Stage Gatekeeping**: Enforces strict cutoffs for incomplete weekends (`pre_fp1`, `after_fp1`, `after_fp2`, `after_fp3`, `after_qualifying`).

3. **FIA Sporting Regulations Grid Construction**
   - Accurately models FIA starting grid procedures with penalty resolution:
     - Back-of-grid penalties (>=15 places) relegated to rear slots.
     - Grid-drop penalties shift drivers backward while unpenalized drivers shift forward into vacated slots.
     - Verified: In Sepang 2026, Hadjar (Q3, +5 penalty) starts P8; Colapinto (+15) starts P21; Lindblad (+30) starts P22.

4. **25,000-Iteration Monte Carlo Simulation**
   - **Residual-Calibrated Uncertainty**: Replaces arbitrary pace noise with out-of-fold walk-forward validation residuals ($\sigma = 1.4826 \times \text{MAD}(\text{residuals})$).
   - **Empirical Incident Modeling**: Direct empirical DNF hazard modeling from completed season rounds with heavy-tailed exponential impact shocks.
   - **Full Probabilistic Distributions**: Outputs Win%, Podium% (Top 3), Top 5%, Points% (Top 10), and P5–P95 percentiles.

---

## Directory Structure

```text
f1-race-prediction/
├── config/
│   ├── tracks.json          # Circuit profiles, lengths, round numbers, penalties
│   ├── model_config.json    # RF & XGB hyperparameters, 31 feature definitions, MC parameters
│   └── settings.json        # System paths, logging configuration
├── data/
│   ├── raw/                 # Season rounds (1-15) and circuit session CSVs
│   └── processed/           # Cached feature matrices
├── outputs/
│   ├── sepang/              # Sepang predictions, model comparison, markdown report
│   └── baku/                # Baku predictions, model comparison, markdown report
├── src/
│   ├── utils/               # Timing converters, driver/team normalizers, logger
│   ├── data/                # DataLoader, DataValidator, FastF1 collectors
│   ├── features/            # Grid penalties, season EWMA, practice & quali features
│   ├── models/              # ModelTrainer, ModelEvaluator, Predictor
│   ├── simulation/          # 25,000 MonteCarloSimulator
│   └── pipeline/            # Master RacePredictionPipeline
├── tests/                   # Pytest automated test suite (leakage, grid, timing, config)
├── main.py                  # Master CLI entrypoint
└── pytest.ini               # Test configuration
```

---

## Command Line Interface (CLI)

### 1. Predict Race Outcome
Run the end-to-end pipeline (data validation, feature construction, model fitting, deterministic blending, and 25,000 Monte Carlo simulations):

```bash
# Predict Sepang 2026 after qualifying
python main.py predict --track sepang --year 2026 --stage after_qualifying

# Predict Baku 2026 after qualifying
python main.py predict --track baku --year 2026 --stage after_qualifying

# Predict earlier stage (e.g. after Friday practice)
python main.py predict --track sepang --year 2026 --stage after_fp2
```

Outputs are automatically saved to:
- CSV Table: `outputs/<track>/<track>_<year>_<stage>_predictions.csv`
- Model Evaluation: `outputs/<track>/<track>_<year>_model_comparison.csv`
- Executive Report: `outputs/<track>/<track>_<year>_<stage>_report.md`

### 2. Walk-Forward Cross-Validation
Run chronological walk-forward out-of-fold evaluation across completed season rounds:

```bash
python main.py evaluate --track sepang --year 2026
```

Evaluates 7 ensemble weight combinations [100% RF down to 100% XGB] across MAE, MedAE, RMSE, and Spearman rank correlation.

### 3. Validate Data & Audit for Leakage
Run automated integrity and anti-leakage audits:

```bash
python main.py validate --track sepang --year 2026
```

Verifies:
- Zero post-race columns in pre-race target feature sets.
- Strict absence of target round or future rounds in training data.
- Starting grid uniqueness and 1..22 permutation.

---

## Benchmark Validation Results

### Chronological Walk-Forward Validation (Rounds 4–15)

| Model Configuration | MAE | MedAE | RMSE | Spearman Correlation |
|:--------------------|:---:|:-----:|:----:|:-------------------:|
| **100% Random Forest** | **3.651** | 2.934 | **4.803** | **0.656** |
| 75% RF / 25% XGB | 3.637 | 2.948 | 4.814 | 0.656 |
| 60% RF / 40% XGB | 3.636 | 2.934 | 4.830 | 0.653 |
| 50% RF / 50% XGB | 3.638 | 2.911 | 4.843 | 0.651 |
| 40% RF / 60% XGB | 3.643 | 2.848 | 4.859 | 0.649 |
| 25% RF / 75% XGB | 3.655 | 2.825 | 4.888 | 0.646 |
| 100% XGBoost | 3.685 | 2.766 | 4.949 | 0.639 |

- **Empirical DNF Risk**: 15.8%
- **Calibrated Uncertainty ($\sigma$)**: 4.2676 (derived from 264 walk-forward residuals)

### Sepang 2026 Starting Grid & Predicted Order (After Qualifying)
- Grid Penalties Applied: `HAD +5` (starts P8), `COL +15` (starts P21), `LIN +30` (starts P22).
- Top 3 Predicted Finishers:
  1. **VER** (Red Bull Racing, Grid P1) — 27.2% Win, 56.7% Podium
  2. **HAM** (Ferrari, Grid P2) — 16.9% Win, 43.7% Podium
  3. **ANT** (Mercedes, Grid P3) — 14.7% Win, 40.7% Podium

---

## Running the Automated Test Suite

```bash
python -m pytest tests/ -v
```

10 automated tests verify:
- Anti-leakage safeguards and failure triggers.
- Penalty application and FIA grid drop logic.
- Robust lap timing conversions.
- Dynamic circuit resolution from `tracks.json`.
