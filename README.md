# F1 Unified Race Prediction Engine

A config-driven machine learning and Monte Carlo simulation platform designed to predict Formula 1 Grand Prix race results across the championship calendar without requiring circuit-specific Python code.

The engine combines Random Forest and XGBoost models, chronological walk-forward validation, stage-aware feature engineering, multi-source starting-grid resolution, and probabilistic race simulations.

---

## Key Capabilities & Architectural Highlights

### 1. Config-Driven Circuit Extensibility

- Circuit parameters, physical characteristics, historical round numbers, Sprint configuration, and known grid penalties are defined in `config/tracks.json`.
- Adding or configuring a circuit requires **zero track-specific Python code**.
- Supports all 23 configured circuits on the 2026 calendar, covering both Normal and Sprint weekend formats.

### 2. Stage-Aware Data Architecture & Leakage Prevention

- Chronological walk-forward training uses completed season rounds prior to the target event.
- Target race results are excluded from pre-race feature construction and ML model inputs.
- Stage-aware session loading enforces information cutoff boundaries:

  - `pre_fp1`: Zero weekend telemetry (rolling form and circuit priors).
  - `after_fp1`: Practice 1 telemetry observed.
  - `after_fp2`: Practice 1 + 2 telemetry observed (Normal weekends).
  - `after_fp3`: Practice 1 + 2 + 3 telemetry observed (Normal weekends).
  - `after_sprint_qualifying`: FP1 + Sprint Shootout classification observed (Sprint weekends).
  - `after_sprint_race`: FP1 + Sprint Qualifying + Sprint Race results observed (Sprint weekends).
  - `after_qualifying`: Grand Prix Qualifying information observed, including available starting-grid information.
  - `post_race`: Post-race evaluation against race results.

- Automated anti-leakage audits check chronological data integrity before model training.

### 3. Multi-Source Starting Grid Resolution

The engine explicitly distinguishes `QualifyingPosition` from `ActualStartingGrid` through a multi-tier starting-grid resolution architecture.

1. **OpenF1 API Ingestion:** Retrieves published starting-grid data from the OpenF1 API when available. OpenF1 is treated as a third-party data source.
2. **Official FIA Grid Manifests:** Retrieves locally stored starting grids from manifests (`data/grids/{year}_round_{round}_starting_grid.json`) referencing authoritative FIA documents.
3. **Deterministic FIA Penalty Resolver (Fallback):** When a suitable external manifest is unavailable, applies the implemented starting-grid penalty resolution logic:
   - Cumulative place drops (+3, +5, +10, +15 places).
   - Back-of-grid penalties (+30 / power unit changes).
   - Ordered cascade re-alignment of unaffected cars.
   - Pit-lane starter separation.

- **Provenance Tracking:** Prediction tables, CLI summaries, and Markdown reports record the starting-grid source, distinguishing `Official FIA Manifest`, `OpenF1 API Grid`, and `Algorithmic Provisional Grid`.

### 4. 37-Feature Engineering System

The feature vector combines:

- Starting-grid position and grid normalization.
- Grid-to-race position conversion dynamics.
- Qualifying speed and Q1/Q2/Q3 progression pace gaps.
- FP1/FP2/FP3 telemetry and representative lap pace.
- Driver Exponentially Weighted Moving Average (EWMA) form.
- Historical and empirical driver DNF rates.
- Constructor rolling pace and team points-scoring form.
- Sprint qualifying and Sprint race finishing transitions.
- Teammate qualifying pace gap and teammate grid delta.
- Driver-versus-teammate head-to-head performance ratios.

### 5. Selectable ML Models & Custom Hybrid Weighting

Predictions support user-selectable models and customizable ensemble weighting.

- **Random Forest Only (`rf`):** Uses the Random Forest model at 100%.
- **XGBoost Only (`xgb`):** Uses the XGBoost model at 100%.
- **Hybrid Ensemble (`hybrid`):** Default blended ensemble using 60% Random Forest and 40% XGBoost.
- **Custom Hybrid Blends:** User-defined RF and XGBoost weights, such as 70% RF / 30% XGBoost.
- **Interactive Model Selection:** Supplying the bare `--model` flag opens an interactive terminal menu with separate weight inputs, sum-to-100% validation, and configuration confirmation.
- **Model-Aligned Evaluation:** Walk-forward evaluation is designed to use the selected model and hybrid weights, with corresponding validation residuals used for Monte Carlo uncertainty calibration.

### 6. 25,000-Iteration Monte Carlo Simulation

- Runs 25,000 stochastic simulations per driver to generate finishing-position distributions.
- Uses walk-forward validation residuals for uncertainty calibration.
- Incorporates empirical DNF hazard modeling and attrition simulation.
- Generates probabilistic prediction metrics:

  - Expected finishing position.
  - Win probability.
  - Podium probability (Top 3).
  - Top 5 probability.
  - Points probability (Top 10).
  - P5–P95 uncertainty intervals.

---

## Installation & Environment Setup

### Prerequisites

- Python 3.10, 3.11, 3.12, or 3.13.
- Internet connection (required for initial FastF1 telemetry and OpenF1 starting-grid downloads).

### Setup Instructions

**1. Clone the repository**

```bash
git clone https://github.com/dineshdhayfule/F1-Race-Predictions.git
cd F1-Race-Predictions
```

**2. Create and activate a virtual environment**

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage Guide

### 1. Interactive Terminal Menu (Recommended)

Launch the guided terminal interface:

```bash
python main.py
```

Or explicitly:

```bash
python main.py menu
```

The interactive menu provides six workflows:

1. **Predict a Race:** Guided prediction wizard with circuit selection, prediction-stage selection, model selection (RF, XGBoost, Hybrid, or Custom Weights), and simulation.
2. **Load / Collect Data:** Inspect dataset inventory, collect track practice telemetry or season rounds through FastF1, and audit schema integrity.
3. **View Previous Predictions & Reports:** Browse generated output folders, inspect prediction tables, and view Markdown reports.
4. **Evaluate a Prediction:** Evaluate saved forecast CSVs against race results with duplicate and mismatch safeguards.
5. **System Health & Data Readiness:** Environment diagnostics, dependency validation, and test execution.
6. **Configuration & Supported Circuits:** Inspect model hyperparameters and browse configured circuits.

### 2. Command Line Interface (CLI)

The engine provides CLI commands for automation, scripting, and direct prediction execution.

#### Predict a Grand Prix

**Default Hybrid Ensemble (60% RF / 40% XGBoost)**

```bash
python main.py predict --track mexico --year 2026 --stage after_qualifying --top 10
```

**Pure Random Forest (100%)**

```bash
python main.py predict --track mexico --year 2026 --stage after_qualifying --model rf
```

**Pure XGBoost (100%)**

```bash
python main.py predict --track mexico --year 2026 --stage after_qualifying --model xgb
```

**Custom Hybrid Weights (70% RF / 30% XGBoost)**

```bash
python main.py predict --track mexico --year 2026 --stage after_qualifying --model hybrid --rf-weight 70 --xgb-weight 30
```

**Interactive Model Selection**

```bash
python main.py predict --track mexico --year 2026 --stage after_qualifying --model
```

This opens the model-selection menu, where you can choose RF, XGBoost, or a custom hybrid configuration.

#### Run Chronological Model Evaluation

```bash
python main.py evaluate --track sepang --year 2026
```

#### Validate Dataset & Anti-Leakage

```bash
python main.py validate --track baku --year 2026 --stage after_qualifying
```

### 3. Data Collection

Collect race results and telemetry through FastF1.

**Collect season race and qualifying results (example: Round 18)**

```bash
python -m src.data.collect_season --year 2026 --rounds 18
```

**Collect weekend practice and qualifying telemetry for a specific circuit**

```bash
python -m src.data.collect_track --track austin --year 2026
```

---

## License

This project is licensed under the [MIT License](LICENSE).