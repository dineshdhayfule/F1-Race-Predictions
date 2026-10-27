# F1 Unified Race Prediction Engine

A production-grade, config-driven machine learning and Monte Carlo simulation platform for predicting Formula 1 Grand Prix race results across the championship calendar without circuit-specific Python code.

---

## Key Capabilities & Architectural Highlights

### 1. Config-Driven Circuit Extensibility
- Circuit parameters, physical characteristics, historical round numbers, Sprint configuration, and known grid penalties are defined in `config/tracks.json`.
- Adding or configuring a circuit requires **zero track-specific Python code**.
- Supports all 23 configured circuits on the 2026 calendar, covering both Normal and Sprint weekend formats.

### 2. Strict Zero-Leakage Architecture
- Chronological walk-forward training uses only completed season rounds prior to the target event.
- Target race results are strictly excluded from pre-race feature construction and ML models.
- Stage-aware session loading enforces information cutoff boundaries:
  - `pre_fp1`: Zero weekend telemetry (pure rolling form and circuit priors).
  - `after_fp1`: Practice 1 telemetry observed.
  - `after_fp2`: Practice 1 + 2 telemetry observed (Normal weekends).
  - `after_fp3`: Practice 1 + 2 + 3 telemetry observed (Normal weekends).
  - `after_sprint_qualifying`: FP1 + Sprint Shootout classification observed (Sprint weekends).
  - `after_sprint_race`: FP1 + SQ + Sprint Race results observed (Sprint weekends).
  - `after_qualifying`: Official Grand Prix Qualifying observed (actual starting grid verified).
  - `post_race`: Post-race evaluation against official race results.
- Automated anti-leakage audits verify chronological data integrity before model training.

### 3. FIA-Aware Starting Grid Construction
- The engine explicitly distinguishes `QualifyingPosition` from `ActualStartingGrid`.
- Sporting grid penalties are resolved prior to race simulation:
  - Cumulative place drops (+3, +5, +10, +15).
  - Back-of-grid penalties (+30 / power unit changes).
  - Ordered cascade re-alignment of unaffected cars.
- Validated on historical and live events (e.g. Sepang, Baku, Austin).

### 4. 37-Feature Engineering System
The feature vector combines:
- Starting grid position and grid normalization
- Grid-to-race position conversion dynamics
- Qualifying speed and Q1/Q2/Q3 progression pace gaps
- FP1/FP2/FP3 telemetry and representative lap pace
- Driver Exponentially Weighted Moving Average (EWMA) form
- Historical and empirical driver DNF rates
- Constructor rolling pace and team points-scoring form
- Sprint qualifying and Sprint race finishing transitions
- Teammate qualifying pace gap and teammate grid delta
- Driver-vs-teammate head-to-head performance ratios

### 5. Random Forest + XGBoost Ensemble
- Dynamic model selection via chronological walk-forward cross-validation.
- Configurable ensemble blending (e.g. 60% Random Forest / 40% XGBoost).
- Out-of-fold performance comparison tracked by:
  - Mean Absolute Error (MAE)
  - Median Absolute Error (MedAE)
  - Root Mean Squared Error (RMSE)
  - Spearman Rank Correlation

### 6. 25,000-Iteration Monte Carlo Simulation
- Full distribution forecasting running 25,000 stochastic draws per driver.
- Uncertainty calibrated from walk-forward validation residuals.
- Empirical DNF hazard modeling with attrition simulation.
- Generates comprehensive probabilistic metrics:
  - Expected finishing position
  - Win probability
  - Podium probability (Top 3)
  - Top 5 probability
  - Points probability (Top 10)
  - P5–P95 uncertainty interval ranges

---



## Installation & Environment Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Internet connection (required only for initial FastF1 telemetry downloads)

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dineshdhayfule/F1-Race-Predictions.git
   cd F1-Race-Predictions
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Usage Guide

### 1. Interactive Terminal Menu (Recommended)

Launch the guided terminal interface without remembering CLI arguments:

```bash
python main.py
```
*(or explicitly: `python main.py menu`)*

The interactive menu provides 6 comprehensive workflows:
1. **Predict a Race**: Guided step-by-step prediction wizard with circuit selection, stage auditing, and simulation.
2. **Load / Collect Data**: Inspect dataset inventory, collect track practice telemetry or season rounds via FastF1, and audit schema integrity.
3. **View Previous Predictions & Reports**: Browse generated output folders, inspect prediction tables, and paginate markdown reports.
4. **Evaluate a Prediction**: Score saved forecast CSVs against official race results with duplicate/mismatch safeguards.
5. **System Health & Data Readiness**: Environment diagnostic center, dependency validation, and quick test execution.
6. **Configuration & Supported Circuits**: Inspect model hyperparameters and browse all 23 circuits on the 2026 calendar.

---

### 2. Command Line Interface (CLI)

The engine retains full backward-compatible CLI command dispatching:

#### Predict a Grand Prix
```bash
# Predict Austin 2026 after qualifying (Top 10 display)
python main.py predict --track austin --year 2026 --stage after_qualifying --top 10

# Predict pre-FP1 rolling form for Mexico City (Podium Top 3)
python main.py predict --track mexico --year 2026 --stage pre_fp1 --top 3

# Predict Sepang 2026 full grid (Top 22)
python main.py predict --track sepang --year 2026 --stage after_qualifying --top 22
```

#### Run Chronological Model Evaluation
```bash
python main.py evaluate --track sepang --year 2026
```

#### Validate Dataset & Anti-Leakage
```bash
python main.py validate --track baku --year 2026 --stage after_qualifying
```

---

### 3. Data Collection

Collect race results and telemetry directly from FastF1:

```bash
# Collect season race and qualifying results (e.g. Round 18)
python -m src.data.collect_season --year 2026 --rounds 18

# Collect weekend practice and qualifying telemetry for a specific circuit
python -m src.data.collect_track --track austin --year 2026
```

---

## Testing & Quality Assurance

The test suite contains 14 specialized test modules with zero failures:

```bash
# Run the entire test suite
pytest

# Run fast unit and feature tests
pytest tests/test_leakage.py tests/test_sprint_architecture.py tests/test_config_driven_circuits.py tests/test_grid_conversion.py tests/test_grid_penalties.py tests/test_sprint_features.py tests/test_teammate_features.py tests/test_timing.py

# Run interactive menu and CLI dispatch tests
pytest tests/test_interactive_menu.py

# Run COTA session discovery and early stage validation
pytest tests/test_cota_session_discovery.py

# Run baseline parity regression tests (Sepang, Baku, Singapore)
pytest tests/test_parity_regression.py

# Run circuit registry integrity verification
pytest tests/test_circuit_integrity.py
```

---

## License

This project is licensed under the [MIT License](LICENSE).