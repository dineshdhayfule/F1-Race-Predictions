<div align="center">

# ApexGrid — Formula 1 Race Prediction & Motorsport Intelligence Engine

**ML-powered Formula 1 race predictions using FastF1 telemetry ingestion, a calibrated multi-model ensemble (Random Forest, XGBoost, Ridge Regression), and 25,000-iteration stochastic Monte Carlo simulation with stage-aware qualifying grid blending.**

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-183%20passed%20%E2%9C%93-brightgreen.svg)](tests/)
[![ML Ensemble](https://img.shields.io/badge/ensemble-RF%20%2B%20XGB%20%2B%20Ridge-orange.svg)](src/models/)
[![Monte Carlo](https://img.shields.io/badge/Monte%20Carlo-25%2C000%20runs-purple.svg)](src/simulation/)

</div>

---

ApexGrid is a production-grade Formula 1 race prediction platform that forecasts finishing positions, win probabilities, podium chances, and points finishes across all 23 championship rounds. It ingests real-time practice and qualifying telemetry via FastF1 and OpenF1, engineers 25+ predictive features (fuel-corrected pace, tire degradation curves, weather dynamics, safety car hazard priors), blends them through a calibrated multi-model ML ensemble of Random Forest, XGBoost, and Ridge Regression, and runs 25,000 stochastic Monte Carlo race simulations with empirical DNF hazard modeling — all behind strict stage-aware temporal boundaries that prevent future-session data leakage from qualifying or race results contaminating earlier predictions.

```
FastF1 / OpenF1 ──► Feature Engineering ──► ML Ensemble ──► 25,000 Monte Carlo ──► Predictions
  Telemetry            (25+ features)       (RF+XGB+Ridge)     (DNF, weather,      (Win%, Podium%,
  (FP1–Quali)                                                   SC hazards)         ExpectedFinish)
```

---

## Table of Contents

- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Interactive Menu](#interactive-menu)
- [CLI Usage Recipes](#cli-usage-recipes)
- [Technical Architecture](#technical-architecture)
- [Configuration](#configuration)
- [Verification & Testing](#verification--testing)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)
- [License](#license)

---

## Key Features

- **Calibrated 3-Way ML Ensemble** — Random Forest (40%) + XGBoost (30%) + Ridge Regression (30%), tuned via chronological walk-forward validation. Also supports a 2-way hybrid mode (60/40 RF/XGB) and fully custom weight splits.
- **Circuit-Aware Starting-Grid Blending** — Dynamically adjusts grid vs. ML weighting by circuit topology: street circuits (Monaco, Baku, Singapore) at 50/50, permanent tracks (Spa, Monza, Silverstone) at 35/65, temporary venues (Melbourne, Montreal) at 45/55.
- **Stage-Gated Anti-Leakage** — Seven temporal stages (`pre_fp1`, `after_fp1`, `after_fp2`, `after_fp3`, `after_sprint_qualifying`, `after_sprint_race`, `after_qualifying`) strictly prevent qualifying or race data from contaminating earlier-stage predictions.
- **25,000-Run Monte Carlo Engine** — Stochastic race simulation with residual-calibrated pace variance, empirical DNF hazard modeling, wet-skill differentials, field compression dynamics, and venue-specific safety car disruption probabilities.
- **Zero-Leakage Walk-Forward Validation** — Shuffled k-fold is strictly prohibited. Models train on rounds {1…r−1} and test on out-of-sample round r, with blocked residual arrays on any fold failure.
- **Historical Intelligence (1950–2026)** — 11 analytics modules: driver career stats, teammate H2H, qualifying performance, race craft, circuit records, championship standings, session results, constructor dominance, all-time records, reliability analysis, and regulatory era comparisons.

---

## Quick Start

```bash
git clone https://github.com/dineshdhayfule/F1-Race-Predictions.git
cd F1-Race-Predictions
python -m venv .venv
```

Activate the virtual environment:

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install dependencies and launch:

```bash
pip install -r requirements.txt
pip install -e .              # enables the `apexgrid` CLI binary
```

```bash
python main.py                # argparse entrypoint
# or
apexgrid                      # Typer entrypoint (requires editable install)
```

---

## Interactive Menu

`python main.py` launches a guided Rich terminal interface with eight workflows:

| # | Option | Description |
|---|--------|-------------|
| 1 | **Predict a Race** | Circuit → weekend stage → model selection → forecast |
| 2 | **Load / Collect Data** | Ingest FastF1 practice, qualifying, and race telemetry |
| 3 | **Evaluate a Prediction** | Score forecast accuracy against official classifications |
| 4 | **View Previous Predictions** | Browse saved prediction tables and Markdown reports |
| 5 | **Historical Statistics** | 11 analytics modules spanning 1950–2026 |
| 6 | **System Health** | Diagnostic checks and data readiness audit |
| 7 | **Configuration** | View circuit parameters, track types, model settings |
| 8 | **Data Management** | Preview and prune cached telemetry files safely |

---

## CLI Usage Recipes

### Race Predictions

```bash
# Full 3-way ensemble, pre-qualifying stage
python main.py predict --track monza --year 2026 --stage pre_fp1 --model ensemble

# Default hybrid (60% RF / 40% XGB), post-qualifying
python main.py predict --track silverstone --year 2026 --stage after_qualifying

# Enhanced physics (race pace, tire deg, pit strategy, weather, race control)
python main.py predict --track monaco --year 2026 --stage after_fp3 --model ensemble --enhanced

# Custom ensemble weights (70% RF / 30% XGB)
python main.py predict --track baku --year 2026 --stage after_qualifying --model hybrid --rf-weight 70 --xgb-weight 30
```

### Walk-Forward Validation

```bash
python main.py evaluate --track sepang --year 2026
python main.py validate --track baku --year 2026 --stage after_qualifying
```

### Historical Analytics

```bash
python main.py history
```

Subcommands: `driver` · `h2h` · `qualifying` · `race` · `circuit` · `standings` · `session` · `dominance` · `records` · `reliability` · `eras`

### Data Collection & Cleanup

```bash
python main.py collect-season --year 2026       # Full season ingestion
python main.py clean --target cache --preview    # Dry-run preview
python main.py clean --target cache --force      # Execute cleanup
```

---

## Technical Architecture

<details>
<summary><b>Technical Methodology & Mathematical Formulation</b></summary>

### Stage-Gated Grid Blending

Starting grid positions heavily influence race outcomes, but using qualifying data before qualifying introduces catastrophic lookahead bias. ApexGrid resolves this with stage-aware gating:

**Post-Qualifying:** Official FIA grid positions (including penalty cascades and pit-lane starts) blend with pure ML pace scores using circuit-topology weights:


$$
\text{Score}_{\text{blended}} = w_{\text{ml}} \cdot \text{Score}_{\text{ML}} + w_{\text{grid}} \cdot \text{Grid}_{\text{actual}}
$$


| Circuit Type | $w_{\text{grid}}$ | $w_{\text{ml}}$ | Examples |
|---|---|---|---|
| Street | 0.50 | 0.50 | Monaco, Baku, Singapore |
| Permanent | 0.35 | 0.65 | Spa, Monza, Silverstone |
| Temporary | 0.45 | 0.55 | Melbourne, Montreal |

**Pre-Qualifying:** Grid data is masked to NaN. The pipeline generates provisional qualifying forecasts (displayed as `~P1`, `~P2`, etc.) or cleanly disables blending ($w_{\text{grid}} = 0$, $w_{\text{ml}} = 1$).

---

### Dual-View Outcome Presentation

Monte Carlo simulations produce full finishing distributions, presented in two views:

- **Podium Banner:** Top 3 drivers selected by Win% ↓, then Podium% ↓, then ExpectedFinish ↑ (tie-breaker).
- **Full Field Table:** All drivers ranked by DNF-adjusted ExpectedFinish. Non-finishers receive a +1000.0 penalty, ensuring they rank behind all classified finishers.

---

### Walk-Forward Validation

Shuffled k-fold is strictly prohibited. For each validation round $r > \text{min\_train\_rounds}$, models train exclusively on rounds $\{1 \dots r{-}1\}$ and test on out-of-sample round $r$. If any fold fails, validation transitions to `BLOCKED` and all residual arrays are cleared to prevent partial contamination.

---

### Residual-Calibrated Pace Volatility

Monte Carlo variance is empirically calibrated from walk-forward residuals using Median Absolute Deviation:


$$
\sigma = \max\!\left(1.5,\; 1.4826 \cdot \text{MAD}(\text{residuals})\right)
$$


Under wet-weather sampling: volatility scales by 1.35×, DNF hazard scales by 1.40×.

---

### Decoupled Two-Stage Modeling

ML regressors (RF, XGBoost, Ridge) train exclusively on classified finishers — mechanical retirements and lap-1 incidents are excluded so pace signal stays uncorrupted. DNF hazards are modeled separately via binomial draws during Monte Carlo simulation.

---

### EWMA Form Tracking

Driver and constructor rolling form uses exponentially weighted moving average with decay $\alpha = 0.30$, heavily weighting the most recent 3–5 rounds to capture mid-season aerodynamic upgrades while maintaining baseline stability.

</details>

---

## Configuration

JSON configuration files in [`config/`](config/):

| File | Purpose |
|------|---------|
| [`model_config.json`](config/model_config.json) | Ensemble weights, hyperparameters, feature sets, circuit blending weights |
| [`tracks.json`](config/tracks.json) | 23 championship circuits — lengths, types, calendar mappings |
| [`strategy_config.json`](config/strategy_config.json) | Tire compounds, pit loss times, 15 circuit-specific strategy priors |
| [`settings.json`](config/settings.json) | Cache paths, logging thresholds, output directories |

---

## Verification & Testing

**183 tests** across 21 modules — 100% pass rate:

```bash
pytest                                       # Full suite
pytest tests/test_walk_forward.py            # Walk-forward validation
pytest tests/test_core_features.py           # Stage isolation & anti-leakage
pytest tests/test_dynamic_roster.py          # Driver substitution handling
pytest tests/test_weekend_only_pipeline.py   # Weekend-only ML pipeline
```

---

## Tech Stack

| Component | Technologies |
|-----------|-------------|
| **Language** | Python 3.10 – 3.13 |
| **Machine Learning** | Scikit-Learn (Random Forest, Ridge), XGBoost |
| **Scientific** | NumPy, SciPy, Pandas |
| **Telemetry** | FastF1, OpenF1 REST API |
| **CLI / TUI** | Typer, Rich |
| **Testing** | Pytest |

---

## Contributing

1. Fork the repository and create a feature branch
2. Add tests matching existing conventions in [`tests/`](tests/)
3. Confirm `pytest` exits with code 0 (zero regressions)
4. Open a PR against `main` with verification logs

---

## License

This project is licensed under the [MIT License](LICENSE).
