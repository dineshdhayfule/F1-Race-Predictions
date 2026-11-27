<div align="center">

# ApexGrid

**Formula 1 race outcome prediction engine using FastF1 telemetry, a calibrated ML ensemble, and 25,000-iteration Monte Carlo simulation.**

[![PyPI Version](https://img.shields.io/pypi/v/apexgrid.svg?color=blue)](https://pypi.org/project/apexgrid/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-197%20passed-brightgreen.svg)](tests/)

</div>

---

ApexGrid forecasts finishing positions, win probabilities, and podium chances across all 23 Formula 1 championship rounds. It ingests practice and qualifying telemetry via FastF1 and OpenF1, builds 40 configured race-pace and degradation features (stage-gated from 15 at `pre_fp1` up to 40 at `after_qualifying`), blends a 3-way ML ensemble (Random Forest, XGBoost, Ridge), and runs 25,000 Monte Carlo race simulations under strict stage-gated temporal cutoffs.

```
FastF1 / OpenF1 ──► Feature Engineering ──► ML Ensemble ──► 25,000 Monte Carlo ──► Predictions
  (FP1–Quali)       (40 features, gated)   (RF+XGB+Ridge)    (DNF, SC, weather)     (Win%, Podium%, EF)
```

---

## Contents

- [Capabilities](#capabilities)
- [Installation](#installation)
- [CLI Reference](#cli-reference)
- [Interactive Menu](#interactive-menu)
- [Methodology](#methodology)
- [Configuration](#configuration)
- [Testing](#testing)
- [License](#license)

---

## Capabilities

- **3-Way ML Ensemble:** 40% Random Forest + 30% XGBoost + 30% Ridge Regression, calibrated via chronological walk-forward validation. Supports 2-way hybrid (`60/40` RF/XGB) and custom weight splits.
- **Circuit-Aware Grid Blending:** Adjusts grid vs. ML weights by circuit topology: street (`50/50`), permanent (`35/65`), temporary (`45/55`).
- **Stage-Gated Anti-Leakage:** Seven temporal cutoffs (`pre_fp1` through `after_qualifying`) block future-session qualifying or race data from leaking into earlier forecasts.
- **25,000-Run Monte Carlo:** Residual-calibrated pace variance, empirical DNF hazard draws, wet-skill scaling, and venue-specific Safety Car disruption priors.
- **Walk-Forward Validation:** Expanding-window chronological validation (`{1…r−1} → r`). Shuffled k-fold is disabled.
- **Historical Analytics (1950–2026):** 11 CLI modules covering driver stats, teammate H2H, qualifying, race craft, circuits, standings, sessions, dominance, records, reliability, and eras.

---

## Installation

### End Users (PyPI)

Install the package:

```bash
pip install apexgrid
```

Upgrade to the latest release:

```bash
pip install --upgrade apexgrid
```

Launch ApexGrid:

```bash
apexgrid
```

Bundled configurations load from the package; caches and outputs are stored in `~/.apexgrid/`.

### Developers (Editable)

```bash
git clone https://github.com/dineshdhayfule/F1-Race-Predictions.git
cd F1-Race-Predictions
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Both `apexgrid` and `python main.py` work inside the cloned repository.

---

## CLI Reference

### Predict

```bash
apexgrid predict --track monza --year 2026 --stage pre_fp1 --model ensemble
apexgrid predict --track silverstone --year 2026 --stage after_qualifying
apexgrid predict --track monaco --year 2026 --stage after_fp3 --model ensemble --enhanced
apexgrid predict --track baku --year 2026 --stage after_qualifying --model hybrid --rf-weight 70 --xgb-weight 30
apexgrid circuits
```

### Validate & Evaluate

```bash
apexgrid evaluate --track sepang --year 2026
apexgrid validate --track baku --year 2026 --stage after_qualifying
```

### History

```bash
apexgrid history
```

Subcommands: `driver` · `h2h` · `qualifying` · `race` · `circuit` · `standings` · `session` · `dominance` · `records` · `reliability` · `eras`

### Data & Cache

```bash
apexgrid collect --year 2026
apexgrid clean --target cache --preview
apexgrid clean --target cache --force
```

---

## Interactive Menu

Run `apexgrid` (or `python main.py`) with no arguments to open the 8-option terminal menu:

| # | Workflow | Scope |
|---|----------|-------|
| 1 | **Predict a Race** | Circuit, stage, model, and simulation |
| 2 | **Load / Collect Data** | FastF1 practice, qualifying, and race ingestion |
| 3 | **Evaluate a Prediction** | Score forecasts against official classifications |
| 4 | **View Previous Predictions** | Inspect saved CSV tables and Markdown reports |
| 5 | **Historical Statistics** | 11 analytics modules (1950–2026) |
| 6 | **System Health** | Dependency, config, and dataset checks |
| 7 | **Configuration** | Circuit parameters and model settings |
| 8 | **Data Management** | Preview and prune cached files |

---

## Methodology

<details>
<summary><b>Mathematical Formulation & Validation Rules</b></summary>

### Stage-Gated Grid Blending

Post-qualifying runs blend official FIA starting grids with ML pace scores by circuit type:

$$
\text{Score}_{\text{blended}} = w_{\text{ml}} \cdot \text{Score}_{\text{ML}} + w_{\text{grid}} \cdot \text{Grid}_{\text{actual}}
$$

| Circuit Type | $w_{\text{grid}}$ | $w_{\text{ml}}$ | Examples |
|---|---|---|---|
| Street | 0.50 | 0.50 | Monaco, Baku, Singapore |
| Permanent | 0.35 | 0.65 | Spa, Monza, Silverstone |
| Temporary | 0.45 | 0.55 | Melbourne, Montreal |

Pre-qualifying runs mask grid data to `NaN` and apply a `0.5x` `provisional_grid_discount` to circuit grid weights when using `provisional_grid_position` (re-normalizing $w_{\text{ml}} = 1.0 - w_{\text{grid}}$, or disabling blending $w_{\text{grid}} = 0$, $w_{\text{ml}} = 1$ when no grid is available); verified pre-2018 starting grids use a `0.75 / 0.25` override.

### Residual-Calibrated Pace Volatility

Monte Carlo pace variance is calibrated from walk-forward residuals via Median Absolute Deviation:

$$
\sigma = \max\!\left(1.5,\; 1.4826 \cdot \text{MAD}(\text{residuals})\right)
$$

Sparse grids ($n < 20$) scale $\sigma$ proportionally down to a `1.20` floor, and constructors with $\ge 8$ out-of-fold residuals use shrunk team volatility $\text{clip}(0.70 \cdot \sigma_{\text{team}} + 0.30 \cdot \sigma_{\text{global}},\, 1.20,\, 1.5 \cdot \sigma_{\text{global}})$. Wet-weather draws scale volatility by `1.35×`, Safety Car probability by `1.40×`, and DNF hazard by `1.50×`.

### Decoupled Modeling & Ranking

- **Two-Stage Separation:** Regressors train only on classified finishers; DNF hazards are sampled via binomial draws in Monte Carlo.
- **EWMA Form:** Driver and constructor form uses $\alpha = 0.30$ exponential decay.
- **Dual-View Output:** Podium banner sorts by `Win%` ↓, `Top3%` ↓, `ExpectedFinish` ↑; full table sorts by DNF-adjusted `ExpectedFinish` (`+1000.0` penalty for non-finishers) alongside `ExpFinishClassified` (mean finish across non-DNF runs).
- **Walk-Forward Guard:** Any failed validation fold transitions status to `BLOCKED` and clears residual arrays.

</details>

---

## Configuration

| File | Purpose |
|------|---------|
| [`src/config/model_config.json`](src/config/model_config.json) | Ensemble weights, hyperparameters, feature sets, topology blending |
| [`src/config/tracks.json`](src/config/tracks.json) | 23 championship circuits, lengths, types, calendar rounds |
| [`src/config/strategy_config.json`](src/config/strategy_config.json) | Tire compounds, pit loss times, 23 circuit strategy priors |
| [`src/config/settings.json`](src/config/settings.json) | Cache paths, logging, output directories |

---

## Testing

**197 tests** across `23 test_*.py` files (`194` fast tests + `3` `@pytest.mark.slow` tests):

```bash
pytest -m "not slow" -q   # fast inner-loop suite
pytest -q                 # full suite
pytest tests/test_walk_forward.py
pytest tests/test_core_features.py
pytest tests/test_dynamic_roster.py
pytest tests/test_weekend_only_pipeline.py
```

---

## License

[MIT](LICENSE)
