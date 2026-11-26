<div align="center">

# ApexGrid

**Predicting the Race Beyond the Grid**

[![Python](https://img.shields.io/badge/python-3.10–3.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-183%20passed-brightgreen.svg)](tests/)
[![Monte Carlo](https://img.shields.io/badge/Monte%20Carlo-25%2C000%20runs-purple.svg)](src/simulation/)

</div>

An ML-powered Formula 1 race prediction platform that blends multi-model ensembling, circuit-aware grid weighting, and 25,000-iteration Monte Carlo simulation to forecast finishing positions, win probabilities, and podium chances across all 23 championship rounds — without future-session data leakage.

```
FastF1 Telemetry ──► Feature Engineering ──► Multi-Model Ensemble ──► 25k Monte Carlo ──► Predictions
  (FP1–FP3, Quali)      (60+ features)       (RF + XGB + Ridge)       (DNF hazards,       (Win%, Podium%,
                                                                        weather, SC)        ExpectedFinish)
```

---

## Highlights

- **3-Way Ensemble** — Random Forest (40%) + XGBoost (30%) + Ridge (30%), calibrated via chronological walk-forward validation. Also supports 2-way hybrid (60/40 RF/XGB) and custom weight splits.
- **Circuit-Aware Grid Blending** — Street circuits (Monaco, Baku) weight the starting grid at 50%; permanent tracks (Spa, Monza) at 35%; temporary venues (Melbourne, Montreal) at 45%.
- **Stage-Gated Anti-Leakage** — Seven temporal stages (`pre_fp1` → `after_qualifying`) strictly prevent qualifying or race data from contaminating earlier predictions.
- **Weekend-Only Mode** — Run the full ensemble using only current-weekend telemetry, with zero dependency on prior-season historical form.
- **25k Monte Carlo** — Stochastic simulation with residual-calibrated variance, empirical DNF modeling, wet-skill differentials, and safety car disruption.
- **Historical Intelligence** — 11 analytics modules covering 1950–2026: driver stats, H2H, qualifying, race craft, circuit records, standings, session results, dominance, records, reliability, and era comparisons.

---

## Quick Start

```bash
# Clone & set up
git clone https://github.com/dineshdhayfule/F1-Race-Predictions.git
cd F1-Race-Predictions
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Launch the interactive menu:

```bash
python main.py
```

---

## CLI Usage

### Predict a Race

```bash
# Full 3-way ensemble, pre-qualifying stage
python main.py predict --track monza --year 2026 --stage pre_fp1 --model ensemble

# Hybrid 2-way (default model), after qualifying
python main.py predict --track silverstone --year 2026 --stage after_qualifying

# Weekend-only telemetry, enhanced physics
python main.py predict --track monaco --year 2026 --stage after_fp3 --enhanced
```

### Evaluate & Validate

```bash
# Walk-forward cross-validation
python main.py evaluate --track sepang --year 2026

# Check anti-leakage boundaries
python main.py validate --track baku --year 2026 --stage after_qualifying
```

### Historical Analytics

```bash
python main.py history
```

11 subcommands: `driver`, `h2h`, `qualifying`, `race`, `circuit`, `standings`, `session`, `dominance`, `records`, `reliability`, `eras`.

### Data & Maintenance

```bash
# Collect a full season
python main.py collect-season --year 2026

# Preview cache before deleting
python main.py clean --target cache --preview
```

---

## Interactive Menu

`python main.py` opens a guided TUI with eight options:

| # | Option | Description |
|---|--------|-------------|
| 1 | Predict a Race | Circuit selection → stage → model → forecast |
| 2 | Load / Collect Data | Ingest FastF1 telemetry for sessions or full seasons |
| 3 | Evaluate a Prediction | Score accuracy against official classifications |
| 4 | View Previous Predictions | Browse saved prediction tables and reports |
| 5 | Historical Statistics | 11 analytics modules spanning 1950–2026 |
| 6 | System Health | Diagnostic checks and data readiness audit |
| 7 | Configuration | View circuit parameters and model settings |
| 8 | Data Management | Preview and prune cached files safely |

---

<details>
<summary><strong>Architecture & Methodology</strong></summary>

### Stage-Gated Grid Blending

Starting grid positions heavily influence race outcomes, but using qualifying data before qualifying introduces lookahead bias. ApexGrid resolves this with stage-aware gating:

- **Post-Qualifying:** Official grid positions blend with ML pace scores using circuit-topology weights:

$$\text{Score}_{\text{blended}} = w_{\text{ml}} \cdot \text{Score}_{\text{ML}} + w_{\text{grid}} \cdot \text{Grid}$$

- **Pre-Qualifying:** Grid data is masked. The pipeline generates provisional qualifying forecasts (`~P1`, `~P2`, etc.) or cleanly disables blending ($w_{\text{grid}} = 0$).

### Dual-View Presentation

- **Podium Banner:** Top 3 by Win% descending → Podium% descending → ExpectedFinish ascending.
- **Full Field Table:** All drivers ranked by DNF-adjusted ExpectedFinish (non-finishers receive a +1000 penalty).

### Walk-Forward Validation

Shuffled k-fold is strictly prohibited. Models train on rounds $\{1 \dots r{-}1\}$ and test on out-of-sample round $r$. Residual-calibrated pace volatility:

$$\sigma = \max\!\left(1.5,\; 1.4826 \cdot \text{MAD}(\text{residuals})\right)$$

Under wet conditions: volatility scales 1.35×, DNF hazard scales 1.40×.

### EWMA Form Tracking

Driver and constructor rolling form uses $\alpha = 0.30$ exponential decay, weighting the most recent 3–5 rounds to capture mid-season upgrades.

### Decoupled Two-Stage Modeling

ML regressors train exclusively on classified finishers (no mechanical retirements polluting pace signal). DNF hazards are modeled separately via binomial draws in Monte Carlo.

</details>

---

## Configuration

JSON config files in [`config/`](config/):

| File | Purpose |
|------|---------|
| [`model_config.json`](config/model_config.json) | Ensemble weights, hyperparameters, feature sets, circuit blending |
| [`tracks.json`](config/tracks.json) | 23 circuits — track lengths, types, calendar mappings |
| [`strategy_config.json`](config/strategy_config.json) | Tire compounds, pit loss times, 15 circuit-specific priors |
| [`settings.json`](config/settings.json) | Cache paths, logging, output directories |

---

## Testing

**183 tests** across 21 modules — 100% pass rate:

```bash
pytest                                    # Full suite
pytest tests/test_walk_forward.py         # Walk-forward validation
pytest tests/test_core_features.py        # Stage isolation & anti-leakage
pytest tests/test_dynamic_roster.py       # Driver substitution handling
pytest tests/test_weekend_only_pipeline.py  # Weekend-only mode
```

---

## Tech Stack

| Component | Technologies |
|-----------|-------------|
| Language | Python 3.10 – 3.13 |
| ML | Scikit-Learn (RF, Ridge), XGBoost |
| Scientific | NumPy, SciPy, Pandas |
| Telemetry | FastF1, OpenF1 API |
| CLI / TUI | Typer, Rich |
| Testing | Pytest |

---

## Contributing

1. Fork the repo and create a feature branch
2. Add tests matching existing conventions
3. Confirm `pytest` exits with code 0
4. Open a PR against `main`

---

## License

This project is licensed under the [MIT License](LICENSE).
