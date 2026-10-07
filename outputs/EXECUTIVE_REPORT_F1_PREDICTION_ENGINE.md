# EXECUTIVE REPORT: F1 UNIFIED RACE PREDICTION PLATFORM & 2026 SEPANG PRE-RACE FORECAST

**Date**: October 4, 2026  
**Subject**: Architecture Unification, Anti-Leakage Audit, and Pre-Race Prediction Delivery  
**Target Event**: Formula 1 2026 Malaysian Grand Prix (Round 16, Sepang International Circuit)  
**Status**: Verified & Ready for Production Dispatch  

---

## 1. Executive Summary

We have completed the architecture unification of the Formula 1 Race Prediction Platform. Previously fragmented, track-specific prediction scripts (`f1-baku-prediction` and `sepang race prediction`) have been successfully synthesized into a single, modular engine: **`f1-race-prediction/`**.

### Key Deliverables:
1. **Config-Driven Architecture**: Any F1 circuit (e.g., Sepang, Baku, Monza, Silverstone) can now be loaded, evaluated, and predicted via `config/tracks.json` with **zero code modifications**.
2. **Guaranteed Zero Data Leakage**: Target race results (`ActualFinish`, `Position`, `Points`, `Status`) are strictly quarantined and prohibited from model training and target feature matrices. Form EWMA metrics strictly enforce `Round < target_round`.
3. **FIA Sporting Code Starting Grid**: Automated grid construction accurately accounts for back-of-grid relegations and grid drop shifts (e.g., Hadjar drops from Q3 to P8; Colapinto and Lindblad start P21 and P22).
4. **25,000-Iteration Monte Carlo Simulation**: Replaced static pace noise with **out-of-fold walk-forward residual calibration** ($\sigma = 4.2676$) and empirical season DNF risk ($15.77\%$).
5. **Validation & Quality Assurance**: All 10 automated unit test suites passed with 100% compliance. Legacy directories remain untouched and functional.

---

## 2. 2026 Sepang Grand Prix — Pre-Race Forecast

*Forecast generated strictly post-qualifying prior to the Grand Prix race start.*

### Starting Grid Penalties Applied:
- **HAD (Isack Hadjar)**: +5 grid positions (Qualifying: P3 $\rightarrow$ **Starting Grid: P8**)
- **COL (Franco Colapinto)**: +15 grid positions / Back of Grid $\rightarrow$ **Starting Grid: P21**
- **LIN (Arvid Lindblad)**: +30 grid positions / Back of Grid $\rightarrow$ **Starting Grid: P22**
- *Beneficiaries*: ANT (P3), LEC (P4), NOR (P5), PIA (P6), and RUS (P7) shift up one row to fill vacated starting boxes.

### Official Finishing Order Forecast & Probabilities (25,000 Simulations):

| Pred. Pos | Driver | Team | Grid Box | Expected Finish | Median Finish | Win % | Podium % (Top 3) | Top 5 % | Points % (Top 10) | P5–P95 Interval |
|:---------:|:------:|:-----|:--------:|:---------------:|:-------------:|:-----:|:----------------:|:-------:|:-----------------:|:---------------:|
| **1** | **VER** | Red Bull Racing | 1 | **4.73** | 3 | **27.2%** | **56.7%** | 72.5% | 89.2% | P1 – P18 |
| **2** | **HAM** | Ferrari | 2 | **5.77** | 4 | **16.9%** | **43.7%** | 61.9% | 85.3% | P1 – P19 |
| **3** | **ANT** | Mercedes | 3 | **6.01** | 4 | **14.7%** | **40.7%** | 59.5% | 84.5% | P1 – P19 |
| **4** | **PIA** | McLaren | 6 | **6.82** | 5 | **10.1%** | **31.9%** | 50.7% | 80.6% | P1 – P19 |
| **5** | **LEC** | Ferrari | 4 | **7.17** | 6 | **8.8%** | **29.0%** | 47.7% | 78.6% | P1 – P20 |
| **6** | **RUS** | Mercedes | 7 | **7.65** | 6 | **7.0%** | **24.5%** | 42.8% | 75.8% | P1 – P20 |
| **7** | **NOR** | McLaren | 5 | **8.55** | 7 | **4.5%** | **17.9%** | 34.0% | 69.3% | P2 – P20 |
| **8** | **HAD** | Red Bull Racing | 8 | **8.83** | 8 | **3.9%** | **16.3%** | 31.7% | 67.3% | P2 – P20 |
| **9** | **GAS** | Alpine | 9 | **10.25** | 10 | **1.8%** | **9.3%** | 21.1% | 56.1% | P3 – P21 |
| **10** | **BOR** | Audi | 10 | **10.60** | 10 | **1.6%** | **8.2%** | 18.7% | 52.9% | P3 – P21 |
| **11** | **LAW** | Racing Bulls | 11 | **10.88** | 10 | **1.4%** | **7.6%** | 17.2% | 50.4% | P3 – P21 |
| **12** | **ALO** | Aston Martin | 12 | **12.03** | 12 | **0.7%** | **4.4%** | 11.3% | 41.3% | P4 – P21 |
| **13** | **SAI** | Williams | 13 | **13.19** | 13 | **0.3%** | **2.5%** | 7.2% | 30.8% | P5 – P21 |
| **14** | **HUL** | Audi | 15 | **13.51** | 14 | **0.3%** | **2.2%** | 6.3% | 28.6% | P5 – P21 |
| **15** | **STR** | Aston Martin | 14 | **14.13** | 14 | **0.2%** | **1.5%** | 4.6% | 23.8% | P6 – P21 |
| **16** | **COL** | Alpine | 21 | **14.92** | 15 | **0.1%** | **1.0%** | 3.1% | 18.4% | P7 – P22 |
| **17** | **OCO** | Haas F1 Team | 17 | **14.95** | 15 | **0.1%** | **0.9%** | 3.1% | 18.3% | P7 – P22 |
| **18** | **LIN** | Racing Bulls | 22 | **15.38** | 16 | **0.1%** | **0.8%** | 2.4% | 15.6% | P7 – P22 |
| **19** | **BEA** | Haas F1 Team | 16 | **15.50** | 16 | **0.1%** | **0.6%** | 2.2% | 14.8% | P7 – P22 |
| **20** | **ALB** | Williams | 18 | **16.91** | 18 | <0.1% | 0.2% | 0.8% | 7.8% | P9 – P22 |
| **21** | **BOT** | Cadillac | 19 | **17.09** | 18 | <0.1% | 0.2% | 0.8% | 7.1% | P9 – P22 |
| **22** | **PER** | Cadillac | 20 | **18.13** | 19 | <0.1% | 0.1% | 0.3% | 3.6% | P11 – P22 |

---

## 3. Walk-Forward Cross-Validation & Model Selection

Model validation was conducted strictly out-of-fold across 2026 season rounds 4 through 15 (12 sequential folds; 264 predictions against actual race outcomes):

| Evaluated Architecture | MAE (Mean Abs Error) | MedAE | RMSE | Spearman Rank Correlation |
|:-----------------------|:-------------------:|:-----:|:----:|:-------------------------:|
| **100% Random Forest (Primary)** | **3.651** | **2.934** | **4.803** | **0.656** |
| 75% RF / 25% XGBoost | 3.637 | 2.948 | 4.814 | 0.656 |
| 60% RF / 40% XGBoost | 3.636 | 2.934 | 4.830 | 0.653 |
| 50% RF / 50% XGBoost | 3.638 | 2.911 | 4.843 | 0.651 |
| 40% RF / 60% XGBoost | 3.643 | 2.848 | 4.859 | 0.649 |
| 25% RF / 75% XGBoost | 3.655 | 2.825 | 4.888 | 0.646 |
| 100% XGBoost | 3.685 | 2.766 | 4.949 | 0.639 |

### Technical Decision:
- **Random Forest** achieved the lowest RMSE (4.803) and highest Spearman rank correlation (0.656).
- **Residual Standard Deviation**: Derived empirical variance of $\sigma = 4.2676$ with Median Absolute Deviation (MAD) scaling, avoiding arbitrary pace assumptions.
- **Empirical Season DNF Hazard**: Computed at **15.77%** across rounds 1–15, applied directly in the stochastic simulation loop.

---

## 4. Multi-Track Architecture Parity Verification

To prove cross-circuit generality, the unified platform was tested against two distinct historical track contexts:

### Case 1: Malaysian GP (Sepang — Round 16)
- **Circuit Context**: Hiatus from 2018–2025; historical data flag set to `false`.
- **Pipeline Behavior**: Relies strictly on 2026 rolling season form and Sepang FP1/FP2/FP3/Qualifying telemetry.
- **Result**: Complete 22-driver probabilistic matrix with Hadjar P8 penalty integration.

### Case 2: Azerbaijan GP (Baku — Round 15)
- **Circuit Context**: Continuous hybrid-era data (2021–2025); historical data flag set to `true`.
- **Pipeline Behavior**: Automatically restricts training data to rounds 1–14; Baku race result quarantined; George Russell correctly forecast as P1 pole victor (42.0% win probability).
- **Result**: Exact parity with legacy Baku forecasts without custom scripts.

### Case 3: Italian GP (Monza — Extensibility Test)
- Added to `config/tracks.json` as a high-speed circuit. Resolved dynamically by `DataLoader` without writing a single line of Python.

---

## 5. CLI Quick-Reference Guide

The platform is operated via command line:

```bash
# 1. Run Race Prediction for any configured circuit
python main.py predict --track sepang --year 2026 --stage after_qualifying
python main.py predict --track baku --year 2026 --stage after_qualifying

# 2. Run Chronological Model Evaluation across season rounds
python main.py evaluate --track sepang --year 2026

# 3. Audit Datasets for Data Integrity & Zero Leakage
python main.py validate --track sepang --year 2026

# 4. Run Automated Test Suite
python -m pytest tests/ -v
```

---

## 6. Sign-off & Audit Checklist

- [x] **Zero Post-Race Leakage**: Pre-race datasets verified free of finishing outcomes.
- [x] **Starting Grid Legitimacy**: 22 unique drivers; positions strictly 1–22.
- [x] **Penalties Verified**: HAD P8, COL P21, LIN P22 confirmed in outputs.
- [x] **Simulation Convergence**: 25,000 iterations completed in < 1 second.
- [x] **Non-Destructive Migration**: Original `f1-baku-prediction` and `sepang race prediction` repositories remain preserved and 100% functional.

*Platform developed and certified by the Advanced Agentic Engineering Team.*
