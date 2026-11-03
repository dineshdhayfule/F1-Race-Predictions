"""
Benchmark Experiment Configuration.
Centralizes verified production parameters, feature sets, and experimental settings.
"""

from pathlib import Path
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data" / "raw"
OUTPUTS_DIR = ROOT_DIR / "outputs"
REPORTS_DIR = ROOT_DIR / "reports" / "experiments" / "model_benchmark"

RANDOM_SEED = 42

# Verified Production Weights
DEFAULT_RF_WEIGHT = 0.60
DEFAULT_XGB_WEIGHT = 0.40
DEFAULT_ML_WEIGHT = 0.65
DEFAULT_GRID_WEIGHT = 0.35

# Ridge Reference Parameter
RIDGE_ALPHA = 40.0

# Benchmark Stage
BENCHMARK_STAGE = "after_qualifying"

# Minimum training rounds prior to walk-forward validation
MIN_TRAIN_ROUNDS_2026 = 3
MIN_TRAIN_ROUNDS_HISTORICAL = 3

# Baseline Feature List (37 features)
BASELINE_FEATURES = [
    "GridPosition",
    "grid_norm",
    "expected_grid_finish",
    "grid_retention_prob",
    "QualifyingPosition",
    "quali_pos_norm",
    "quali_gap_pct",
    "quali_no_time",
    "q1_gap_pct",
    "q2_gap_pct",
    "q3_gap_pct",
    "practice_gap_pct",
    "practice_rank",
    "practice_laps_norm",
    "fp1_gap_pct",
    "fp2_gap_pct",
    "fp3_gap_pct",
    "driver_finish_ewma",
    "driver_quali_ewma",
    "driver_dnf_rate",
    "recent_finish",
    "recent_grid",
    "recent_points",
    "avg_finish",
    "avg_grid",
    "total_points",
    "races_completed",
    "team_finish_ewma",
    "team_points_rate",
    "team_avg_finish",
    "team_points",
    "sprint_quali_pos",
    "sprint_finish_pos",
    "sprint_pos_delta",
    "teammate_quali_gap_pct",
    "teammate_grid_delta",
    "driver_teammate_h2h_ratio"
]

# Random Forest Verified Hyperparameters
RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "min_samples_leaf": 2,
    "random_state": RANDOM_SEED,
    "n_jobs": -1
}

# XGBoost Verified Hyperparameters
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "objective": "reg:squarederror"
}

# Monte Carlo Verified Settings
MC_PARAMS = {
    "default_simulations": 25000,
    "fallback_sigma": 2.85,
    "fallback_dnf_prob": 0.158,
    "random_seed": RANDOM_SEED
}
