"""
Unit Tests for Dynamic Model Selection & Simulation Order Integrity
===================================================================
Tests dynamic walk-forward ensemble selection, reporting metadata,
grid-order sorting before Monte Carlo RNG draws, and zero target race leakage.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import json

from src.models.evaluate import ModelEvaluator
from src.simulation.monte_carlo import MonteCarloSimulator
from src.models.predict import Predictor
from src.data.data_validator import DataValidator


@pytest.fixture
def sample_config():
    root = Path(__file__).resolve().parents[1]
    with open(root / "config" / "model_config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def test_dynamic_ensemble_selection(sample_config):
    """Verifies that evaluate_walk_forward tests all 7 configurations and selects the lowest MAE."""
    evaluator = ModelEvaluator(sample_config)

    # Synthetic multi-round dataset
    np.random.seed(42)
    rows = []
    features = sample_config["features"]

    for r in range(1, 8):
        for d in range(1, 23):
            row = {"Round": r, "ActualFinish": d, "driver_dnf_rate": 0.15}
            for f in features:
                row[f] = np.random.randn()
            rows.append(row)

    df = pd.DataFrame(rows)
    results = evaluator.evaluate_walk_forward(df)

    assert "ensemble_comparisons" in results
    comparisons = results["ensemble_comparisons"]
    assert len(comparisons) == 7

    tested_names = [c["model_name"] for c in comparisons]
    expected_names = [
        "100% RF",
        "75% RF / 25% XGB",
        "60% RF / 40% XGB",
        "50% RF / 50% XGB",
        "40% RF / 60% XGB",
        "25% RF / 75% XGB",
        "100% XGB"
    ]
    assert tested_names == expected_names

    # The winner must be the configuration with the minimum MAE
    min_mae_cfg = min(comparisons, key=lambda x: x["mae"])
    assert results["best_model_name"] == min_mae_cfg["model_name"]
    assert results["best_w_rf"] == min_mae_cfg["rf_weight"]
    assert results["best_w_xgb"] == min_mae_cfg["xgb_weight"]
    assert results["best_mae"] == min_mae_cfg["mae"]


def test_selected_weights_reporting(sample_config):
    """Verifies selected weights and metadata are propagated correctly."""
    evaluator = ModelEvaluator(sample_config)

    rows = []
    for r in range(1, 6):
        for d in range(1, 10):
            row = {"Round": r, "ActualFinish": d, "driver_dnf_rate": 0.15}
            for f in sample_config["features"]:
                row[f] = float(d) + np.random.randn() * 0.1
            rows.append(row)

    results = evaluator.evaluate_walk_forward(pd.DataFrame(rows))
    assert 0.0 <= results["best_w_rf"] <= 1.0
    assert 0.0 <= results["best_w_xgb"] <= 1.0
    assert round(results["best_w_rf"] + results["best_w_xgb"], 4) == 1.0
    assert len(results["selected_residuals"]) > 0


def test_grid_order_sorting_before_monte_carlo(sample_config):
    """
    Verifies that Monte Carlo simulation reproduces exact deterministic results
    when target dataframe is pre-sorted by ActualStartingGrid.
    """
    simulator = MonteCarloSimulator(sample_config)

    # Drivers in arbitrary qualifying/entry order
    drivers = ["HAD", "VER", "HAM", "LEC"]
    actual_grids = [8, 1, 2, 4]
    mle_scores = [9.5, 4.0, 5.0, 7.0]

    df_unsorted = pd.DataFrame({
        "Abbreviation": drivers,
        "ActualStartingGrid": actual_grids,
        "GridPosition": actual_grids,
        "QualifyingPosition": [3, 1, 2, 5],
        "TeamName": ["Red Bull", "Red Bull", "Ferrari", "Ferrari"]
    })

    # Sort strictly by ActualStartingGrid before RNG loop
    df_sorted = df_unsorted.sort_values("ActualStartingGrid").reset_index(drop=True)
    mle_sorted = np.array([4.0, 5.0, 7.0, 9.5])

    res1, meta1 = simulator.simulate(
        target_df=df_sorted,
        deterministic_scores=mle_sorted,
        residuals=np.array([0.5, -0.5, 0.2, -0.2]),
        empirical_dnf_rate=0.15
    )

    # Run again with identical seed - must match perfectly
    res2, meta2 = simulator.simulate(
        target_df=df_sorted,
        deterministic_scores=mle_sorted,
        residuals=np.array([0.5, -0.5, 0.2, -0.2]),
        empirical_dnf_rate=0.15
    )

    pd.testing.assert_frame_equal(res1, res2)
    assert res1.iloc[0]["Abbreviation"] == "VER"
    assert res1.iloc[1]["Abbreviation"] == "HAM"


def test_zero_target_race_leakage():
    """Verifies that target race data cannot contaminate training features or validation."""
    validator = DataValidator()

    train_df = pd.DataFrame({
        "Round": [1, 2, 3],
        "Abbreviation": ["VER", "VER", "VER"],
        "ActualFinish": [1, 2, 1],
        "GridPosition": [1, 2, 1]
    })

    target_df = pd.DataFrame({
        "Round": [4],
        "Abbreviation": ["VER"],
        "GridPosition": [1],
        "QualifyingPosition": [1]
    })

    # Legitimate pre-race target
    report = validator.audit_leakage(train_df, target_df, target_round=4)
    assert report["is_clean"] is True

    # Contaminated target with target round inside training
    train_leak = train_df.copy()
    train_leak.loc[len(train_leak)] = {"Round": 4, "Abbreviation": "VER", "ActualFinish": 1, "GridPosition": 1}
    report_leak = validator.audit_leakage(train_leak, target_df, target_round=4)
    assert report_leak["is_clean"] is False
    assert any("Target Round" in v for v in report_leak["violations"])
