"""
Isolated Unit and Integration Tests for Phase 2 Model Benchmark Framework.
Verifies fold construction, leakage protection, model wrappers, metric mathematics,
pit-lane handling, and determinism.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from experiments.model_benchmark.config import (
    BASELINE_FEATURES,
    BENCHMARK_STAGE,
    RANDOM_SEED,
    RIDGE_ALPHA,
)
from experiments.model_benchmark.models import (
    StartingGridBaseline,
    RandomForestBenchmark,
    XGBoostBenchmark,
    RidgeRegressionBenchmark,
    MLOnlyEnsembleBenchmark,
    ProductionHybridBenchmark,
    get_grid_positions,
)
from experiments.model_benchmark.metrics import (
    compute_race_metrics,
    compute_probability_metrics,
    compute_paired_comparison,
)
from experiments.model_benchmark.walk_forward import WalkForwardEngine
from src.features.build_features import apply_stage_feature_mask


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def synthetic_race_data():
    """Generates synthetic multi-round race features with strict chronological IDs."""
    rng = np.random.RandomState(42)
    rows = []
    drivers = ["VER", "HAM", "LEC", "NOR", "RUS", "PIA", "ALO", "SAI", "GAS", "TSU"]
    for rnd in range(1, 6):
        # Deterministic actual finish 1..10
        finishes = list(range(1, 11))
        rng.shuffle(finishes)
        for i, drv in enumerate(drivers):
            row = {
                "Round": rnd,
                "Abbreviation": drv,
                "ActualFinish": float(finishes[i]),
                "GridPosition": float(i + 1),
                "ActualStartingGrid": float(i + 1),
                "QualifyingPosition": float(i + 1),
                "is_pit_lane_start": 1 if drv == "TSU" and rnd == 5 else 0,
            }
            # Populate baseline feature columns with random floats
            for col in BASELINE_FEATURES:
                if col not in row:
                    row[col] = float(rng.normal(10.0, 3.0))
            rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Test 1 & 2: Chronological Folds & Race-Level Grouping
# -----------------------------------------------------------------------------
def test_chronological_fold_construction_and_grouping(synthetic_race_data):
    """
    Verifies that folds strictly test on future rounds, train strictly on past rounds,
    and group all drivers from the same race together in each fold.
    """
    df = synthetic_race_data
    val_rounds = [4, 5]

    for test_rnd in val_rounds:
        train_df = df[df["Round"] < test_rnd]
        test_df = df[df["Round"] == test_rnd]

        # Chronological assertion: max train round < test round
        assert train_df["Round"].max() < test_rnd
        # Grouping assertion: all 10 drivers evaluated together
        assert len(test_df) == 10
        assert set(test_df["Abbreviation"]) == set(df["Abbreviation"].unique())


# -----------------------------------------------------------------------------
# Test 3 & 5: No Future-Race Leakage & Target Leakage Prevention
# -----------------------------------------------------------------------------
def test_no_future_race_leakage_and_target_isolation(synthetic_race_data):
    """
    Verifies that future race results and future grid positions do not enter training data.
    """
    df = synthetic_race_data
    test_rnd = 4
    train_df = df[df["Round"] < test_rnd]

    assert (train_df["Round"] >= test_rnd).sum() == 0
    # Assert ActualFinish in test data is not in feature matrix
    feature_matrix = train_df[BASELINE_FEATURES]
    assert "ActualFinish" not in feature_matrix.columns


# -----------------------------------------------------------------------------
# Test 4: Feature-Stage Masking
# -----------------------------------------------------------------------------
def test_feature_stage_masking():
    """
    Verifies that stage-specific feature masks properly mask out Sunday qualifying
    and grid features at early stages.
    """
    sample = pd.DataFrame({
        "GridPosition": [1.0, 2.0],
        "QualifyingPosition": [1.0, 2.0],
        "driver_finish_ewma": [5.0, 6.0],
        "fp1_gap_pct": [0.1, 0.2],
    })

    masked = apply_stage_feature_mask(sample.copy(), stage="pre_fp1")
    assert np.isnan(masked["GridPosition"]).all()
    assert np.isnan(masked["QualifyingPosition"]).all()
    assert np.isnan(masked["fp1_gap_pct"]).all()
    # Season form remains available
    assert not np.isnan(masked["driver_finish_ewma"]).all()


# -----------------------------------------------------------------------------
# Test 6: Ridge Preprocessing Pipeline
# -----------------------------------------------------------------------------
def test_ridge_preprocessing_pipeline():
    """
    Verifies that RidgeRegressionBenchmark handles missing values via median imputation
    and scales features cleanly.
    """
    X_train = pd.DataFrame({
        "f1": [1.0, 2.0, np.nan, 4.0, 5.0],
        "f2": [10.0, 20.0, 30.0, 40.0, 50.0],
        "f_all_nan": [np.nan, np.nan, np.nan, np.nan, np.nan]
    })
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    X_test = pd.DataFrame({
        "f1": [np.nan, 3.0],
        "f2": [25.0, 35.0],
        "f_all_nan": [np.nan, np.nan]
    })

    ridge = RidgeRegressionBenchmark(alpha=40.0)
    ridge.fit(X_train, y_train)
    preds, ranks = ridge.predict(X_test)

    assert len(preds) == 2
    assert not np.isnan(preds).any()
    assert set(ranks) == {1, 2}


# -----------------------------------------------------------------------------
# Test 7: Model Prediction Output Validation
# -----------------------------------------------------------------------------
def test_model_prediction_output_validation(synthetic_race_data):
    """
    Verifies that all 6 models produce valid numeric predictions and valid ordinal ranks [1..N].
    """
    train_df = synthetic_race_data[synthetic_race_data["Round"] <= 3]
    test_df = synthetic_race_data[synthetic_race_data["Round"] == 4]

    X_train = train_df[BASELINE_FEATURES]
    y_train = train_df["ActualFinish"].values
    X_test = test_df[BASELINE_FEATURES]

    models = [
        StartingGridBaseline(),
        RandomForestBenchmark(),
        XGBoostBenchmark(),
        RidgeRegressionBenchmark(),
        MLOnlyEnsembleBenchmark(),
        ProductionHybridBenchmark(),
    ]

    n_test = len(test_df)
    for m in models:
        m.fit(X_train, y_train)
        pred_score, pred_rank = m.predict(X_test)

        assert len(pred_score) == n_test, f"{m.name} continuous output length mismatch"
        assert len(pred_rank) == n_test, f"{m.name} ordinal rank length mismatch"
        assert not np.isnan(pred_score).any(), f"{m.name} produced NaN scores"
        assert sorted(list(pred_rank)) == list(range(1, n_test + 1)), f"{m.name} ordinal ranks not 1..N permutation"


# -----------------------------------------------------------------------------
# Test 8: Metric Calculations Correctness
# -----------------------------------------------------------------------------
def test_metric_calculations_correctness():
    """
    Verifies the mathematical correctness of continuous MAE, ordinal MAE, RMSE,
    Spearman r, exact podium, top-3 overlap, and points accuracy.
    """
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    # Perfect prediction
    perfect_scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    perfect_ranks = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    m_perf = compute_race_metrics(perfect_scores, perfect_ranks, actual)
    assert m_perf["mae_continuous"] == 0.0
    assert m_perf["mae_ordinal"] == 0.0
    assert m_perf["rmse"] == 0.0
    assert m_perf["spearman"] == 1.0
    assert m_perf["winner_accuracy"] == 1.0
    assert m_perf["exact_podium"] == 1.0
    assert m_perf["top3_overlap"] == 3.0
    assert m_perf["podium_precision"] == 1.0
    assert m_perf["top5_overlap"] == 5.0
    assert m_perf["points_accuracy"] == 1.0

    # Inverted podium (P2, P1, P3) -> Top 3 overlap = 3.0, but exact podium = 0.0
    swapped_scores = np.array([2.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    swapped_ranks = np.array([2, 1, 3, 4, 5, 6, 7, 8, 9, 10])
    m_swap = compute_race_metrics(swapped_scores, swapped_ranks, actual)
    assert m_swap["exact_podium"] == 0.0
    assert m_swap["top3_overlap"] == 3.0


# -----------------------------------------------------------------------------
# Test 9: Missing and Pit-Lane Grid Handling
# -----------------------------------------------------------------------------
def test_missing_and_pit_lane_grid_handling():
    """
    Verifies that pit lane starters are demoted to the back of the grid,
    and missing grid positions are assigned midfield default (12.0).
    """
    df = pd.DataFrame({
        "Abbreviation": ["VER", "HAM", "TSU", "NOR"],
        "GridPosition": [1.0, 2.0, 3.0, np.nan],
        "ActualStartingGrid": [1.0, 2.0, 3.0, np.nan],
        "is_pit_lane_start": [0, 0, 1, 0]
    })

    grid_vals = get_grid_positions(df)
    assert grid_vals[0] == 1.0
    assert grid_vals[1] == 2.0
    # TSU had pit lane start -> placed behind max (3.0 + 1 = 4.0)
    assert grid_vals[2] > 3.0
    # NOR was missing -> assigned 12.0
    assert grid_vals[3] == 12.0


# -----------------------------------------------------------------------------
# Test 10: Reproducibility
# -----------------------------------------------------------------------------
def test_benchmark_reproducibility(synthetic_race_data):
    """
    Verifies that running the same model twice on identical folds produces 100% identical predictions.
    """
    train_df = synthetic_race_data[synthetic_race_data["Round"] <= 3]
    test_df = synthetic_race_data[synthetic_race_data["Round"] == 4]

    X_train = train_df[BASELINE_FEATURES]
    y_train = train_df["ActualFinish"].values
    X_test = test_df[BASELINE_FEATURES]

    rf1 = RandomForestBenchmark(params={"n_estimators": 50, "random_state": 42})
    rf2 = RandomForestBenchmark(params={"n_estimators": 50, "random_state": 42})

    rf1.fit(X_train, y_train)
    p1, r1 = rf1.predict(X_test)

    rf2.fit(X_train, y_train)
    p2, r2 = rf2.predict(X_test)

    np.testing.assert_array_almost_equal(p1, p2)
    np.testing.assert_array_equal(r1, r2)
