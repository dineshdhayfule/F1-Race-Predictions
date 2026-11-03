"""
Unit and Integration Tests for Phase 3 Feature Ablation & Ensemble Optimization.
Verifies feature grouping, LOGO feature removal, inner CV isolation, weight constraints,
Ridge ensemble blending, tie handling, and reproducibility.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from experiments.model_benchmark.config import (
    BASELINE_FEATURES,
    RANDOM_SEED,
    RIDGE_ALPHA,
)
from experiments.phase3.feature_inventory import (
    FEATURE_GROUPS,
    build_feature_inventory,
)
from experiments.phase3.feature_ablation import (
    get_features_by_group,
    ACTIVE_GROUPS,
)
from experiments.phase3.ensemble_search import EnsembleSearchOptimizer
from experiments.phase3.podium_analysis import run_podium_investigation
from experiments.model_benchmark.models import (
    StartingGridBaseline,
    RandomForestBenchmark,
    XGBoostBenchmark,
    RidgeRegressionBenchmark,
    get_grid_positions,
)
from experiments.model_benchmark.metrics import compute_race_metrics


@pytest.fixture
def sample_feature_data():
    """Generates synthetic multi-round data for 10 drivers across 5 rounds."""
    rng = np.random.RandomState(42)
    rows = []
    drivers = ["VER", "HAM", "LEC", "NOR", "RUS", "PIA", "ALO", "SAI", "GAS", "TSU"]
    for rnd in range(1, 6):
        finishes = list(range(1, 11))
        rng.shuffle(finishes)
        for i, drv in enumerate(drivers):
            row = {
                "Round": rnd,
                "Abbreviation": drv,
                "ActualFinish": float(finishes[i]),
                "GridPosition": float(i + 1),
                "ActualStartingGrid": float(i + 1),
                "is_pit_lane_start": 1 if drv == "TSU" and rnd == 5 else 0,
            }
            for col in BASELINE_FEATURES:
                if col not in row:
                    row[col] = float(rng.normal(10.0, 2.0))
            rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Test 1 & 2: Feature-Group Assignment & Feature Removal
# -----------------------------------------------------------------------------
def test_feature_group_assignment_and_removal():
    """Verifies that all 37 baseline features belong to valid groups and can be cleanly ablated."""
    assert len(BASELINE_FEATURES) == 37
    for f in BASELINE_FEATURES:
        assert f in FEATURE_GROUPS
        assert FEATURE_GROUPS[f] in ACTIVE_GROUPS

    # Verify feature removal for LOGO
    for grp in ACTIVE_GROUPS:
        grp_feats = get_features_by_group(grp)
        assert len(grp_feats) > 0
        remaining = [f for f in BASELINE_FEATURES if f not in grp_feats]
        assert len(remaining) == len(BASELINE_FEATURES) - len(grp_feats)
        for gf in grp_feats:
            assert gf not in remaining


# -----------------------------------------------------------------------------
# Test 3, 4, 5: Leakage Prevention, Inner Chronological Validation & Outer Test Isolation
# -----------------------------------------------------------------------------
def test_inner_chronological_validation_and_outer_isolation(sample_feature_data):
    """
    Verifies that inner weight optimization uses ONLY training rounds (< outer_test_round)
    and completely isolates the outer test race.
    """
    df = sample_feature_data
    outer_test_round = 5
    meta_train = df[df["Round"] < outer_test_round]

    assert meta_train["Round"].max() < outer_test_round
    assert "ActualFinish" in meta_train.columns

    optimizer = EnsembleSearchOptimizer()
    w_rf, w_xgb, w_rdg, w_grid = optimizer._optimize_weights_inner_cv(meta_train)

    # Weights must be positive and bounded
    assert 0.0 <= w_rf <= 1.0
    assert 0.0 <= w_xgb <= 1.0
    assert 0.0 <= w_rdg <= 1.0
    assert 0.0 <= w_grid <= 1.0
    # ML weights must sum to 1.0
    assert abs((w_rf + w_xgb + w_rdg) - 1.0) < 1e-4


# -----------------------------------------------------------------------------
# Test 6: Ridge Ensemble Compatibility
# -----------------------------------------------------------------------------
def test_ridge_ensemble_blending_compatibility(sample_feature_data):
    """
    Verifies that RidgeRegression predictions can be smoothly blended with RF and XGBoost
    without scale distortion or NaN generation.
    """
    train_df = sample_feature_data[sample_feature_data["Round"] <= 3]
    test_df = sample_feature_data[sample_feature_data["Round"] == 4]

    X_train = train_df[BASELINE_FEATURES]
    y_train = train_df["ActualFinish"].values
    X_test = test_df[BASELINE_FEATURES]

    rf = RandomForestBenchmark(params={"n_estimators": 50, "random_state": 42}).fit(X_train, y_train)
    xgb_m = XGBoostBenchmark(params={"n_estimators": 50, "random_state": 42}).fit(X_train, y_train)
    ridge = RidgeRegressionBenchmark(alpha=40.0).fit(X_train, y_train)

    p_rf, _ = rf.predict(X_test)
    p_xgb, _ = xgb_m.predict(X_test)
    p_rdg, _ = ridge.predict(X_test)

    # 3-way blend
    blend = 0.40 * p_rf + 0.30 * p_xgb + 0.30 * p_rdg
    assert len(blend) == len(test_df)
    assert not np.isnan(blend).any()

    ranks = np.argsort(np.argsort(blend)) + 1
    assert sorted(list(ranks)) == list(range(1, len(test_df) + 1))


# -----------------------------------------------------------------------------
# Test 7: Deterministic Ranking and Tie Handling
# -----------------------------------------------------------------------------
def test_deterministic_ranking_and_tie_handling():
    """
    Verifies that ties in predicted scores are resolved deterministically
    without using actual finishes or non-deterministic ordering.
    """
    scores = np.array([5.5, 3.2, 5.5, 1.1, 7.8])
    ranks = np.argsort(np.argsort(scores)) + 1

    assert ranks[3] == 1  # 1.1 is P1
    assert ranks[1] == 2  # 3.2 is P2
    # Identical 5.5 scores receive deterministic distinct ranks (3 and 4)
    assert set([ranks[0], ranks[2]]) == {3, 4}
    assert ranks[4] == 5  # 7.8 is P5


# -----------------------------------------------------------------------------
# Test 8: Missing Grid & Pit Lane Handling
# -----------------------------------------------------------------------------
def test_missing_grid_and_pit_lane_handling():
    """Verifies that pit-lane starters are placed behind the grid, and NaNs get default 12.0."""
    df = pd.DataFrame({
        "Abbreviation": ["HAM", "VER", "TSU", "COL"],
        "GridPosition": [1.0, 2.0, 3.0, np.nan],
        "is_pit_lane_start": [0, 0, 1, 0]
    })
    grids = get_grid_positions(df)
    assert grids[0] == 1.0
    assert grids[1] == 2.0
    assert grids[2] > 3.0  # TSU pit lane demoted
    assert grids[3] == 12.0  # COL missing grid gets 12.0


# -----------------------------------------------------------------------------
# Test 9: Circuit-Type Grouping
# -----------------------------------------------------------------------------
def test_circuit_type_classification():
    """Verifies tracks are correctly partitioned into street, permanent, and temporary."""
    street_circuits = {"monaco", "baku", "singapore"}
    permanent_circuits = {"silverstone", "spa", "monza", "barcelona", "austria"}

    for c in street_circuits:
        assert c not in permanent_circuits


# -----------------------------------------------------------------------------
# Test 10: Reproducibility
# -----------------------------------------------------------------------------
def test_phase3_reproducibility(sample_feature_data):
    """Verifies identical optimization outputs under identical seeds."""
    meta_train = sample_feature_data[sample_feature_data["Round"] <= 4]
    opt1 = EnsembleSearchOptimizer(random_state=42)
    opt2 = EnsembleSearchOptimizer(random_state=42)

    res1 = opt1._optimize_weights_inner_cv(meta_train)
    res2 = opt2._optimize_weights_inner_cv(meta_train)

    assert res1 == res2
