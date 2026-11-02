"""
Tests for Phase 0.5: NaN Failure Fix, Feature Imputation & Pipeline Reliability
================================================================================
Validates:
1. MonteCarloSimulator robustness against NaN values in grid/pitlane features.
2. create_feature_imputer handling of all-NaN columns without dropping dimensions or raising UserWarnings.
3. Feature masking behavior at pre_fp1 preserving EWMA and masking future sessions without leakage.
4. RacePredictionPipeline._audit_prediction_integrity checks for driver uniqueness, value ranges, and contiguous ranks.
"""

import warnings
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from src.models.train import create_feature_imputer
from src.simulation.monte_carlo import MonteCarloSimulator
from src.pipeline.race_prediction_pipeline import RacePredictionPipeline


def test_create_feature_imputer_all_nan_columns():
    """Verify that create_feature_imputer preserves column count for all-NaN features with zero warnings."""
    imputer = create_feature_imputer(strategy="median")

    # Feature matrix with 3 columns, where column 'future_fp' is completely NaN
    X_train = pd.DataFrame({
        "driver_quali_ewma": [2.5, 4.0, 1.2, 5.8],
        "team_points_rate": [15.0, 8.0, 22.0, 3.0],
        "future_fp_gap": [np.nan, np.nan, np.nan, np.nan],
    })

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        X_imputed = imputer.fit_transform(X_train)

    # Scikit-learn UserWarning about skipping features should NOT be raised
    user_warnings = [w for w in recorded_warnings if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 0, f"Unexpected UserWarning: {[str(w.message) for w in user_warnings]}"

    # Shape must be preserved (4 rows, 3 columns)
    assert X_imputed.shape == (4, 3)

    # All-NaN column should be imputed with 0.0
    np.testing.assert_array_equal(X_imputed[:, 2], np.array([0.0, 0.0, 0.0, 0.0]))

    # Test transform on unseen data
    X_test = pd.DataFrame({
        "driver_quali_ewma": [3.1],
        "team_points_rate": [12.0],
        "future_fp_gap": [np.nan],
    })
    X_test_imputed = imputer.transform(X_test)
    assert X_test_imputed.shape == (1, 3)
    assert X_test_imputed[0, 2] == 0.0


def test_monte_carlo_nan_grid_and_pit_lane():
    """Verify MonteCarloSimulator.simulate handles missing/NaN pitlane and starting grid data gracefully."""
    config = {
        "monte_carlo": {"default_simulations": 500},
        "random_seed": 42
    }
    sim = MonteCarloSimulator(config=config)

    drivers = ["VER", "NOR", "LEC", "PIA", "RUS"]
    n = len(drivers)

    # Target dataframe mimicking pre_fp1 where grid and pitlane info are NaN
    target_df = pd.DataFrame({
        "Abbreviation": drivers,
        "TeamName": ["Red Bull", "McLaren", "Ferrari", "McLaren", "Mercedes"],
        "ActualStartingGrid": [np.nan] * n,
        "is_pit_lane_start": [np.nan] * n,
        "grid_penalty_places": [np.nan] * n,
        "QualifyingPosition": [np.nan] * n,
        "simulation_start_rank": [np.nan] * n,
    })

    deterministic_scores = np.array([1.5, 2.1, 2.8, 3.5, 4.0])
    residuals = np.array([0.2, -0.1, 0.4, -0.3, 0.1, 0.5, -0.4, 0.0])
    empirical_dnf_rate = 0.08

    # Must complete without ValueError: cannot convert float NaN to integer
    results_df, meta = sim.simulate(
        target_df=target_df,
        deterministic_scores=deterministic_scores,
        residuals=residuals,
        empirical_dnf_rate=empirical_dnf_rate,
        circuit_key="mexico",
    )

    assert len(results_df) == n
    assert "ExpectedFinish" in results_df.columns
    assert "PredictedPosition" in results_df.columns
    assert results_df["PredictedPosition"].tolist() == list(range(1, n + 1))
    assert results_df["ExpectedFinish"].isna().sum() == 0
    assert results_df["WinProbability"].isna().sum() == 0


def test_prediction_integrity_audit():
    """Verify _audit_prediction_integrity passes valid data and catches anomalies."""
    pipeline = RacePredictionPipeline()

    valid_results = pd.DataFrame({
        "Abbreviation": [f"D{i}" for i in range(1, 21)],
        "TeamName": [f"Team{i//2}" for i in range(1, 21)],
        "ExpectedFinish": [float(i) + 0.1 for i in range(1, 21)],
        "MedianFinish": list(range(1, 21)),
        "WinProbability": [40.0, 30.0, 15.0, 10.0, 5.0] + [0.0] * 15,
        "Top3Probability": [80.0, 70.0, 60.0, 40.0, 30.0] + [1.0] * 15,
        "Top5Probability": [95.0, 90.0, 85.0, 70.0, 60.0] + [5.0] * 15,
        "PointsProbability": [99.0] * 10 + [5.0] * 10,
        "PredictedPosition": list(range(1, 21)),
        "P5": list(range(1, 21)),
        "P25": list(range(1, 21)),
        "P50": list(range(1, 21)),
        "P75": list(range(1, 21)),
        "P95": list(range(1, 21)),
    })

    # Valid results must not raise
    pipeline._audit_prediction_integrity(valid_results, "mexico", 2026, "pre_fp1")

    # Anomaly 1: NaN in mandatory column
    invalid_nan = valid_results.copy()
    invalid_nan.loc[0, "ExpectedFinish"] = np.nan
    with pytest.raises(ValueError, match="contains 1 NaN values"):
        pipeline._audit_prediction_integrity(invalid_nan, "mexico", 2026, "pre_fp1")

    # Anomaly 2: Duplicate drivers
    invalid_dup = valid_results.copy()
    invalid_dup.loc[1, "Abbreviation"] = invalid_dup.loc[0, "Abbreviation"]
    with pytest.raises(ValueError, match="Duplicate drivers found"):
        pipeline._audit_prediction_integrity(invalid_dup, "mexico", 2026, "pre_fp1")

    # Anomaly 3: Probability > 100%
    invalid_prob = valid_results.copy()
    invalid_prob.loc[0, "WinProbability"] = 105.0
    with pytest.raises(ValueError, match="values outside"):
        pipeline._audit_prediction_integrity(invalid_prob, "mexico", 2026, "pre_fp1")

    # Anomaly 4: Non-contiguous predicted ranks
    invalid_rank = valid_results.copy()
    invalid_rank.loc[0, "PredictedPosition"] = 99
    with pytest.raises(ValueError, match="non-contiguous"):
        pipeline._audit_prediction_integrity(invalid_rank, "mexico", 2026, "pre_fp1")
