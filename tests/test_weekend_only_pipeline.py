"""
Focused Unit Tests for Weekend-Only Prediction Architecture (Option B)
======================================================================
Tests:
- Analytical weekend scoring across stages (FP1-FP3, Sprint, Quali)
- Missing telemetry graceful fallbacks (zero NaNs, bounded [1.0, 22.0])
- Feature contract decoupling for weekend-only
- Pipeline execution without prior season rounds (Round 1 readiness)
- Deterministic random-seed behavior in Monte Carlo simulations
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from apexgrid.models.predict import calculate_analytical_weekend_score, Predictor
from apexgrid.features.build_features import FeatureEngineer, validate_stage_feature_contract
from apexgrid.simulation.monte_carlo import MonteCarloSimulator
from apexgrid.pipeline.race_prediction_pipeline import RacePredictionPipeline


@pytest.fixture
def sample_target_df():
    """Synthetic target race DataFrame with 20 drivers."""
    drivers = [
        "VER", "NOR", "LEC", "PIA", "SAI", "HAM", "RUS", "PER",
        "ALO", "STR", "TSU", "LAW", "HUL", "ALB", "OCO", "GAS",
        "BOT", "ZHO", "MAG", "COL"
    ]
    teams = [
        "Red Bull", "McLaren", "Ferrari", "McLaren", "Ferrari", "Mercedes", "Mercedes", "Red Bull",
        "Aston Martin", "Aston Martin", "RB", "RB", "Haas", "Williams", "Alpine", "Alpine",
        "Kick Sauber", "Kick Sauber", "Haas", "Williams"
    ]
    n = len(drivers)
    return pd.DataFrame({
        "Abbreviation": drivers,
        "TeamName": teams,
        "GridPosition": list(range(1, n + 1)),
        "simulation_start_rank": list(range(1, n + 1)),
        "ActualStartingGrid": list(range(1, n + 1)),
        "QualifyingPosition": list(range(1, n + 1)),
        "long_run_pace_gap_pct": [0.0, 0.15, 0.25, 0.30, 0.45, 0.50, 0.60, 0.70,
                                  0.85, 1.10, 1.20, 1.35, 1.50, 1.65, 1.80, 1.95,
                                  2.10, 2.30, 2.50, 2.80],
        "practice_gap_pct": [0.0, 0.10, 0.20, 0.35, 0.40, 0.55, 0.65, 0.75,
                             0.80, 1.05, 1.15, 1.30, 1.45, 1.60, 1.75, 1.90,
                             2.05, 2.25, 2.45, 2.75],
        "best_practice_lap": [80.0 + i * 0.15 for i in range(n)],
        "estimated_tyre_deg_rate": [0.050 + (i % 5) * 0.015 for i in range(n)],
        "fp_tyre_deg_slope": [0.040 + (i % 4) * 0.010 for i in range(n)],
        "sprint_finish_pos": list(range(1, n + 1)),
        "sprint_pos_delta": [0, 1, -1, 2, -2, 0, 1, -1, 0, 0, 1, -1, 2, -1, 0, -2, 1, 0, -1, 0],
        "provisional_grid_position": list(range(1, n + 1)),
    })


def test_calculate_analytical_weekend_score_shapes_and_bounds(sample_target_df):
    """Verifies scores are 1D float array bounded in [1.0, 22.0] with zero NaNs."""
    for stage in ["pre_fp1", "after_fp1", "after_fp2", "after_fp3", "after_sprint_race", "after_qualifying"]:
        scores = calculate_analytical_weekend_score(
            sample_target_df, stage=stage, circuit_type="permanent"
        )
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (len(sample_target_df),)
        assert not np.isnan(scores).any()
        assert (scores >= 1.0).all()
        assert (scores <= 22.0).all()


def test_calculate_analytical_weekend_score_relative_ordering(sample_target_df):
    """Pace leader (VER) must receive a better score than backmarker (COL)."""
    scores = calculate_analytical_weekend_score(
        sample_target_df, stage="after_qualifying", circuit_type="permanent"
    )
    # Lower score is better (finishing position rank space)
    assert scores[0] < scores[-1]
    assert scores[0] < 3.0
    assert scores[-1] > 15.0


def test_circuit_aware_grid_blending(sample_target_df):
    """Street circuits must weight starting grid more heavily (50%) than permanent circuits (35%)."""
    # Create target df where pace and grid disagree
    disagree_df = sample_target_df.copy()
    # Driver 0 qualified P20 but has P1 pace (gap = 0.0)
    disagree_df.loc[0, "GridPosition"] = 20.0
    disagree_df.loc[0, "simulation_start_rank"] = 20.0
    disagree_df.loc[0, "ActualStartingGrid"] = 20.0

    score_permanent = calculate_analytical_weekend_score(disagree_df, stage="after_qualifying", circuit_type="permanent")
    score_street = calculate_analytical_weekend_score(disagree_df, stage="after_qualifying", circuit_type="street")

    # On street circuits, starting P20 should penalize the driver more than on permanent tracks
    assert score_street[0] > score_permanent[0]


def test_sprint_telemetry_integration(sample_target_df):
    """Stage after_sprint_race must incorporate sprint finish and pos delta."""
    # Driver with good sprint performance vs poor sprint performance
    df = sample_target_df.copy()
    df.loc[0, "sprint_finish_pos"] = 1.0
    df.loc[0, "sprint_pos_delta"] = 4.0  # gained 4 spots

    df.loc[1, "sprint_finish_pos"] = 18.0
    df.loc[1, "sprint_pos_delta"] = -10.0  # lost 10 spots

    scores = calculate_analytical_weekend_score(df, stage="after_sprint_race", circuit_type="permanent")
    assert scores[0] < scores[1]


def test_graceful_fallback_missing_telemetry(sample_target_df):
    """All long-run and practice telemetry columns are NaN; pipeline must fall back cleanly without error."""
    sparse_df = sample_target_df.copy()
    sparse_df["long_run_pace_gap_pct"] = np.nan
    sparse_df["practice_gap_pct"] = np.nan
    sparse_df["best_practice_lap"] = np.nan
    sparse_df["estimated_tyre_deg_rate"] = np.nan
    sparse_df["fp_tyre_deg_slope"] = np.nan

    scores = calculate_analytical_weekend_score(sparse_df, stage="after_qualifying", circuit_type="permanent")
    assert isinstance(scores, np.ndarray)
    assert len(scores) == len(sample_target_df)
    assert not np.isnan(scores).any()
    assert (scores >= 1.0).all()
    assert (scores <= 22.0).all()


def test_empty_dataframe_handling():
    """Empty dataframe returns empty ndarray without error."""
    empty_df = pd.DataFrame()
    scores = calculate_analytical_weekend_score(empty_df, stage="after_qualifying")
    assert isinstance(scores, np.ndarray)
    assert len(scores) == 0


def test_predictor_class_method_delegation(sample_target_df):
    """Predictor instance method delegates properly to calculate_analytical_weekend_score."""
    predictor = Predictor(config={})
    scores_obj = predictor.calculate_analytical_weekend_score(sample_target_df, stage="after_qualifying", circuit_type="permanent")
    scores_func = calculate_analytical_weekend_score(sample_target_df, stage="after_qualifying", circuit_type="permanent")
    np.testing.assert_allclose(scores_obj, scores_func)


def test_feature_contract_weekend_only():
    """validate_stage_feature_contract accepts weekend_only=True with pure weekend features and rejects historical form features."""
    from apexgrid.features.build_features import WEEKEND_ONLY_FEATURES, HISTORICAL_FORM_FEATURES
    
    # 1. Pure weekend features should pass
    configured_features = ["practice_gap_pct", "practice_rank", "QualifyingPosition", "GridPosition"]
    df = pd.DataFrame(columns=configured_features)
    validate_stage_feature_contract(df, stage="after_qualifying", feature_cols=configured_features, weekend_only=True)

    # 2. Including historical form features must raise ValueError
    leaked_features = configured_features + ["driver_finish_ewma"]
    df_leaked = pd.DataFrame(columns=leaked_features)
    with pytest.raises(ValueError, match="Historical form features forbidden in weekend_only mode"):
        validate_stage_feature_contract(df_leaked, stage="after_qualifying", feature_cols=leaked_features, weekend_only=True)


def test_monte_carlo_deterministic_seed_with_analytical_score(sample_target_df):
    """Monte Carlo simulator produces identical deterministic outputs given same random_seed."""
    scores = calculate_analytical_weekend_score(sample_target_df, stage="after_qualifying")
    cfg = {"simulation": {"random_seed": 42, "default_std": 2.85, "overtaking_difficulty": 0.5}}
    
    sim1 = MonteCarloSimulator(cfg)
    res1, meta1 = sim1.simulate(
        target_df=sample_target_df,
        deterministic_scores=scores,
        residuals=None,
        n_simulations=1000,
    )

    sim2 = MonteCarloSimulator(cfg)
    res2, meta2 = sim2.simulate(
        target_df=sample_target_df,
        deterministic_scores=scores,
        residuals=None,
        n_simulations=1000,
    )

    np.testing.assert_allclose(res1["ExpectedFinish"].values, res2["ExpectedFinish"].values)
    np.testing.assert_allclose(res1["WinProbability"].values, res2["WinProbability"].values)
    assert (res1["PredictedPosition"].values == res2["PredictedPosition"].values).all()


@pytest.mark.slow
def test_race_prediction_pipeline_weekend_only_execution(tmp_path):
    """Pipeline runs successfully with weekend_only=True using analytical weekend telemetry scoring."""
    pipeline = RacePredictionPipeline(root_dir=Path("."))
    
    # Run with weekend_only=True on sepang 2026 Round 16 at after_qualifying
    results = pipeline.run(
        track_key="sepang",
        year=2026,
        round_num=16,
        stage="after_qualifying",
        weekend_only=True,
        n_simulations=500,
    )

    assert "predictions" in results
    df = results["predictions"]
    assert len(df) == 22
    assert "ExpectedFinish" in df.columns
    assert "WinProbability" in df.columns
    assert "Analytical Weekend Telemetry Scorer" in results["selected_model"]
    assert results["validation_status"] == "WEEKEND_ONLY_ANALYTICAL"
    assert results["is_out_of_sample"] is False
    assert results["selected_mae"] is None
    assert results["simulation_meta"]["readiness_classification"] == "WEEKEND_ONLY"
    # Stage after_qualifying should use pace_sigma = 2.55
    assert results["simulation_meta"]["calibrated_sigma"] == 2.55


@pytest.mark.slow
def test_race_prediction_pipeline_stage_adaptive_sigma():
    """Pipeline uses stage-adaptive sigma across stages in weekend_only mode."""
    pipeline = RacePredictionPipeline(root_dir=Path("."))
    
    # Test after_fp1 stage (should have sigma = 3.50)
    results_fp1 = pipeline.run(
        track_key="sepang",
        year=2026,
        round_num=16,
        stage="after_fp1",
        weekend_only=True,
        n_simulations=100,
    )
    assert results_fp1["simulation_meta"]["calibrated_sigma"] == 3.50


def test_tyre_compound_normalization_ranking():
    """Tests that compound pace normalization appropriately adjusts Soft vs Hard long runs."""
    from apexgrid.features.race_pace import compute_driver_race_pace

    # Driver A did 10 laps on SOFT at 90.0s (normalized to Medium: 90.0 - (-0.65) = 90.65s)
    # Driver B did 10 laps on HARD at 90.8s (normalized to Medium: 90.8 - 0.55 = 90.25s)
    stints = [
        {
            "driver": "DRV_A",
            "stint_number": 1,
            "compound": "SOFT",
            "lap_count": 10,
            "raw_lap_times": [90.0] * 10,
            "fuel_adj_times": [90.0] * 10,
            "mean_lap": 90.0,
            "median_lap": 90.0,
            "best_lap": 90.0,
            "std_lap": 0.1,
            "fuel_adj_mean_lap": 90.0,
            "deg_slope": 0.05,
        },
        {
            "driver": "DRV_B",
            "stint_number": 1,
            "compound": "HARD",
            "lap_count": 10,
            "raw_lap_times": [90.8] * 10,
            "fuel_adj_times": [90.8] * 10,
            "mean_lap": 90.8,
            "median_lap": 90.8,
            "best_lap": 90.8,
            "std_lap": 0.1,
            "fuel_adj_mean_lap": 90.8,
            "deg_slope": 0.05,
        },
    ]
    df = compute_driver_race_pace(stints, ["DRV_A", "DRV_B"])
    drv_a_pace = df[df["Abbreviation"] == "DRV_A"]["fuel_adj_pace"].iloc[0]
    drv_b_pace = df[df["Abbreviation"] == "DRV_B"]["fuel_adj_pace"].iloc[0]

    # After normalization to Medium, DRV_B (90.25s) is faster than DRV_A (90.65s)
    assert drv_b_pace < drv_a_pace


def test_sprint_telemetry_persistence_after_qualifying(sample_target_df):
    """Sprint finishing position persists into pace score at after_qualifying."""
    df_with_sprint = sample_target_df.copy()
    # Driver 0 won sprint, Driver 1 finished P20 in sprint
    df_with_sprint.loc[0, "sprint_finish_pos"] = 1.0
    df_with_sprint.loc[1, "sprint_finish_pos"] = 20.0

    scores_with = calculate_analytical_weekend_score(df_with_sprint, stage="after_qualifying", circuit_type="permanent")
    
    # Driver 0 pace should be improved relative to Driver 1
    assert scores_with[0] < scores_with[1]

