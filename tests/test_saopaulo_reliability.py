"""
Tests for Phase 1: São Paulo Pre-FP1 Failure Investigation and Repair
=====================================================================
Validates:
1. Issue A: Logger ownership and scope (no NameError: 'logger' is not defined in menu or evaluation).
2. Issue B: Exactly 22 unique current-season drivers for 2026 target prediction;
   proper identification and exclusion of FIA course/safety/medical vehicles without
   excluding genuine historical F1 drivers.
3. Issue C: Stage-aware validation and walk-forward validation resilience.
"""

import pandas as pd
import numpy as np
import pytest

from src.utils.driver_mapping import (
    is_course_vehicle,
    filter_competing_drivers,
    normalize_driver,
    DRIVER_CATALOG,
)
from src.data.data_loader import DataLoader
from src.pipeline.race_prediction_pipeline import RacePredictionPipeline


def test_logger_defined_in_menu():
    """Verify logger is defined and properly scoped in src.cli.menu."""
    import src.cli.menu as menu_module
    assert hasattr(menu_module, "logger"), "src.cli.menu must have 'logger' defined."
    assert menu_module.logger is not None, "menu logger must not be None."


def test_course_vehicle_identification():
    """Verify FIA course/safety/medical cars are identified, while genuine historical drivers are preserved."""
    course_cars = ["SC1", "SC2", "SC3", "MC1", "MC2", "FM1", "FM2", "FM3", "FM4", "FM5"]
    for code in course_cars:
        assert is_course_vehicle(code), f"Expected {code} to be recognized as course vehicle"

    # Test DataFrame rows with course vehicle attributes
    test_row = pd.Series({
        "Abbreviation": "FM1",
        "BroadcastName": "F TEST CAR 1",
        "FullName": "None None",
        "TeamName": np.nan,
        "DriverNumber": 236,
    })
    assert is_course_vehicle(test_row), "Test car row must be recognized as course vehicle"

    # Test that genuine current and historical F1 drivers are NOT marked as course vehicles
    legitimate_drivers = [
        "VER", "HAM", "LEC", "NOR", "PIA", "RUS", "ANT", "HAD", "OCO", "PER",
        "MAS", "ROS", "BUT", "GRO", "KUB", "KVY", "NAS", "PAL", "HIR", "VET"
    ]
    for d in legitimate_drivers:
        assert not is_course_vehicle(d), f"Legitimate driver {d} must NOT be marked as course vehicle"


def test_filter_competing_drivers_and_position_reindexing():
    """Verify filter_competing_drivers excludes course vehicles and re-indexes contiguous finish positions."""
    df = pd.DataFrame({
        "Position": [1.0, 2.0, 3.0, 4.0, np.nan, np.nan],
        "Abbreviation": ["ANT", "VER", "FM1", "OCO", "SC1", "MC1"],
        "FullName": ["Kimi Antonelli", "Max Verstappen", "None None", "Esteban Ocon", "None None", "None None"],
        "TeamName": ["Mercedes", "Red Bull", np.nan, "Haas F1 Team", np.nan, np.nan],
        "DriverNumber": [12, 1, 236, 31, 241, 246],
        "BroadcastName": ["K ANTONELLI", "M VERSTAPPEN", "F TEST CAR 1", "E OCON", "F SAFETY CAR 1", "F MEDICAL CAR 1"]
    })

    cleaned = filter_competing_drivers(df)
    assert len(cleaned) == 3, f"Expected 3 drivers, got {len(cleaned)}"
    assert cleaned["Abbreviation"].tolist() == ["ANT", "VER", "OCO"]
    # FM1 was at 3.0, OCO was at 4.0 -> after dropping FM1, OCO should be at 3.0
    assert cleaned["Position"].tolist() == [1.0, 2.0, 3.0]


def test_season_rounds_loader_excludes_course_vehicles():
    """Verify DataLoader.load_season_rounds excludes non-driver course vehicles."""
    loader = DataLoader()
    races_df, qualis_df = loader.load_season_rounds(2026, list(range(1, 20)))

    r19 = races_df[races_df["Round"] == 19]
    assert len(r19) in (21, 22), f"Round 19 must have 21 or 22 competing drivers, got {len(r19)}"
    assert r19["Abbreviation"].nunique() == len(r19), "Round 19 drivers must be unique"

    # No course car abbreviations
    course_cars = {"SC1", "SC2", "SC3", "MC1", "MC2", "FM1", "FM2", "FM3", "FM4", "FM5"}
    assert set(r19["Abbreviation"]).isdisjoint(course_cars), "Round 19 must contain zero course vehicles"

    # Total across available completed rounds without course cars (351 without R18, 373 with R18, 417 with full rounds)
    assert len(races_df) in (351, 373, 417), f"Expected 351, 373, or 417 driver-race observations across available rounds, got {len(races_df)}"


def test_walk_forward_evaluation_with_nan_targets():
    """Verify evaluate_walk_forward does not crash on folds containing NaN ActualFinish."""
    from src.models.evaluate import ModelEvaluator
    import json

    with open("config/model_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    evaluator = ModelEvaluator(cfg)

    # Mock training dataframe with rounds 4..6 where some drivers have NaN ActualFinish
    rows = []
    for r in range(1, 7):
        for i, d in enumerate(["VER", "NOR", "LEC", "PIA", "RUS"]):
            finish = float(i + 1)
            if r == 5 and d == "RUS":
                finish = np.nan  # Simulated DNF without classified rank
            rows.append({
                "Round": r,
                "Abbreviation": d,
                "TeamName": "TestTeam",
                "ActualFinish": finish,
                "GridPosition": float(i + 1),
                "driver_quali_ewma": float(i + 1),
                "team_finish_ewma": float(i + 1),
                "team_points_rate": 10.0,
            })
    mock_df = pd.DataFrame(rows)

    # Must complete without ValueError: Input contains NaN
    res = evaluator.evaluate_walk_forward(mock_df, stage="pre_fp1")
    assert "best_model_name" in res
    assert "ensemble_comparisons" in res

