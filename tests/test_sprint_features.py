"""
Unit Tests for Phase 2C: Sprint-Derived Features
================================================
Validates:
1. Stage 'after_fp1': all Sprint features are NaN.
2. Stage 'after_sprint_qualifying': 'sprint_quali_pos' populated, Sprint Race features are NaN.
3. Stage 'after_sprint_race': Sprint Qualifying and Sprint Race features are populated.
4. Stage 'after_qualifying': all three Sprint features are populated when source data exists.
5. Normal weekend: all Sprint features are NaN, pipeline runs without error.
6. Sprint position delta: exact calculation SprintStartingGrid - SprintFinish (positive = gained, negative = lost).
7. Missing Sprint data: graceful fallback to NaN with zero exceptions.
8. Anti-leakage: GP Race result is never accessed or leaked into Sprint features.
9. Historical chronology: future Sprint information cannot enter an earlier prediction stage.
10. Model configuration: features list in config/model_config.json contains exactly 34 features.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import FeatureEngineer
from src.features.weekend_features import (
    build_sprint_qualifying_features,
    build_sprint_race_features,
)
from src.data.data_loader import DataLoader


@pytest.fixture
def mock_drivers():
    return ["VER", "NOR", "LEC", "PIA", "RUS"]


@pytest.fixture
def mock_sprint_quali_df():
    return pd.DataFrame({
        "Abbreviation": ["VER", "NOR", "LEC", "PIA", "RUS"],
        "Position": [1, 2, 3, 4, 5],
        "QualifyingTime": ["1:30.100", "1:30.250", "1:30.300", "1:30.400", "1:30.500"],
    })


@pytest.fixture
def mock_sprint_race_df():
    # VER: Grid 1 -> Finish 2 (delta = 1 - 2 = -1)
    # NOR: Grid 2 -> Finish 1 (delta = 2 - 1 = +1)
    # LEC: Grid 3 -> Finish 3 (delta = 3 - 3 = 0)
    # PIA: Grid 4 -> Finish 6 (delta = 4 - 6 = -2)
    # RUS: Grid 5 -> Finish 4 (delta = 5 - 4 = +1)
    return pd.DataFrame({
        "Abbreviation": ["NOR", "VER", "LEC", "RUS", "PIA"],
        "GridPosition": [2, 1, 3, 5, 4],
        "Position": [1, 2, 3, 4, 6],
    })


@pytest.fixture
def mock_qualifying_df():
    return pd.DataFrame({
        "Abbreviation": ["VER", "NOR", "LEC", "PIA", "RUS"],
        "Position": [1, 2, 3, 4, 5],
        "TeamName": ["Red Bull Racing", "McLaren", "Ferrari", "McLaren", "Mercedes"],
    })


@pytest.fixture
def mock_season_races_df():
    rows = []
    drivers = ["VER", "NOR", "LEC", "PIA", "RUS"]
    teams = ["Red Bull Racing", "McLaren", "Ferrari", "McLaren", "Mercedes"]
    for r in range(1, 5):
        for i, drv in enumerate(drivers):
            rows.append({
                "Round": r,
                "Race": f"Race_{r}",
                "Abbreviation": drv,
                "TeamName": teams[i],
                "GridPosition": i + 1,
                "Position": ((i + r - 1) % len(drivers)) + 1,
                "Points": max(0, 10 - i),
            })
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------
# 1. after_fp1: all Sprint features are NaN
# -------------------------------------------------------------------------
def test_stage_after_fp1_all_sprint_features_nan(
    mock_qualifying_df, mock_season_races_df
):
    fe = FeatureEngineer(expected_grid_size=len(mock_qualifying_df))
    # At after_fp1, only fp1 is available
    practice_sessions = {
        "fp1": pd.DataFrame({"Abbreviation": ["VER", "NOR"], "BestLapTime": [90.0, 90.5]}),
        "sprint_qualifying": None,
        "sprint_race": None,
    }
    target_df = fe.build_target_dataset(
        target_round=5,
        track_name="Baku",
        quali_df=mock_qualifying_df,
        practice_sessions=practice_sessions,
        season_races_df=mock_season_races_df,
    )

    assert "sprint_quali_pos" in target_df.columns
    assert "sprint_finish_pos" in target_df.columns
    assert "sprint_pos_delta" in target_df.columns

    assert target_df["sprint_quali_pos"].isna().all()
    assert target_df["sprint_finish_pos"].isna().all()
    assert target_df["sprint_pos_delta"].isna().all()


# -------------------------------------------------------------------------
# 2. after_sprint_qualifying: sprint_quali_pos populated, race features NaN
# -------------------------------------------------------------------------
def test_stage_after_sprint_qualifying(
    mock_qualifying_df, mock_sprint_quali_df, mock_season_races_df
):
    fe = FeatureEngineer(expected_grid_size=len(mock_qualifying_df))
    practice_sessions = {
        "fp1": pd.DataFrame({"Abbreviation": ["VER", "NOR"], "BestLapTime": [90.0, 90.5]}),
        "sprint_qualifying": mock_sprint_quali_df,
        "sprint_race": None,
    }
    target_df = fe.build_target_dataset(
        target_round=5,
        track_name="Baku",
        quali_df=mock_qualifying_df,
        practice_sessions=practice_sessions,
        season_races_df=mock_season_races_df,
    )

    ver_row = target_df[target_df["Abbreviation"] == "VER"].iloc[0]
    assert ver_row["sprint_quali_pos"] == 1.0
    assert np.isnan(ver_row["sprint_finish_pos"])
    assert np.isnan(ver_row["sprint_pos_delta"])

    assert target_df["sprint_quali_pos"].notna().all()
    assert target_df["sprint_finish_pos"].isna().all()
    assert target_df["sprint_pos_delta"].isna().all()


# -------------------------------------------------------------------------
# 3. after_sprint_race: sprint_quali_pos and sprint race features populated
# -------------------------------------------------------------------------
def test_stage_after_sprint_race(
    mock_qualifying_df, mock_sprint_quali_df, mock_sprint_race_df, mock_season_races_df
):
    fe = FeatureEngineer(expected_grid_size=len(mock_qualifying_df))
    practice_sessions = {
        "fp1": pd.DataFrame({"Abbreviation": ["VER", "NOR"], "BestLapTime": [90.0, 90.5]}),
        "sprint_qualifying": mock_sprint_quali_df,
        "sprint_race": mock_sprint_race_df,
    }
    target_df = fe.build_target_dataset(
        target_round=5,
        track_name="Baku",
        quali_df=mock_qualifying_df,
        practice_sessions=practice_sessions,
        season_races_df=mock_season_races_df,
    )

    assert target_df["sprint_quali_pos"].notna().all()
    assert target_df["sprint_finish_pos"].notna().all()
    assert target_df["sprint_pos_delta"].notna().all()

    # NOR: Grid 2 -> Finish 1 (delta = +1)
    nor = target_df[target_df["Abbreviation"] == "NOR"].iloc[0]
    assert nor["sprint_finish_pos"] == 1.0
    assert nor["sprint_pos_delta"] == 1.0

    # VER: Grid 1 -> Finish 2 (delta = -1)
    ver = target_df[target_df["Abbreviation"] == "VER"].iloc[0]
    assert ver["sprint_finish_pos"] == 2.0
    assert ver["sprint_pos_delta"] == -1.0


# -------------------------------------------------------------------------
# 4. after_qualifying: all three Sprint features populated when data exists
# -------------------------------------------------------------------------
def test_stage_after_qualifying_all_three_populated(
    mock_qualifying_df, mock_sprint_quali_df, mock_sprint_race_df, mock_season_races_df
):
    fe = FeatureEngineer(expected_grid_size=len(mock_qualifying_df))
    practice_sessions = {
        "fp1": pd.DataFrame({"Abbreviation": ["VER", "NOR"], "BestLapTime": [90.0, 90.5]}),
        "sprint_qualifying": mock_sprint_quali_df,
        "sprint_race": mock_sprint_race_df,
    }
    target_df = fe.build_target_dataset(
        target_round=5,
        track_name="Baku",
        quali_df=mock_qualifying_df,
        practice_sessions=practice_sessions,
        season_races_df=mock_season_races_df,
    )

    for col in ["sprint_quali_pos", "sprint_finish_pos", "sprint_pos_delta"]:
        assert col in target_df.columns
        assert not target_df[col].isna().any()


# -------------------------------------------------------------------------
# 5. Normal weekend: all Sprint features NaN and pipeline does not crash
# -------------------------------------------------------------------------
def test_normal_weekend_all_sprint_features_nan_no_crash(
    mock_qualifying_df, mock_season_races_df
):
    fe = FeatureEngineer(expected_grid_size=len(mock_qualifying_df))
    normal_sessions = {
        "fp1": pd.DataFrame({"Abbreviation": ["VER", "NOR"], "BestLapTime": [91.0, 91.2]}),
        "fp2": pd.DataFrame({"Abbreviation": ["VER", "NOR"], "BestLapTime": [90.5, 90.8]}),
        "fp3": pd.DataFrame({"Abbreviation": ["VER", "NOR"], "BestLapTime": [90.1, 90.3]}),
    }
    # No sprint keys present
    target_df = fe.build_target_dataset(
        target_round=16,
        track_name="Sepang",
        quali_df=mock_qualifying_df,
        practice_sessions=normal_sessions,
        season_races_df=mock_season_races_df,
    )

    assert target_df["sprint_quali_pos"].isna().all()
    assert target_df["sprint_finish_pos"].isna().all()
    assert target_df["sprint_pos_delta"].isna().all()
    assert len(target_df) == len(mock_qualifying_df)


# -------------------------------------------------------------------------
# 6. Sprint position delta: verify exact calculation
# -------------------------------------------------------------------------
def test_sprint_position_delta_calculation(mock_drivers):
    # Driver test cases:
    # VER: StartingGrid 1, Finish 3 -> delta = 1 - 3 = -2 (lost 2)
    # NOR: StartingGrid 5, Finish 1 -> delta = 5 - 1 = +4 (gained 4)
    # LEC: StartingGrid 10, Finish 10 -> delta = 10 - 10 = 0 (held)
    sr_df = pd.DataFrame({
        "Abbreviation": ["VER", "NOR", "LEC"],
        "StartingGrid": [1, 5, 10],
        "Position": [3, 1, 10],
    })
    res = build_sprint_race_features(sr_df, mock_drivers)
    res_map = res.set_index("Abbreviation").to_dict(orient="index")

    assert res_map["VER"]["sprint_finish_pos"] == 3.0
    assert res_map["VER"]["sprint_pos_delta"] == -2.0

    assert res_map["NOR"]["sprint_finish_pos"] == 1.0
    assert res_map["NOR"]["sprint_pos_delta"] == 4.0

    assert res_map["LEC"]["sprint_finish_pos"] == 10.0
    assert res_map["LEC"]["sprint_pos_delta"] == 0.0

    # Drivers not in sprint race dataframe should be NaN
    assert np.isnan(res_map["PIA"]["sprint_finish_pos"])
    assert np.isnan(res_map["PIA"]["sprint_pos_delta"])


# -------------------------------------------------------------------------
# 7. Missing Sprint files: graceful fallback to NaN with zero exceptions
# -------------------------------------------------------------------------
def test_missing_sprint_files_safe_nan_fallback(mock_drivers):
    # None DataFrame
    sq_none = build_sprint_qualifying_features(None, mock_drivers)
    assert sq_none["sprint_quali_pos"].isna().all()
    assert len(sq_none) == len(mock_drivers)

    sr_none = build_sprint_race_features(None, mock_drivers)
    assert sr_none["sprint_finish_pos"].isna().all()
    assert sr_none["sprint_pos_delta"].isna().all()
    assert len(sr_none) == len(mock_drivers)

    # Empty DataFrame
    sq_empty = build_sprint_qualifying_features(pd.DataFrame(), mock_drivers)
    assert sq_empty["sprint_quali_pos"].isna().all()

    sr_empty = build_sprint_race_features(pd.DataFrame(), mock_drivers)
    assert sr_empty["sprint_finish_pos"].isna().all()
    assert sr_empty["sprint_pos_delta"].isna().all()


# -------------------------------------------------------------------------
# 8. Anti-leakage: GP Race result is never accessed by Sprint builders
# -------------------------------------------------------------------------
def test_gp_race_leakage_protection(
    mock_qualifying_df, mock_sprint_quali_df, mock_sprint_race_df, mock_season_races_df
):
    fe = FeatureEngineer(expected_grid_size=len(mock_qualifying_df))
    practice_sessions = {
        "sprint_qualifying": mock_sprint_quali_df,
        "sprint_race": mock_sprint_race_df,
    }
    target_df = fe.build_target_dataset(
        target_round=5,
        track_name="Baku",
        quali_df=mock_qualifying_df,
        practice_sessions=practice_sessions,
        season_races_df=mock_season_races_df,
    )

    # Guaranteed absence of GP race finishing position
    assert "ActualFinish" not in target_df.columns
    assert "Position" not in target_df.columns or "QualifyingPosition" in target_df.columns


# -------------------------------------------------------------------------
# 9. Historical chronology: future Sprint information cannot enter earlier stage
# -------------------------------------------------------------------------
def test_historical_chronology_no_future_sprint_leakage():
    loader = DataLoader()
    # At after_fp1 on a sprint weekend, sprint_qualifying and sprint_race must be blocked
    sessions_fp1 = loader.load_weekend_sessions("baku", year=2026, stage="after_fp1", is_sprint=True)
    assert sessions_fp1["sprint_qualifying"] is None
    assert sessions_fp1["sprint_race"] is None
    assert sessions_fp1["qualifying"] is None
    assert sessions_fp1["race"] is None

    # At after_sprint_qualifying, sprint_race must still be blocked
    sessions_sq = loader.load_weekend_sessions("baku", year=2026, stage="after_sprint_qualifying", is_sprint=True)
    assert sessions_sq["sprint_race"] is None
    assert sessions_sq["qualifying"] is None
    assert sessions_sq["race"] is None


# -------------------------------------------------------------------------
# 10. Model configuration: features list contains exactly 34 features
# -------------------------------------------------------------------------
def test_model_config_has_34_features():
    config_path = Path(__file__).resolve().parent.parent / "config" / "model_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    features = cfg.get("features", [])
    assert len(features) == 34, f"Expected 34 configured features, found {len(features)}: {features}"
    assert "sprint_quali_pos" in features
    assert "sprint_finish_pos" in features
    assert "sprint_pos_delta" in features

    # Verify no disallowed features were introduced
    disallowed = [
        "sprint_points",
        "sprint_dnf",
        "weather",
        "track_temp",
        "air_temp",
        "teammate_quali_delta",
    ]
    for d in disallowed:
        assert d not in features, f"Disallowed feature '{d}' found in config!"
