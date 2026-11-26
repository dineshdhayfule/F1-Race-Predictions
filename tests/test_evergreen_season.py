"""
Evergreen Season Resolution, Pre-2018 Stage Filtering & Historical Grid Provenance Tests
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import pytest

from src.cli.menu import get_available_stages_for_event
from src.utils.driver_mapping import resolve_driver_metadata
from src.utils.season_utils import MissingSessionTelemetryError, get_active_season


def test_get_active_season_uses_utc_year():
    assert get_active_season(datetime(2027, 1, 15, tzinfo=timezone.utc)) == 2027
    assert get_active_season(datetime(2028, 12, 31, tzinfo=timezone.utc)) == 2028


def test_missing_session_telemetry_error_message():
    err = MissingSessionTelemetryError(year=2027, round_num=1)
    msg = str(err)
    assert "2027 Round 1" in msg
    assert "--year <YYYY> --round <R>" in msg


def test_dynamic_stage_filtering_by_era_and_format():
    pre_2018 = [s[1] for s in get_available_stages_for_event(1950, is_sprint=False)]
    assert "before_race" in pre_2018
    assert "pre_fp1" in pre_2018
    assert "after_fp1" not in pre_2018
    assert "after_fp2" not in pre_2018
    assert "after_fp3" not in pre_2018
    assert "after_sprint_qualifying" not in pre_2018

    modern_std = [s[1] for s in get_available_stages_for_event(2025, is_sprint=False)]
    assert modern_std == [
        "pre_fp1", "after_fp1", "after_fp2", "after_fp3",
        "after_qualifying", "before_race", "post_race",
    ]

    modern_sprint = [s[1] for s in get_available_stages_for_event(2025, is_sprint=True)]
    assert modern_sprint == [
        "pre_fp1", "after_fp1", "after_sprint_qualifying",
        "after_sprint_race", "after_qualifying", "before_race", "post_race",
    ]


def test_resolve_driver_metadata_prefers_session_df():
    session_df = pd.DataFrame([
        {"Abbreviation": "HAM", "FullName": "Lewis Hamilton", "DriverNumber": 44, "TeamName": "Mercedes"},
    ])
    meta = resolve_driver_metadata("HAM", session_df=session_df)
    assert meta["default_team"] == "Mercedes"
    assert meta["number"] == 44


def test_pre_2018_after_qualifying_accepts_f1_website_grid_provenance(tmp_path):
    """
    Verifies that a pre-2018 race with cached f1_website_grid provenance and quali_df=None
    does not raise the 'Official qualifying results are unavailable' ValueError.
    """
    from src.pipeline.race_prediction_pipeline import RacePredictionPipeline

    pipeline = RacePredictionPipeline(root_dir=tmp_path)
    fake_races = pd.DataFrame([
        {"Abbreviation": "FAR", "TeamName": "Alfa Romeo", "Position": 1, "GridPosition": 1, "Status": "Finished", "Round": 1},
        {"Abbreviation": "FAN", "TeamName": "Alfa Romeo", "Position": 2, "GridPosition": 2, "Status": "Finished", "Round": 1},
    ])
    fake_grid = pd.DataFrame([
        {"Abbreviation": "FAN", "TeamName": "Alfa Romeo", "ActualStartingGrid": 1, "QualifyingPosition": 1},
        {"Abbreviation": "FAR", "TeamName": "Alfa Romeo", "ActualStartingGrid": 2, "QualifyingPosition": 2},
    ])

    class _StopAfterGridCheck(Exception):
        pass

    with patch.object(pipeline.data_loader, "load_season_rounds", return_value=(fake_races, pd.DataFrame())), \
         patch.object(pipeline.data_loader, "load_all_practice_sessions", return_value={}), \
         patch.object(pipeline.data_loader, "load_weekend_sessions", return_value={}), \
         patch.object(pipeline.data_loader, "load_target_qualifying", return_value=None), \
         patch.object(pipeline.data_loader, "load_starting_grid", return_value=(fake_grid, {"provenance": "f1_website_grid"})), \
         patch.object(pipeline.feature_engineer, "build_training_dataset", side_effect=_StopAfterGridCheck("reached_feature_eng")):
        with pytest.raises(_StopAfterGridCheck):
            pipeline.run(track_key="monaco", year=1950, round_num=2, stage="after_qualifying")
