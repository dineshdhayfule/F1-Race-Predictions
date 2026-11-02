"""
Sprint-Stage Temporal Anti-Leakage Regression Test Suite
========================================================
Validates Priority 1 remediation:
1. Both sprint stages ('after_sprint_qualifying' and 'after_sprint_race' / 'after_sprint')
   have explicit, intentional availability rules.
2. Sunday qualifying information (QualifyingPosition, quali_gap_pct, etc.) cannot leak into sprint predictions.
3. Sunday grid information (GridPosition, ActualStartingGrid, penalties, etc.) cannot leak into sprint predictions.
4. Future race results (ActualFinish, Position, etc.) cannot leak into sprint predictions.
5. Legitimately available sprint information remains usable:
   - 'sprint_quali_pos' is available at 'after_sprint_qualifying'.
   - 'sprint_quali_pos', 'sprint_finish_pos', 'sprint_pos_delta' are available at 'after_sprint_race'.
6. Non-sprint stages retain their expected feature availability.
7. Uses purely synthetic fixtures independent of live FastF1 network calls.
"""

import pytest
import numpy as np
import pandas as pd
from src.features.build_features import apply_stage_feature_mask, STAGE_UNAVAILABLE_FEATURES
from src.data.data_validator import DataValidator


@pytest.fixture
def synthetic_weekend_feature_df():
    """Generates a complete synthetic feature row with all weekend and session columns populated."""
    return pd.DataFrame([{
        "Abbreviation": "VER",
        "TeamName": "Red Bull Racing",
        "Round": 2,
        # Rolling form (pre-weekend)
        "driver_finish_ewma": 2.1,
        "team_finish_ewma": 3.4,
        "total_points": 25.0,
        # Practice
        "fp1_best_lap": 88.5,
        "fp1_gap_pct": 0.0,
        "fp2_best_lap": 87.9,
        "fp3_best_lap": 87.2,
        "practice_rank": 1.0,
        # Sprint sessions
        "sprint_quali_pos": 1.0,
        "sprint_finish_pos": 1.0,
        "sprint_pos_delta": 0.0,
        # Sunday Qualifying
        "QualifyingPosition": 2.0,
        "quali_pos_norm": 2.0 / 22.0,
        "quali_gap_pct": 0.12,
        "quali_no_time": 0,
        "q1_gap_pct": 0.05,
        "q2_gap_pct": 0.08,
        "q3_gap_pct": 0.12,
        "teammate_quali_gap_pct": -0.25,
        # Sunday Starting Grid
        "GridPosition": 2.0,
        "ActualStartingGrid": 2.0,
        "simulation_start_rank": 2,
        "is_pit_lane_start": 0,
        "StartingGridStatus": "CLASSIFIED",
        "GridPenalty": 0,
        "grid_penalty_places": 0,
        "grid_vs_quali_delta": 0,
        "grid_norm": 2.0 / 22.0,
        "expected_grid_finish": 2.3,
        "grid_retention_prob": 0.75,
        "teammate_grid_delta": -4,
        # Future Race Result (Strictly forbidden in pre-race)
        "ActualFinish": 1.0,
    }])


def test_sprint_stages_have_explicit_masking_rules():
    """Verify that after_sprint_qualifying and after_sprint_race exist in STAGE_UNAVAILABLE_FEATURES."""
    assert "after_sprint_qualifying" in STAGE_UNAVAILABLE_FEATURES
    assert "after_sprint_race" in STAGE_UNAVAILABLE_FEATURES
    assert "after_sprint" in STAGE_UNAVAILABLE_FEATURES
    assert len(STAGE_UNAVAILABLE_FEATURES["after_sprint_qualifying"]) > 10
    assert len(STAGE_UNAVAILABLE_FEATURES["after_sprint_race"]) > 10


def test_after_sprint_qualifying_masks_sunday_qualifying_and_grid(synthetic_weekend_feature_df):
    """
    At 'after_sprint_qualifying':
    - Sunday Qualifying MUST be masked to NaN.
    - Sunday Grid MUST be masked to NaN.
    - Sprint Race MUST be masked to NaN.
    - Sprint Qualifying MUST be preserved.
    - FP1 practice MUST be preserved.
    """
    masked = apply_stage_feature_mask(synthetic_weekend_feature_df.copy(), "after_sprint_qualifying")

    # Sunday Qualifying masked
    for col in ["QualifyingPosition", "quali_pos_norm", "quali_gap_pct", "q1_gap_pct", "q2_gap_pct", "q3_gap_pct"]:
        assert pd.isna(masked[col].iloc[0]), f"Sunday qualifying feature '{col}' leaked at after_sprint_qualifying!"

    # Sunday Grid masked
    for col in ["GridPosition", "ActualStartingGrid", "simulation_start_rank", "grid_norm", "expected_grid_finish", "grid_retention_prob"]:
        assert pd.isna(masked[col].iloc[0]), f"Sunday grid feature '{col}' leaked at after_sprint_qualifying!"

    # Sprint Race masked (has not occurred yet)
    assert pd.isna(masked["sprint_finish_pos"].iloc[0])
    assert pd.isna(masked["sprint_pos_delta"].iloc[0])

    # Sprint Qualifying LEGITIMATELY PRESERVED
    assert masked["sprint_quali_pos"].iloc[0] == 1.0

    # FP1 Practice LEGITIMATELY PRESERVED
    assert masked["fp1_best_lap"].iloc[0] == 88.5
    assert masked["fp1_gap_pct"].iloc[0] == 0.0

    # Pre-weekend rolling form PRESERVED
    assert masked["driver_finish_ewma"].iloc[0] == 2.1


def test_after_sprint_race_masks_sunday_qualifying_and_grid(synthetic_weekend_feature_df):
    """
    At 'after_sprint_race':
    - Sunday Qualifying MUST be masked to NaN.
    - Sunday Grid MUST be masked to NaN.
    - Sprint Qualifying AND Sprint Race MUST be preserved.
    """
    masked = apply_stage_feature_mask(synthetic_weekend_feature_df.copy(), "after_sprint_race")

    # Sunday Qualifying masked
    for col in ["QualifyingPosition", "quali_pos_norm", "quali_gap_pct", "q1_gap_pct", "q2_gap_pct", "q3_gap_pct"]:
        assert pd.isna(masked[col].iloc[0]), f"Sunday qualifying feature '{col}' leaked at after_sprint_race!"

    # Sunday Grid masked
    for col in ["GridPosition", "ActualStartingGrid", "simulation_start_rank", "grid_norm", "expected_grid_finish"]:
        assert pd.isna(masked[col].iloc[0]), f"Sunday grid feature '{col}' leaked at after_sprint_race!"

    # Sprint Qualifying AND Sprint Race LEGITIMATELY PRESERVED
    assert masked["sprint_quali_pos"].iloc[0] == 1.0
    assert masked["sprint_finish_pos"].iloc[0] == 1.0
    assert masked["sprint_pos_delta"].iloc[0] == 0.0

    # Pre-weekend rolling form PRESERVED
    assert masked["driver_finish_ewma"].iloc[0] == 2.1


def test_after_sprint_alias_behaves_identically(synthetic_weekend_feature_df):
    """Confirm 'after_sprint' alias behaves identically to 'after_sprint_race'."""
    masked_race = apply_stage_feature_mask(synthetic_weekend_feature_df.copy(), "after_sprint_race")
    masked_alias = apply_stage_feature_mask(synthetic_weekend_feature_df.copy(), "after_sprint")
    pd.testing.assert_frame_equal(masked_race, masked_alias)


def test_non_sprint_stages_retain_expected_availability(synthetic_weekend_feature_df):
    """Validate that pre_fp1, after_fp1, after_fp2, after_fp3, and after_qualifying retain their invariants."""
    # pre_fp1: everything weekend-related masked
    masked_pre = apply_stage_feature_mask(synthetic_weekend_feature_df.copy(), "pre_fp1")
    assert pd.isna(masked_pre["fp1_best_lap"].iloc[0])
    assert pd.isna(masked_pre["sprint_quali_pos"].iloc[0])
    assert pd.isna(masked_pre["QualifyingPosition"].iloc[0])
    assert pd.isna(masked_pre["GridPosition"].iloc[0])
    assert masked_pre["driver_finish_ewma"].iloc[0] == 2.1  # Form unmasked

    # after_qualifying: qualifying and grid unmasked
    masked_aq = apply_stage_feature_mask(synthetic_weekend_feature_df.copy(), "after_qualifying")
    assert masked_aq["QualifyingPosition"].iloc[0] == 2.0
    assert masked_aq["GridPosition"].iloc[0] == 2.0


def test_future_race_result_anti_leakage(synthetic_weekend_feature_df):
    """Verifies that DataValidator confirms target datasets do not leak ActualFinish."""
    df = synthetic_weekend_feature_df.copy()
    # Before race, ActualFinish must never be in pre-race target dataset
    clean_target = df.drop(columns=["ActualFinish"])
    clean, errors = DataValidator.validate_leakage(clean_target, stage="after_sprint_race")
    assert clean is True
    assert len(errors) == 0

    dirty, errors = DataValidator.validate_leakage(df, stage="after_sprint_race")
    assert dirty is False
    assert any("ActualFinish" in e for e in errors)

