"""
Unit Test Suite for Dynamic F1 Season Discovery
================================================
Validates:
1. Discovery of multiple supported seasons (contiguous and non-contiguous).
2. Missing season directories (returns empty / handles gracefully).
3. Empty datasets (directory exists with 0 files).
4. Incomplete season data (0-byte files or missing qualifying/race CSVs).
5. Non-contiguous supported seasons formatting.
6. 2026 default selection logic when available vs fallback.
7. No available seasons (clean message and graceful handling).
8. Historical data-source compatibility (>= 2018 FastF1 detailed vs < 2018 results-only).
"""

from pathlib import Path
import pytest
import pandas as pd

from src.data.season_discovery import (
    SeasonDiscovery,
    FASTF1_MIN_YEAR,
    SOURCE_FASTF1_DETAILED,
    SOURCE_RESULTS_ONLY,
    SOURCE_INCOMPLETE,
    SOURCE_NONE,
)
from src.cli.terminal_ui import prompt_int


def create_mock_season(season_dir: Path, year: int, rounds: list, valid_content: bool = True, zero_byte: bool = False):
    """Helper to create mock round files in a season directory."""
    y_dir = season_dir / str(year)
    y_dir.mkdir(parents=True, exist_ok=True)

    for r in rounds:
        rf = y_dir / f"{year}_round_{r}_race.csv"
        qf = y_dir / f"{year}_round_{r}_qualifying.csv"
        fp1 = y_dir / f"{year}_round_{r}_fp1.csv"

        if zero_byte:
            rf.touch()
            qf.touch()
            fp1.touch()
        elif valid_content:
            sample_df = pd.DataFrame({
                "Abbreviation": ["VER", "HAM", "LEC"],
                "Position": [1, 2, 3],
                "GridPosition": [1, 2, 3],
                "TeamName": ["Red Bull Racing", "Mercedes", "Ferrari"],
            })
            sample_df.to_csv(rf, index=False)
            sample_df.to_csv(qf, index=False)
            sample_df.to_csv(fp1, index=False)
        else:
            # Header only with 0 rows
            empty_df = pd.DataFrame(columns=["Abbreviation", "Position", "GridPosition"])
            empty_df.to_csv(rf, index=False)
            empty_df.to_csv(qf, index=False)


class TestSeasonDiscovery:
    """Tests for dynamic championship season discovery."""

    def test_discovery_multiple_contiguous_seasons(self, tmp_path):
        """TC-SD-01: Discovers multiple contiguous supported seasons."""
        raw_season = tmp_path / "data" / "raw" / "season"
        for yr in [2022, 2023, 2024, 2025, 2026]:
            create_mock_season(raw_season, yr, rounds=[1, 2])

        discovery = SeasonDiscovery(root_dir=tmp_path)
        seasons = discovery.discover_seasons(workflow="prediction")

        assert seasons == [2022, 2023, 2024, 2025, 2026]
        assert SeasonDiscovery.is_contiguous(seasons) is True
        assert SeasonDiscovery.format_available_seasons(seasons) == "2022–2026"
        assert SeasonDiscovery.format_prompt_hint(seasons) == " (2022-2026)"
        assert SeasonDiscovery.get_default_season(seasons) == 2026

    def test_missing_season_directories(self, tmp_path):
        """TC-SD-02: Missing season directories return empty list gracefully without error."""
        discovery = SeasonDiscovery(root_dir=tmp_path)
        seasons = discovery.discover_seasons(workflow="prediction")

        assert seasons == []
        assert SeasonDiscovery.format_available_seasons(seasons) == "None available"
        assert SeasonDiscovery.format_prompt_hint(seasons) == ""
        assert SeasonDiscovery.get_default_season(seasons) is None

        status = discovery.get_season_status(2026)
        assert status["exists"] is False
        assert status["usable"] is False
        assert status["source_type"] == SOURCE_NONE

    def test_empty_dataset_directory(self, tmp_path):
        """TC-SD-03: Empty season directory with zero files is excluded from discovery."""
        empty_dir = tmp_path / "data" / "raw" / "season" / "2024"
        empty_dir.mkdir(parents=True, exist_ok=True)

        discovery = SeasonDiscovery(root_dir=tmp_path)
        seasons = discovery.discover_seasons(workflow="prediction")

        assert seasons == []
        status = discovery.get_season_status(2024)
        assert status["exists"] is True
        assert status["usable"] is False
        assert status["source_type"] == SOURCE_INCOMPLETE
        assert "zero CSV files" in status["error_reason"]

    def test_incomplete_season_zero_byte_files(self, tmp_path):
        """TC-SD-04: Season with 0-byte (empty) files is rejected as incomplete."""
        raw_season = tmp_path / "data" / "raw" / "season"
        create_mock_season(raw_season, 2025, rounds=[1], zero_byte=True)

        discovery = SeasonDiscovery(root_dir=tmp_path)
        seasons = discovery.discover_seasons(workflow="prediction")

        assert 2025 not in seasons
        status = discovery.get_season_status(2025)
        assert status["exists"] is True
        assert status["usable"] is False
        assert status["valid_files"] == 0
        assert "0 bytes" in status["error_reason"]

    def test_incomplete_season_missing_qualifying(self, tmp_path):
        """TC-SD-04b: Season with race files but missing qualifying files is rejected."""
        raw_season = tmp_path / "data" / "raw" / "season"
        y_dir = raw_season / "2025"
        y_dir.mkdir(parents=True, exist_ok=True)

        # Race exists but no qualifying
        df = pd.DataFrame({"Abbreviation": ["VER"], "Position": [1]})
        df.to_csv(y_dir / "2025_round_1_race.csv", index=False)

        discovery = SeasonDiscovery(root_dir=tmp_path)
        seasons = discovery.discover_seasons(workflow="prediction")

        assert 2025 not in seasons
        status = discovery.get_season_status(2025)
        assert status["usable"] is False
        assert len(status["completed_rounds"]) == 0

    def test_non_contiguous_supported_seasons(self, tmp_path):
        """TC-SD-05: Non-contiguous seasons formatted as explicit comma-separated list."""
        raw_season = tmp_path / "data" / "raw" / "season"
        for yr in [2018, 2022, 2026]:
            create_mock_season(raw_season, yr, rounds=[1, 2])

        discovery = SeasonDiscovery(root_dir=tmp_path)
        seasons = discovery.discover_seasons(workflow="prediction")

        assert seasons == [2018, 2022, 2026]
        assert SeasonDiscovery.is_contiguous(seasons) is False
        assert SeasonDiscovery.format_available_seasons(seasons) == "2018, 2022, 2026"
        assert SeasonDiscovery.format_prompt_hint(seasons) == " (2018, 2022, 2026)"
        assert SeasonDiscovery.get_default_season(seasons) == 2026

    def test_single_season_formatting_avoids_redundant_range(self, tmp_path):
        """TC-SD-06: Single season formatted cleanly without redundant '(2026-2026)'."""
        raw_season = tmp_path / "data" / "raw" / "season"
        create_mock_season(raw_season, 2026, rounds=[1])

        discovery = SeasonDiscovery(root_dir=tmp_path)
        seasons = discovery.discover_seasons(workflow="prediction")

        assert seasons == [2026]
        assert SeasonDiscovery.format_available_seasons(seasons) == "2026"
        assert SeasonDiscovery.format_prompt_hint(seasons) == ""
        assert SeasonDiscovery.get_default_season(seasons) == 2026

    def test_default_season_selection(self):
        """TC-SD-07: 2026 is preferred default; otherwise latest available season is chosen."""
        # 2026 present
        assert SeasonDiscovery.get_default_season([2018, 2022, 2026]) == 2026
        assert SeasonDiscovery.get_default_season([2026]) == 2026

        # 2026 absent -> picks max
        assert SeasonDiscovery.get_default_season([2022, 2023, 2024]) == 2024
        assert SeasonDiscovery.get_default_season([2019]) == 2019

        # Empty -> None
        assert SeasonDiscovery.get_default_season([]) is None

    def test_historical_data_source_boundary(self, tmp_path):
        """
        TC-SD-08: Distinguishes FastF1 (>= 2018) from pre-2018 results-only (Jolpica/Ergast).
        Pre-2018 seasons must not be accepted for telemetry-based prediction workflows.
        """
        raw_season = tmp_path / "data" / "raw" / "season"

        # Create 2016 (< 2018, pre-FastF1 results only) and 2019 (>= 2018 FastF1)
        create_mock_season(raw_season, 2016, rounds=[1])
        create_mock_season(raw_season, 2019, rounds=[1])

        discovery = SeasonDiscovery(root_dir=tmp_path)

        # 1. Prediction workflow rejects 2016 but accepts 2019
        pred_seasons = discovery.discover_seasons(workflow="prediction")
        assert 2016 not in pred_seasons
        assert 2019 in pred_seasons

        # 2. Check source classification
        s2016 = discovery.get_season_status(2016, workflow="prediction")
        assert s2016["source_type"] == SOURCE_RESULTS_ONLY
        assert s2016["usable"] is False
        assert f"{FASTF1_MIN_YEAR}" in s2016["error_reason"]

        s2019 = discovery.get_season_status(2019, workflow="prediction")
        assert s2019["source_type"] == SOURCE_FASTF1_DETAILED
        assert s2019["usable"] is True

        # 3. Results-only workflow allows 2016
        results_seasons = discovery.discover_seasons(workflow="historical_results")
        assert 2016 in results_seasons
        assert 2019 in results_seasons

    def test_prompt_int_formatting_and_validation(self, monkeypatch):
        """TC-SD-09: prompt_int formats ranges correctly and enforces allowed_values."""
        # 1. Single season: hint is empty, default accepted on Enter
        monkeypatch.setattr("builtins.input", lambda prompt: "")
        val = prompt_int("Enter Season Year", allowed_values=[2026], default=2026)
        assert val == 2026

        # 2. Rejects out-of-range value and accepts valid one
        inputs = iter(["2020", "2026"])
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
        val = prompt_int("Enter Season Year", allowed_values=[2026], default=2026)
        assert val == 2026

        # 3. Non-contiguous allowed values
        inputs2 = iter(["2020", "2022"])
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs2))
        val2 = prompt_int("Enter Season Year", allowed_values=[2018, 2022, 2026], default=2026)
        assert val2 == 2022


class TestStageFeatureAvailability:
    """Tests for stage-specific feature availability and temporal leakage prevention."""

    def test_stage_masking_pre_fp1(self):
        """TC-STAGE-01: pre_fp1 masks practice, sprint, qualifying, and starting grid."""
        from src.features.build_features import apply_stage_feature_mask

        df = pd.DataFrame([{
            "Abbreviation": "VER",
            "driver_finish_ewma": 1.5,
            "team_finish_ewma": 2.0,
            "fp1_gap_pct": 0.001,
            "fp2_gap_pct": 0.002,
            "fp3_gap_pct": 0.003,
            "QualifyingPosition": 1.0,
            "GridPosition": 1.0,
            "ActualStartingGrid": 1.0,
            "expected_grid_finish": 1.8,
        }])

        masked = apply_stage_feature_mask(df.copy(), stage="pre_fp1")

        # Rolling form preserved
        assert masked["driver_finish_ewma"].iloc[0] == 1.5
        assert masked["team_finish_ewma"].iloc[0] == 2.0

        # Weekend telemetry and grid masked to NaN
        assert pd.isna(masked["fp1_gap_pct"].iloc[0])
        assert pd.isna(masked["fp2_gap_pct"].iloc[0])
        assert pd.isna(masked["fp3_gap_pct"].iloc[0])
        assert pd.isna(masked["QualifyingPosition"].iloc[0])
        assert pd.isna(masked["GridPosition"].iloc[0])
        assert pd.isna(masked["ActualStartingGrid"].iloc[0])
        assert pd.isna(masked["expected_grid_finish"].iloc[0])

    def test_stage_masking_after_fp1(self):
        """TC-STAGE-02: after_fp1 preserves FP1 but masks FP2, FP3, qualifying, and grid."""
        from src.features.build_features import apply_stage_feature_mask

        df = pd.DataFrame([{
            "Abbreviation": "VER",
            "fp1_gap_pct": 0.001,
            "fp2_gap_pct": 0.002,
            "fp3_gap_pct": 0.003,
            "QualifyingPosition": 1.0,
            "GridPosition": 1.0,
        }])

        masked = apply_stage_feature_mask(df.copy(), stage="after_fp1")

        assert masked["fp1_gap_pct"].iloc[0] == 0.001
        assert pd.isna(masked["fp2_gap_pct"].iloc[0])
        assert pd.isna(masked["fp3_gap_pct"].iloc[0])
        assert pd.isna(masked["QualifyingPosition"].iloc[0])
        assert pd.isna(masked["GridPosition"].iloc[0])

    def test_stage_masking_after_qualifying(self):
        """TC-STAGE-03: after_qualifying preserves practice, qualifying, and grid."""
        from src.features.build_features import apply_stage_feature_mask

        df = pd.DataFrame([{
            "Abbreviation": "VER",
            "fp1_gap_pct": 0.001,
            "QualifyingPosition": 1.0,
            "GridPosition": 1.0,
            "ActualStartingGrid": 1.0,
        }])

        masked = apply_stage_feature_mask(df.copy(), stage="after_qualifying")

        assert masked["fp1_gap_pct"].iloc[0] == 0.001
        assert masked["QualifyingPosition"].iloc[0] == 1.0
        assert masked["GridPosition"].iloc[0] == 1.0
        assert masked["ActualStartingGrid"].iloc[0] == 1.0
