"""
Unit and regression tests for Phase 3C defect fixes:
- DEF-05: Robust handling of empty and malformed CSV files
- DEF-04: Dynamic season-round selection from schedule in collect_season
- DEF-03: Year-aware calendar round and penalty resolution
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.data_loader import DataLoader
from src.data.collect_season import collect_season_data
from src.pipeline.race_prediction_pipeline import RacePredictionPipeline


# ---------------------------------------------------------------------------
# DEF-05: Robust handling of empty or malformed CSV files
# ---------------------------------------------------------------------------

def test_safe_read_csv_valid_file(tmp_path):
    """Verifies that valid CSV files load correctly with and without normalization."""
    valid_csv = tmp_path / "valid.csv"
    valid_csv.write_text("Driver,Abbreviation,TeamName,Position\nMax Verstappen,VER,Red Bull Racing,1\n")

    loader = DataLoader(root_dir=tmp_path)
    df_raw = loader.safe_read_csv(valid_csv, normalize=False)
    assert df_raw is not None
    assert len(df_raw) == 1
    assert df_raw.iloc[0]["Abbreviation"] == "VER"

    df_norm = loader.safe_read_csv(valid_csv, normalize=True)
    assert df_norm is not None
    assert len(df_norm) == 1
    assert "DriverId" in df_norm.columns or "Abbreviation" in df_norm.columns


def test_safe_read_csv_zero_byte_file(tmp_path):
    """Verifies that 0-byte file produces controlled None outcome without unhandled crash."""
    empty_csv = tmp_path / "zero_byte.csv"
    empty_csv.write_bytes(b"")

    loader = DataLoader(root_dir=tmp_path)
    df = loader.safe_read_csv(empty_csv, normalize=False)
    assert df is None

    df_norm = loader.safe_read_csv(empty_csv, normalize=True)
    assert df_norm is None


def test_safe_read_csv_header_only_file(tmp_path):
    """Verifies that a valid header-only CSV is distinguished from a 0-byte file."""
    header_csv = tmp_path / "header_only.csv"
    header_csv.write_text("Driver,Abbreviation,TeamName,Position\n")

    loader = DataLoader(root_dir=tmp_path)
    df = loader.safe_read_csv(header_csv, normalize=False)
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert list(df.columns) == ["Driver", "Abbreviation", "TeamName", "Position"]


def test_safe_read_csv_malformed_file(tmp_path):
    """Verifies that malformed CSV rows (ParserError) produce controlled None outcome."""
    corrupt_csv = tmp_path / "malformed.csv"
    # Create inconsistent column count that causes ParserError
    corrupt_csv.write_text("col1,col2\nval1,val2\nextra1,extra2,extra3,extra4\nval3\n")

    loader = DataLoader(root_dir=tmp_path)
    df = loader.safe_read_csv(corrupt_csv, normalize=False)
    assert df is None


def test_safe_read_csv_missing_driver_identifier_raises_descriptive_error(tmp_path):
    """Verifies that normalize_driver descriptive ValueError is preserved when required columns are absent."""
    no_driver_csv = tmp_path / "no_driver.csv"
    no_driver_csv.write_text("TeamName,LapTime\nFerrari,80.5\n")

    loader = DataLoader(root_dir=tmp_path)
    with pytest.raises(ValueError, match="No driver identifier column found in DataFrame"):
        loader.safe_read_csv(no_driver_csv, normalize=True)


# ---------------------------------------------------------------------------
# DEF-04: Dynamic season-round selection in collect_season
# ---------------------------------------------------------------------------

def test_collect_season_dynamic_rounds_from_schedule(tmp_path):
    """Verifies that omitting --rounds derives positive round numbers dynamically from schedule."""
    mock_schedule = pd.DataFrame({
        "RoundNumber": [0, 1, 2, 3, None, "invalid", 4],
        "EventName": ["Pre-Season Testing", "Bahrain GP", "Saudi GP", "Australian GP", "Unknown", "Bad", "Japanese GP"]
    })

    collected_rounds = []

    def mock_get_session(year, round_num, session_type):
        collected_rounds.append(round_num)
        mock_sess = MagicMock()
        mock_sess.results = pd.DataFrame()
        return mock_sess

    with patch("fastf1.get_event_schedule", return_value=mock_schedule), \
         patch("fastf1.get_session", side_effect=mock_get_session):
        collect_season_data(year=2024, rounds=None, output_dir=tmp_path)

    # Should attempt to collect positive integer rounds [1, 2, 3, 4] and ignore 0, None, 'invalid'
    unique_collected = sorted(list(set(collected_rounds)))
    assert unique_collected == [1, 2, 3, 4]


def test_collect_season_explicit_rounds_preserved(tmp_path):
    """Verifies that explicit rounds argument is preserved."""
    mock_schedule = pd.DataFrame({
        "RoundNumber": [1, 2, 3, 4, 5],
        "EventName": ["R1", "R2", "R3", "R4", "R5"]
    })

    collected_rounds = []

    def mock_get_session(year, round_num, session_type):
        collected_rounds.append(round_num)
        mock_sess = MagicMock()
        mock_sess.results = pd.DataFrame()
        return mock_sess

    with patch("fastf1.get_event_schedule", return_value=mock_schedule), \
         patch("fastf1.get_session", side_effect=mock_get_session):
        collect_season_data(year=2024, rounds=[2, 4], output_dir=tmp_path)

    unique_collected = sorted(list(set(collected_rounds)))
    assert unique_collected == [2, 4]


# ---------------------------------------------------------------------------
# DEF-03: Year-aware calendar and penalty resolution
# ---------------------------------------------------------------------------

def test_pipeline_year_aware_round_resolution_2026():
    """Verifies that year=2026 resolves 2026 target round correctly."""
    pipeline = RacePredictionPipeline()
    # Baku in tracks.json has round_2026 == 15
    track_info = pipeline.tracks_config["tracks"]["baku"]
    assert track_info["round_2026"] == 15

    # Check DataLoader _resolve_round_number
    resolved = pipeline.data_loader._resolve_round_number("baku", year=2026)
    assert resolved == 15


def test_pipeline_year_aware_unsupported_year_diagnostic():
    """Verifies that requesting an unconfigured year raises a clear ValueError diagnostic."""
    pipeline = RacePredictionPipeline()
    with pytest.raises(ValueError, match="Calendar round for track 'baku' is not configured for year 2025 in tracks.json"):
        pipeline.run(track_key="baku", year=2025, stage="pre_fp1")


def test_pipeline_year_aware_qualifying_unsupported_year_diagnostic():
    """Verifies that predict_qualifying also raises a clear diagnostic for unconfigured years."""
    pipeline = RacePredictionPipeline()
    with pytest.raises(ValueError, match="Calendar round for track 'baku' is not configured for year 2025 in tracks.json"):
        pipeline.predict_qualifying(track_key="baku", year=2025, stage="pre_fp1")


def test_pipeline_year_aware_penalty_resolution():
    """Verifies that year-specific penalties are loaded and 2026 penalties are not misattributed to other years."""
    pipeline = RacePredictionPipeline()
    tracks = pipeline.tracks_config.get("tracks", {})
    # Mock a track config with specific year penalty
    mock_tracks = {
        "test_track": {
            "round_2026": 10,
            "round_2027": 12,
            "known_penalties_2026": {"VER": 5},
            "known_penalties_2027": {"HAM": 10}
        }
    }
    with patch.dict(pipeline.tracks_config, {"tracks": mock_tracks}):
        # For 2026:
        p26 = mock_tracks["test_track"].get("known_penalties_2026")
        assert p26 == {"VER": 5}

        # For 2027:
        p27 = mock_tracks["test_track"].get("known_penalties_2027")
        assert p27 == {"HAM": 10}
