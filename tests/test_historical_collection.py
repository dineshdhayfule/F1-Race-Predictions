"""
Tests for Historical Circuit Data Collection and Menu Integration.
Verifies dynamic circuit configuration, year range formatting, safe re-runs,
multi-circuit selection, loader compatibility, and error handling.
All FastF1 network calls are strictly mocked and filesystem operations use tmp_path.
"""

from pathlib import Path
import json
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from src.data.collect_track import format_year_range, collect_historical_track_data
from src.data.data_loader import DataLoader
from src.cli.menu import InteractiveMenu


# =============================================================================
# 1. Year Range Formatting Tests
# =============================================================================

def test_format_year_range_empty_or_none():
    assert format_year_range(None) == "None configured"
    assert format_year_range([]) == "None configured"
    assert format_year_range(["invalid", None]) == "None configured"


def test_format_year_range_single_year():
    assert format_year_range([2024]) == "2024"


def test_format_year_range_contiguous_range():
    assert format_year_range([2022, 2023, 2024, 2025]) == "2022–2025"
    # Unordered with duplicates
    assert format_year_range([2025, 2022, 2023, 2024, 2023]) == "2022–2025"


def test_format_year_range_split_ranges():
    # Singapore pattern
    years = [2018, 2019, 2022, 2023, 2024, 2025]
    assert format_year_range(years) == "2018–2019, 2022–2025"


def test_format_year_range_multiple_disjoint():
    years = [2018, 2020, 2022, 2024]
    assert format_year_range(years) == "2018, 2020, 2022, 2024"


# =============================================================================
# 2. Configuration & Skip Handling Tests
# =============================================================================

def test_collect_unknown_track():
    result = collect_historical_track_data("nonexistent_circuit", tracks_config={})
    assert result["track"] == "nonexistent_circuit"
    assert len(result["failures"]) == 1
    assert "Unknown circuit key" in result["failures"][0]
    assert len(result["successes"]) == 0


def test_collect_unconfigured_or_disabled_track(tmp_path):
    mock_config = {
        "test_track": {
            "display_name": "Test Grand Prix",
            "historical_data_available": False,
            "historical_years": []
        }
    }
    result = collect_historical_track_data(
        "test_track",
        output_dir=tmp_path,
        tracks_config=mock_config
    )
    assert result["track"] == "test_track"
    assert len(result["skips"]) == 1
    assert "No historical years configured" in result["skips"][0]
    assert len(result["successes"]) == 0
    assert len(result["failures"]) == 0


def test_collect_missing_historical_years_field(tmp_path):
    mock_config = {
        "test_track": {
            "display_name": "Test Grand Prix",
            "historical_data_available": True
            # "historical_years" missing
        }
    }
    result = collect_historical_track_data(
        "test_track",
        output_dir=tmp_path,
        tracks_config=mock_config
    )
    assert len(result["skips"]) == 1
    assert "No historical years configured" in result["skips"][0]


# =============================================================================
# 3. Safe Re-runs: Existing Files Preserved
# =============================================================================

def test_safe_reruns_preserves_existing_files(tmp_path):
    mock_config = {
        "silverstone": {
            "display_name": "British Grand Prix",
            "historical_data_available": True,
            "historical_years": [2023],
            "event_name_patterns": ["British"]
        }
    }
    hist_dir = tmp_path / "tracks" / "silverstone" / "history"
    hist_dir.mkdir(parents=True)
    race_file = hist_dir / "silverstone_2023_race.csv"
    quali_file = hist_dir / "silverstone_2023_qualifying.csv"

    # Pre-populate with sentinel text
    sentinel_race = "DriverNumber,Position\n44,1\n"
    sentinel_quali = "DriverNumber,Position\n44,1\n"
    race_file.write_text(sentinel_race, encoding="utf-8")
    quali_file.write_text(sentinel_quali, encoding="utf-8")

    # Mock FastF1 schedule lookup
    mock_schedule = pd.DataFrame([{
        "RoundNumber": 10,
        "EventName": "British Grand Prix",
        "Location": "Silverstone"
    }])

    with patch("fastf1.get_event_schedule", return_value=mock_schedule), \
         patch("fastf1.get_session") as mock_get_session:

        result = collect_historical_track_data(
            "silverstone",
            years=[2023],
            output_dir=hist_dir,
            force=False,
            tracks_config=mock_config
        )

        # FastF1 should NOT be called because files already exist
        mock_get_session.assert_not_called()
        assert len(result["skips"]) == 2
        assert "already exists" in result["skips"][0]
        assert "already exists" in result["skips"][1]

        # File contents must remain unchanged
        assert race_file.read_text(encoding="utf-8") == sentinel_race
        assert quali_file.read_text(encoding="utf-8") == sentinel_quali


def test_safe_reruns_overwrites_with_force(tmp_path):
    mock_config = {
        "silverstone": {
            "display_name": "British Grand Prix",
            "historical_data_available": True,
            "historical_years": [2023],
            "event_name_patterns": ["British"]
        }
    }
    hist_dir = tmp_path / "tracks" / "silverstone" / "history"
    hist_dir.mkdir(parents=True)
    race_file = hist_dir / "silverstone_2023_race.csv"
    race_file.write_text("Old,Data\n1,1\n", encoding="utf-8")

    mock_schedule = pd.DataFrame([{
        "RoundNumber": 10,
        "EventName": "British Grand Prix",
        "Location": "Silverstone"
    }])

    mock_sess = MagicMock()
    mock_sess.results = pd.DataFrame([
        {"DriverNumber": 44, "Abbreviation": "HAM", "Position": 1.0, "TeamName": "Mercedes"}
    ])

    with patch("fastf1.get_event_schedule", return_value=mock_schedule), \
         patch("fastf1.get_session", return_value=mock_sess):

        result = collect_historical_track_data(
            "silverstone",
            years=[2023],
            output_dir=hist_dir,
            force=True,
            tracks_config=mock_config
        )

        assert any("2023 Race (1 drivers)" in s for s in result["successes"])
        new_content = race_file.read_text(encoding="utf-8")
        assert "HAM" in new_content


# =============================================================================
# 4. Unavailable Seasons and Sessions Handling
# =============================================================================

def test_unavailable_season_event_skips_gracefully(tmp_path):
    mock_config = {
        "china": {
            "display_name": "Chinese Grand Prix",
            "historical_data_available": True,
            "historical_years": [2021, 2024],
            "event_name_patterns": ["Chinese", "Shanghai"]
        }
    }

    # 2021 was cancelled (not on schedule)
    empty_schedule = pd.DataFrame([{"RoundNumber": 1, "EventName": "Bahrain GP", "Location": "Sakhir"}])
    mock_2024_schedule = pd.DataFrame([{"RoundNumber": 5, "EventName": "Chinese Grand Prix", "Location": "Shanghai"}])

    def mock_get_event_schedule(year):
        if year == 2021:
            return empty_schedule
        return mock_2024_schedule

    mock_sess = MagicMock()
    mock_sess.results = pd.DataFrame([{"DriverNumber": 1, "Position": 1.0}])

    with patch("fastf1.get_event_schedule", side_effect=mock_get_event_schedule), \
         patch("fastf1.get_session", return_value=mock_sess):

        result = collect_historical_track_data(
            "china",
            output_dir=tmp_path,
            tracks_config=mock_config
        )

        assert any("2021" in s and "Could not locate event" in s for s in result["skips"])
        assert any("2024 Race" in s for s in result["successes"])


def test_session_exception_records_failure_without_crashing(tmp_path):
    mock_config = {
        "monaco": {
            "display_name": "Monaco Grand Prix",
            "historical_data_available": True,
            "historical_years": [2023],
            "event_name_patterns": ["Monaco"]
        }
    }
    mock_schedule = pd.DataFrame([{"RoundNumber": 6, "EventName": "Monaco Grand Prix", "Location": "Monaco"}])

    with patch("fastf1.get_event_schedule", return_value=mock_schedule), \
         patch("fastf1.get_session", side_effect=RuntimeError("FastF1 network timeout")):

        result = collect_historical_track_data(
            "monaco",
            output_dir=tmp_path,
            tracks_config=mock_config
        )

        assert len(result["failures"]) == 2
        assert "FastF1 network timeout" in result["failures"][0]
        assert "FastF1 network timeout" in result["failures"][1]


# =============================================================================
# 5. Loader Compatibility Test
# =============================================================================

def test_successful_collection_produces_files_compatible_with_loader(tmp_path):
    # Setup mock workspace structure
    raw_tracks = tmp_path / "data" / "raw" / "tracks" / "austria" / "history"
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    raw_tracks.mkdir(parents=True)

    tracks_data = {
        "tracks": {
            "austria": {
                "display_name": "Austrian Grand Prix",
                "historical_data_available": True,
                "historical_years": [2023, 2024],
                "event_name_patterns": ["Austrian", "Spielberg"]
            }
        }
    }
    (config_dir / "tracks.json").write_text(json.dumps(tracks_data), encoding="utf-8")

    mock_schedule = pd.DataFrame([{"RoundNumber": 9, "EventName": "Austrian Grand Prix", "Location": "Spielberg"}])

    r_res = pd.DataFrame([
        {"DriverNumber": 1, "Abbreviation": "VER", "Position": 1.0, "GridPosition": 1.0, "TeamName": "Red Bull Racing"},
        {"DriverNumber": 16, "Abbreviation": "LEC", "Position": 2.0, "GridPosition": 2.0, "TeamName": "Ferrari"}
    ])
    q_res = pd.DataFrame([
        {"DriverNumber": 1, "Abbreviation": "VER", "Position": 1.0, "TeamName": "Red Bull Racing"},
        {"DriverNumber": 16, "Abbreviation": "LEC", "Position": 2.0, "TeamName": "Ferrari"}
    ])

    def mock_get_session(year, round_num, session_type):
        sess = MagicMock()
        sess.results = r_res if session_type == "R" else q_res
        return sess

    with patch("fastf1.get_event_schedule", return_value=mock_schedule), \
         patch("fastf1.get_session", side_effect=mock_get_session):

        result = collect_historical_track_data(
            "austria",
            output_dir=raw_tracks,
            tracks_config=tracks_data["tracks"]
        )

        assert len(result["successes"]) == 4  # 2 years * 2 sessions

        # Verify individual files exist
        assert (raw_tracks / "austria_2023_race.csv").exists()
        assert (raw_tracks / "austria_2023_qualifying.csv").exists()
        assert (raw_tracks / "austria_2024_race.csv").exists()
        assert (raw_tracks / "austria_2024_qualifying.csv").exists()

        # Verify combined files exist
        hist_race = raw_tracks / "austria_historical_races.csv"
        hist_quali = raw_tracks / "austria_historical_qualifying.csv"
        assert hist_race.exists()
        assert hist_quali.exists()

        # Test compatibility with DataLoader
        loader = DataLoader(root_dir=tmp_path)
        df_r, df_q = loader.load_historical_track_data("austria")

        assert df_r is not None and not df_r.empty
        assert df_q is not None and not df_q.empty
        assert len(df_r) == 4  # 2 drivers * 2 years
        assert len(df_q) == 4
        assert "Position" in df_r.columns
        assert "Year" in df_r.columns
        assert "Race" in df_r.columns
        assert set(df_r["Year"].unique()) == {2023, 2024}


# =============================================================================
# 6. Interactive Menu Submenu and Input Validation Tests
# =============================================================================

def test_menu_derived_circuits_and_exit(tmp_path):
    mock_tracks = {
        "track_b": {"display_name": "Circuit B", "historical_data_available": True, "historical_years": [2022, 2023]},
        "track_a": {"display_name": "Circuit A", "historical_data_available": True, "historical_years": [2024]},
        "track_c": {"display_name": "Circuit C", "historical_data_available": False, "historical_years": []},
    }

    menu = InteractiveMenu(root_dir=tmp_path)
    menu.tracks_cfg = mock_tracks

    # Simulate entering "0" (Exit immediately)
    with patch("src.cli.menu.prompt_str", return_value="0") as mock_prompt:
        menu._run_historical_circuit_collector()
        mock_prompt.assert_called_once()


def test_menu_invalid_selection_handled_gracefully(tmp_path):
    mock_tracks = {
        "australia": {"display_name": "Australian Grand Prix", "historical_data_available": True, "historical_years": [2024]}
    }
    menu = InteractiveMenu(root_dir=tmp_path)
    menu.tracks_cfg = mock_tracks

    # Simulate invalid input: "99 abc" then [Enter]
    with patch("src.cli.menu.prompt_str", side_effect=["99 abc", ""]), \
         patch("src.cli.menu.print_warning") as mock_warn, \
         patch("src.cli.menu.print_error") as mock_err:

        menu._run_historical_circuit_collector()
        mock_warn.assert_called_once()
        mock_err.assert_called_once()
        assert "No valid circuits selected" in mock_err.call_args[0][0]


def test_menu_multi_circuit_selection_executes(tmp_path):
    mock_tracks = {
        "austria": {"display_name": "Austrian Grand Prix", "historical_data_available": True, "historical_years": [2023]},
        "baku": {"display_name": "Azerbaijan Grand Prix", "historical_data_available": True, "historical_years": [2023]},
    }
    menu = InteractiveMenu(root_dir=tmp_path)
    menu.tracks_cfg = mock_tracks

    mock_collector = MagicMock(return_value={
        "track": "test",
        "circuit_name": "Test GP",
        "successes": ["2023 Race (20 drivers)"],
        "skips": [],
        "failures": [],
    })

    # Select both circuits "1 2", don't force re-download, press Enter at end
    with patch("src.cli.menu.prompt_str", side_effect=["1 2", ""]), \
         patch("src.cli.menu.prompt_confirm", return_value=False), \
         patch("src.data.collect_track.collect_historical_track_data", mock_collector):

        menu._run_historical_circuit_collector()
        assert mock_collector.call_count == 2
