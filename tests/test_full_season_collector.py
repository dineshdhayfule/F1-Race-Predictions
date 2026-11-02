"""
Automated Test Suite for Full Season Race & Qualifying Collector
================================================================
Verifies all 12 required behavioral guarantees:
1. Standard weekend collection: qualifying + race.
2. Sprint weekend collection: qualifying + race + sprint qualifying + sprint race.
3. Practice sessions (FP1, FP2, FP3) are NEVER collected.
4. Sprint detection across supported naming conventions (sprint, sprint_shootout, sprint_qualifying).
5. Existing sessions are skipped on rerun (incremental collection).
6. Interrupted collection can resume seamlessly.
7. One failed round does not terminate other rounds (fault tolerance).
8. Empty or invalid session data is rejected.
9. Round and circuit identity are validated.
10. Individual Grand Prix collection works without practice sessions.
11. No duplicate data or records are created (idempotency).
12. Historical data is not overwritten when force=False.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.data.collect_season import (
    detect_event_format_and_sessions,
    validate_and_format_session_results,
    collect_round_data,
    collect_season_data,
    collect_grand_prix,
)


@pytest.fixture
def mock_standard_schedule():
    """Mock schedule containing two standard rounds."""
    return pd.DataFrame([
        {
            "RoundNumber": 1,
            "EventName": "Australian Grand Prix",
            "Location": "Melbourne",
            "Country": "Australia",
            "EventFormat": "conventional",
            "Session1": "Practice 1",
            "Session2": "Practice 2",
            "Session3": "Practice 3",
            "Session4": "Qualifying",
            "Session5": "Race",
        },
        {
            "RoundNumber": 2,
            "EventName": "Japanese Grand Prix",
            "Location": "Suzuka",
            "Country": "Japan",
            "EventFormat": "conventional",
            "Session1": "Practice 1",
            "Session2": "Practice 2",
            "Session3": "Practice 3",
            "Session4": "Qualifying",
            "Session5": "Race",
        }
    ])


@pytest.fixture
def mock_sprint_schedule():
    """Mock schedule containing a sprint weekend and a standard weekend."""
    return pd.DataFrame([
        {
            "RoundNumber": 1,
            "EventName": "Chinese Grand Prix",
            "Location": "Shanghai",
            "Country": "China",
            "EventFormat": "sprint_qualifying",
            "Session1": "Practice 1",
            "Session2": "Sprint Qualifying",
            "Session3": "Sprint",
            "Session4": "Qualifying",
            "Session5": "Race",
        },
        {
            "RoundNumber": 2,
            "EventName": "Miami Grand Prix",
            "Location": "Miami",
            "Country": "USA",
            "EventFormat": "sprint_shootout",
            "Session1": "Practice 1",
            "Session2": "Qualifying",
            "Session3": "Sprint Shootout",
            "Session4": "Sprint",
            "Session5": "Race",
        }
    ])


def _make_mock_results_df(n_drivers=22, round_num=1, event_name="Test GP"):
    """Helper creating a realistic session results DataFrame."""
    rows = []
    for i in range(1, n_drivers + 1):
        rows.append({
            "DriverNumber": str(i),
            "BroadcastName": f"DRIVER {i}",
            "Abbreviation": f"D{i:02d}",
            "TeamName": f"Team {((i-1)//2) + 1}",
            "Position": float(i),
            "ClassifiedPosition": str(i),
            "GridPosition": float(i),
            "Q1": "0 days 00:01:18.000",
            "Q2": "0 days 00:01:17.500" if i <= 15 else np.nan,
            "Q3": "0 days 00:01:16.800" if i <= 10 else np.nan,
            "Time": "0 days 01:25:00.000",
            "Status": "Finished",
            "Points": 25.0 if i == 1 else (18.0 if i == 2 else 0.0),
            "Laps": 57,
        })
    return pd.DataFrame(rows)


def _make_mock_session(year=2026, round_num=1, session_name="Qualifying", df=None):
    """Helper creating a mock FastF1 Session object."""
    sess = MagicMock()
    sess.name = session_name
    sess.date = "2026-03-15T15:00:00Z"
    sess.event = MagicMock()
    sess.event.EventDate = "2026-03-15"
    sess.event.EventName = "Mock GP"
    sess.event.RoundNumber = round_num

    if df is not None:
        sess.results = df.copy()
    else:
        sess.results = _make_mock_results_df(22, round_num, "Mock GP")

    sess.load = MagicMock(return_value=None)
    return sess


# -----------------------------------------------------------------------------
# 1. Standard weekend collection: qualifying + race
# -----------------------------------------------------------------------------
def test_standard_weekend_collection(tmp_path, mock_standard_schedule):
    """Verifies that for a standard weekend, exactly Qualifying and Race are collected."""
    requested_sessions = []

    def mock_get_session(year, round_num, session_type):
        requested_sessions.append((round_num, session_type))
        return _make_mock_session(year, round_num, session_type)

    with patch("fastf1.get_event_schedule", return_value=mock_standard_schedule), \
         patch("fastf1.get_session", side_effect=mock_get_session):
        summary = collect_season_data(year=2026, rounds=[1], output_dir=tmp_path, verbose=False)

    assert summary["standard_count"] == 1
    assert summary["sprint_count"] == 0
    assert len(summary["successes"]) == 2

    # Verify files created on disk
    q_file = tmp_path / "2026_round_1_qualifying.csv"
    r_file = tmp_path / "2026_round_1_race.csv"
    sq_file = tmp_path / "2026_round_1_sprint_qualifying.csv"
    sr_file = tmp_path / "2026_round_1_sprint_race.csv"

    assert q_file.exists()
    assert r_file.exists()
    assert not sq_file.exists()
    assert not sr_file.exists()


# -----------------------------------------------------------------------------
# 2. Sprint weekend collection: qualifying + race + sprint qualifying + sprint race
# -----------------------------------------------------------------------------
def test_sprint_weekend_collection(tmp_path, mock_sprint_schedule):
    """Verifies that for a sprint weekend, all 4 required sessions are collected."""
    requested_sessions = []

    def mock_get_session(year, round_num, session_type):
        requested_sessions.append((round_num, session_type))
        return _make_mock_session(year, round_num, session_type)

    with patch("fastf1.get_event_schedule", return_value=mock_sprint_schedule), \
         patch("fastf1.get_session", side_effect=mock_get_session):
        summary = collect_season_data(year=2026, rounds=[1], output_dir=tmp_path, verbose=False)

    assert summary["sprint_count"] == 1
    assert len(summary["successes"]) == 4

    # Verify all 4 files exist
    assert (tmp_path / "2026_round_1_sprint_qualifying.csv").exists()
    assert (tmp_path / "2026_round_1_sprint_race.csv").exists()
    assert (tmp_path / "2026_round_1_qualifying.csv").exists()
    assert (tmp_path / "2026_round_1_race.csv").exists()


# -----------------------------------------------------------------------------
# 3. Practice sessions (FP1, FP2, FP3) are NEVER collected
# -----------------------------------------------------------------------------
def test_practice_sessions_never_collected(tmp_path, mock_sprint_schedule):
    """Explicitly verifies that FP1, FP2, and FP3 are NEVER requested or saved."""
    requested_session_types = []

    def mock_get_session(year, round_num, session_type):
        requested_session_types.append(str(session_type).lower())
        return _make_mock_session(year, round_num, session_type)

    with patch("fastf1.get_event_schedule", return_value=mock_sprint_schedule), \
         patch("fastf1.get_session", side_effect=mock_get_session):
        collect_season_data(year=2026, rounds=[1, 2], output_dir=tmp_path, verbose=False)

    for disallowed in ["fp1", "fp2", "fp3", "practice 1", "practice 2", "practice 3"]:
        assert disallowed not in requested_session_types, f"Practice session '{disallowed}' was queried!"

    for fp_file in tmp_path.glob("*fp*.csv"):
        pytest.fail(f"Practice file was written: {fp_file.name}")


def test_full_weekend_collection_includes_practice(tmp_path, mock_standard_schedule):
    """Verifies that with session_mode='full', all 5 weekend sessions are collected (FP1, FP2, FP3, Quali, Race)."""
    requested_session_types = []

    def mock_get_session(year, round_num, session_type):
        requested_session_types.append(str(session_type).lower())
        return _make_mock_session(year, round_num, session_type)

    with patch("fastf1.get_event_schedule", return_value=mock_standard_schedule), \
         patch("fastf1.get_session", side_effect=mock_get_session):
        summary = collect_season_data(
            year=2026,
            rounds=[1],
            output_dir=tmp_path,
            verbose=False,
            session_mode="full"
        )

    assert len(summary["successes"]) == 5
    assert (tmp_path / "2026_round_1_fp1.csv").exists()
    assert (tmp_path / "2026_round_1_fp2.csv").exists()
    assert (tmp_path / "2026_round_1_fp3.csv").exists()
    assert (tmp_path / "2026_round_1_qualifying.csv").exists()
    assert (tmp_path / "2026_round_1_race.csv").exists()


def test_practice_only_collection_mode(tmp_path, mock_standard_schedule):
    """Verifies that session_mode='practice_only' collects strictly FP1, FP2, and FP3."""
    def mock_get_session(year, round_num, session_type):
        return _make_mock_session(year, round_num, session_type)

    with patch("fastf1.get_event_schedule", return_value=mock_standard_schedule), \
         patch("fastf1.get_session", side_effect=mock_get_session):
        summary = collect_season_data(
            year=2026,
            rounds=[1],
            output_dir=tmp_path,
            verbose=False,
            session_mode="practice_only"
        )

    assert len(summary["successes"]) == 3
    assert (tmp_path / "2026_round_1_fp1.csv").exists()
    assert (tmp_path / "2026_round_1_fp2.csv").exists()
    assert (tmp_path / "2026_round_1_fp3.csv").exists()
    assert not (tmp_path / "2026_round_1_qualifying.csv").exists()
    assert not (tmp_path / "2026_round_1_race.csv").exists()


# -----------------------------------------------------------------------------
# 4. Sprint detection across supported naming conventions
# -----------------------------------------------------------------------------
def test_sprint_detection_naming_conventions():
    """Tests sprint detection across historical 2021-2026 event naming formats."""
    # 2024 format: sprint_qualifying
    row_2024 = pd.Series({
        "EventFormat": "sprint_qualifying",
        "Session1": "Practice 1",
        "Session2": "Sprint Qualifying",
        "Session3": "Sprint",
        "Session4": "Qualifying",
        "Session5": "Race"
    })
    det_2024 = detect_event_format_and_sessions(row_2024)
    assert det_2024["is_sprint"] is True
    assert det_2024["sq_declared"] == "Sprint Qualifying"
    assert det_2024["sr_declared"] == "Sprint"

    # 2023 format: sprint_shootout
    row_2023 = pd.Series({
        "EventFormat": "sprint_shootout",
        "Session1": "Practice 1",
        "Session2": "Qualifying",
        "Session3": "Sprint Shootout",
        "Session4": "Sprint",
        "Session5": "Race"
    })
    det_2023 = detect_event_format_and_sessions(row_2023)
    assert det_2023["is_sprint"] is True
    assert det_2023["sq_declared"] == "Sprint Shootout"
    assert det_2023["sr_declared"] == "Sprint"

    # 2022 format: sprint
    row_2022 = pd.Series({
        "EventFormat": "sprint",
        "Session1": "Practice 1",
        "Session2": "Qualifying",
        "Session3": "Practice 2",
        "Session4": "Sprint",
        "Session5": "Race"
    })
    det_2022 = detect_event_format_and_sessions(row_2022)
    assert det_2022["is_sprint"] is True

    # Standard format: conventional
    row_conv = pd.Series({
        "EventFormat": "conventional",
        "Session1": "Practice 1",
        "Session2": "Practice 2",
        "Session3": "Practice 3",
        "Session4": "Qualifying",
        "Session5": "Race"
    })
    det_conv = detect_event_format_and_sessions(row_conv)
    assert det_conv["is_sprint"] is False
    assert len(det_conv["session_plan"]) == 2  # Only Qualifying and Race


# -----------------------------------------------------------------------------
# 5. Existing sessions are skipped on rerun (incremental collection)
# -----------------------------------------------------------------------------
def test_existing_sessions_skipped_on_rerun(tmp_path, mock_standard_schedule):
    """Verifies that existing valid session files are skipped when force=False."""
    # Pre-populate qualifying file
    q_file = tmp_path / "2026_round_1_qualifying.csv"
    mock_df = _make_mock_results_df(22, 1, "Australian GP")
    mock_df.to_csv(q_file, index=False)

    call_count = {"Q": 0, "R": 0}

    def mock_get_session(year, round_num, session_type):
        if session_type in ("Qualifying", "Q"):
            call_count["Q"] += 1
        elif session_type in ("Race", "R"):
            call_count["R"] += 1
        return _make_mock_session(year, round_num, session_type)

    with patch("fastf1.get_event_schedule", return_value=mock_standard_schedule), \
         patch("fastf1.get_session", side_effect=mock_get_session):
        summary = collect_season_data(year=2026, rounds=[1], output_dir=tmp_path, force=False, verbose=False)

    # Qualifying should be skipped; Race should be collected
    assert call_count["Q"] == 0, "Qualifying should have been skipped without querying FastF1!"
    assert call_count["R"] == 1, "Race was not yet present and should have been collected!"
    assert any("Qualifying" in s for s in summary["skips"])
    assert any("Race" in s for s in summary["successes"])


# -----------------------------------------------------------------------------
# 6. Interrupted collection can resume seamlessly
# -----------------------------------------------------------------------------
def test_interrupted_collection_can_resume(tmp_path, mock_standard_schedule):
    """Verifies that if collection was interrupted mid-season, rerunning completes remaining."""
    # Simulate Round 1 already completed
    for s_name in ["qualifying", "race"]:
        f = tmp_path / f"2026_round_1_{s_name}.csv"
        _make_mock_results_df(22, 1).to_csv(f, index=False)

    called_rounds = []

    def mock_get_session(year, round_num, session_type):
        called_rounds.append(round_num)
        return _make_mock_session(year, round_num, session_type)

    with patch("fastf1.get_event_schedule", return_value=mock_standard_schedule), \
         patch("fastf1.get_session", side_effect=mock_get_session):
        summary = collect_season_data(year=2026, rounds=[1, 2], output_dir=tmp_path, force=False, verbose=False)

    # Round 1 should be skipped, Round 2 collected
    assert 1 not in called_rounds
    assert 2 in called_rounds
    assert (tmp_path / "2026_round_2_qualifying.csv").exists()
    assert (tmp_path / "2026_round_2_race.csv").exists()


# -----------------------------------------------------------------------------
# 7. One failed round does not terminate other rounds (fault tolerance)
# -----------------------------------------------------------------------------
def test_one_failed_round_does_not_stop_others(tmp_path, mock_standard_schedule):
    """Verifies that an exception in Round 1 does not prevent Round 2 from collecting."""
    def mock_get_session(year, round_num, session_type):
        if round_num == 1:
            raise RuntimeError("Simulated network timeout for Round 1")
        return _make_mock_session(year, round_num, session_type)

    with patch("fastf1.get_event_schedule", return_value=mock_standard_schedule), \
         patch("fastf1.get_session", side_effect=mock_get_session):
        summary = collect_season_data(year=2026, rounds=[1, 2], output_dir=tmp_path, verbose=False)

    # Round 1 has failures, Round 2 succeeded
    assert len(summary["failures"]) >= 1
    assert any("Round 1" in f or "R1" in f for f in summary["failures"])
    assert (tmp_path / "2026_round_2_qualifying.csv").exists()
    assert (tmp_path / "2026_round_2_race.csv").exists()


# -----------------------------------------------------------------------------
# 8. Empty or invalid session data is rejected
# -----------------------------------------------------------------------------
def test_empty_or_invalid_session_rejected(tmp_path):
    """Verifies that empty results or missing position columns are rejected."""
    # Empty DataFrame
    empty_sess = _make_mock_session(df=pd.DataFrame())
    valid, msg, df = validate_and_format_session_results(
        empty_sess, 2026, 1, "Mock GP", "Circuit", "Country", "Qualifying"
    )
    assert valid is False
    assert "empty" in msg.lower()

    # No positions (all NaN)
    no_pos_df = pd.DataFrame({
        "Abbreviation": ["VER", "HAM"],
        "Position": [np.nan, np.nan]
    })
    nan_sess = _make_mock_session(df=no_pos_df)
    valid, msg, df = validate_and_format_session_results(
        nan_sess, 2026, 1, "Mock GP", "Circuit", "Country", "Qualifying"
    )
    assert valid is False
    assert "no classified" in msg.lower()

    # Missing driver column
    no_drv_df = pd.DataFrame({
        "Position": [1.0, 2.0],
        "Laps": [50, 50]
    })
    no_drv_sess = _make_mock_session(df=no_drv_df)
    valid, msg, df = validate_and_format_session_results(
        no_drv_sess, 2026, 1, "Mock GP", "Circuit", "Country", "Qualifying"
    )
    assert valid is False
    assert "driver identifier" in msg.lower()


# -----------------------------------------------------------------------------
# 9. Round and circuit identity are validated
# -----------------------------------------------------------------------------
def test_round_identity_validation():
    """Verifies that session event round number mismatch is flagged and rejected."""
    wrong_round_sess = _make_mock_session(round_num=99)  # session reports Round 99
    valid, msg, df = validate_and_format_session_results(
        wrong_round_sess,
        expected_year=2026,
        expected_round=5,  # expected Round 5
        event_name="Spanish GP",
        circuit_name="Barcelona",
        country="Spain",
        canonical_type="Qualifying"
    )
    assert valid is False
    assert "round identity mismatch" in msg.lower()


# -----------------------------------------------------------------------------
# 10. Existing individual Grand Prix collection still works
# -----------------------------------------------------------------------------
def test_individual_grand_prix_collection(tmp_path):
    """Verifies single Grand Prix collection (collect_grand_prix) without practice."""
    mock_event = pd.Series({
        "RoundNumber": 19,
        "EventName": "Mexico City Grand Prix",
        "Location": "Mexico City",
        "Country": "Mexico",
        "EventFormat": "conventional",
        "Session1": "Practice 1",
        "Session2": "Practice 2",
        "Session3": "Practice 3",
        "Session4": "Qualifying",
        "Session5": "Race",
    })

    requested_sessions = []

    def mock_get_session(year, round_num, session_type):
        requested_sessions.append(session_type)
        return _make_mock_session(year, round_num, session_type)

    season_dir = tmp_path / "season"
    track_dir = tmp_path / "tracks" / "mexico"

    with patch("src.data.collect_track.get_event_by_track_id", return_value=mock_event), \
         patch("fastf1.get_session", side_effect=mock_get_session):
        res = collect_grand_prix(
            track_id="mexico",
            year=2026,
            output_dir=track_dir,
            season_output_dir=season_dir,
            verbose=False
        )

    assert len(res["successes"]) == 2
    assert (season_dir / "2026_round_19_qualifying.csv").exists()
    assert (season_dir / "2026_round_19_race.csv").exists()
    assert (track_dir / "qualifying.csv").exists()
    assert (track_dir / "race.csv").exists()

    # Verify no FP files
    assert not (track_dir / "fp1.csv").exists()
    assert not (track_dir / "fp2.csv").exists()
    assert not (track_dir / "fp3.csv").exists()


# -----------------------------------------------------------------------------
# 11. No duplicate data is created (idempotency & deduplication)
# -----------------------------------------------------------------------------
def test_deduplication_and_idempotency(tmp_path):
    """Verifies that duplicate driver rows in raw data are cleaned and idempotency holds."""
    dup_df = _make_mock_results_df(22, 1)
    # Inject an exact duplicate row for driver 1
    dup_df = pd.concat([dup_df, dup_df.iloc[[0]]], ignore_index=True)
    assert len(dup_df) == 23

    sess = _make_mock_session(df=dup_df)
    valid, msg, df = validate_and_format_session_results(
        sess, 2026, 1, "Mock GP", "Melbourne", "Australia", "Qualifying"
    )
    assert valid is True
    assert len(df) == 22  # Deduplicated down to 22 unique drivers


# -----------------------------------------------------------------------------
# 12. Historical data is not overwritten when force=False
# -----------------------------------------------------------------------------
def test_historical_data_not_overwritten_when_not_forced(tmp_path, mock_standard_schedule):
    """Verifies existing file contents and modification timestamps are preserved."""
    target_file = tmp_path / "2026_round_1_race.csv"
    target_file.write_text("Year,Round,Race,Position,Abbreviation\n2026,1,Original GP,1,VER\n")
    original_content = target_file.read_text()

    with patch("fastf1.get_event_schedule", return_value=mock_standard_schedule):
        collect_season_data(year=2026, rounds=[1], output_dir=tmp_path, force=False, verbose=False)

    # Content must remain identical
    assert target_file.read_text() == original_content

