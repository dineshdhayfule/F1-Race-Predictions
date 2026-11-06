"""
Comprehensive Unit Test Suite for CalendarService and Multi-Year Calendar Selection
=====================================================================================
Validates:
1. Valid season calendar retrieval (positive championship rounds only).
2. Retrieval of multiple different seasons and memory caching.
3. Correct event and round mapping with normalized metadata.
4. Different round numbers for the same circuit across seasons.
5. Sprint vs. non-sprint format detection.
6. Empty calendar handling.
7. Failed schedule retrieval handling (raises CalendarRetrievalError).
8. Invalid year input handling (raises ValueError).
9. Circuit present in one season but absent in another.
10. Existing 2026 compatibility.
11. Strict prohibition of silent 2026 fallback for other years.
12. Prediction pipeline integration with explicit round_num and backward compatibility.
13. Interactive menu calendar-driven selection workflow with mocked FastF1.
"""

from unittest.mock import patch, MagicMock
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.data.calendar_service import (
    CalendarService,
    CalendarEvent,
    CalendarRetrievalError,
    MIN_CHAMPIONSHIP_YEAR,
    MAX_CHAMPIONSHIP_YEAR,
)
from src.pipeline.race_prediction_pipeline import RacePredictionPipeline
from src.cli.menu import InteractiveMenu

ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# FIXTURES & MOCK FACTORIES
# =============================================================================

def make_mock_schedule_2024() -> pd.DataFrame:
    """Creates a realistic mocked 2024 FastF1 event schedule."""
    data = [
        # Round 0: Pre-season testing (must be filtered out)
        {
            "RoundNumber": 0,
            "Country": "Bahrain",
            "Location": "Sakhir",
            "OfficialEventName": "FORMULA 1 ARAMCO PRE-SEASON TESTING 2024",
            "EventDate": "2024-02-23",
            "EventName": "Pre-Season Testing",
            "EventFormat": "testing",
            "Session1": "Practice 1", "Session1Date": "2024-02-21",
            "Session2": "Practice 2", "Session2Date": "2024-02-22",
            "Session3": "Practice 3", "Session3Date": "2024-02-23",
            "Session4": None, "Session5": None,
        },
        # Round 1: Bahrain (Conventional)
        {
            "RoundNumber": 1,
            "Country": "Bahrain",
            "Location": "Sakhir",
            "OfficialEventName": "FORMULA 1 GULF AIR BAHRAIN GRAND PRIX 2024",
            "EventDate": "2024-03-02",
            "EventName": "Bahrain Grand Prix",
            "EventFormat": "conventional",
            "Session1": "Practice 1", "Session1Date": "2024-02-29",
            "Session2": "Practice 2", "Session2Date": "2024-02-29",
            "Session3": "Practice 3", "Session3Date": "2024-03-01",
            "Session4": "Qualifying", "Session4Date": "2024-03-01",
            "Session5": "Race", "Session5Date": "2024-03-02",
        },
        # Round 2: Saudi Arabia (Conventional)
        {
            "RoundNumber": 2,
            "Country": "Saudi Arabia",
            "Location": "Jeddah",
            "OfficialEventName": "FORMULA 1 STC SAUDI ARABIAN GRAND PRIX 2024",
            "EventDate": "2024-03-09",
            "EventName": "Saudi Arabian Grand Prix",
            "EventFormat": "conventional",
            "Session1": "Practice 1", "Session1Date": "2024-03-07",
            "Session2": "Practice 2", "Session2Date": "2024-03-07",
            "Session3": "Practice 3", "Session3Date": "2024-03-08",
            "Session4": "Qualifying", "Session4Date": "2024-03-08",
            "Session5": "Race", "Session5Date": "2024-03-09",
        },
        # Round 3: Australia (Conventional)
        {
            "RoundNumber": 3,
            "Country": "Australia",
            "Location": "Melbourne",
            "OfficialEventName": "FORMULA 1 ROLEX AUSTRALIAN GRAND PRIX 2024",
            "EventDate": "2024-03-24",
            "EventName": "Australian Grand Prix",
            "EventFormat": "conventional",
            "Session1": "Practice 1", "Session1Date": "2024-03-22",
            "Session2": "Practice 2", "Session2Date": "2024-03-22",
            "Session3": "Practice 3", "Session3Date": "2024-03-23",
            "Session4": "Qualifying", "Session4Date": "2024-03-23",
            "Session5": "Race", "Session5Date": "2024-03-24",
        },
        # Round 4: Japan (Suzuka) - In 2024 Suzuka was Round 4!
        {
            "RoundNumber": 4,
            "Country": "Japan",
            "Location": "Suzuka",
            "OfficialEventName": "FORMULA 1 MSC CRUISES JAPANESE GRAND PRIX 2024",
            "EventDate": "2024-04-07",
            "EventName": "Japanese Grand Prix",
            "EventFormat": "conventional",
            "Session1": "Practice 1", "Session1Date": "2024-04-05",
            "Session2": "Practice 2", "Session2Date": "2024-04-05",
            "Session3": "Practice 3", "Session3Date": "2024-04-06",
            "Session4": "Qualifying", "Session4Date": "2024-04-06",
            "Session5": "Race", "Session5Date": "2024-04-07",
        },
        # Round 5: China (Sprint Weekend)
        {
            "RoundNumber": 5,
            "Country": "China",
            "Location": "Shanghai",
            "OfficialEventName": "FORMULA 1 LENOVO CHINESE GRAND PRIX 2024",
            "EventDate": "2024-04-21",
            "EventName": "Chinese Grand Prix",
            "EventFormat": "sprint",
            "Session1": "Practice 1", "Session1Date": "2024-04-19",
            "Session2": "Sprint Qualifying", "Session2Date": "2024-04-19",
            "Session3": "Sprint", "Session3Date": "2024-04-20",
            "Session4": "Qualifying", "Session4Date": "2024-04-20",
            "Session5": "Race", "Session5Date": "2024-04-21",
        },
    ]
    return pd.DataFrame(data)


def make_mock_schedule_2026() -> pd.DataFrame:
    """Creates a realistic mocked 2026 FastF1 event schedule."""
    data = [
        # Round 1: Australia (In 2026 Australia is Season Opener Round 1)
        {
            "RoundNumber": 1,
            "Country": "Australia",
            "Location": "Melbourne",
            "OfficialEventName": "FORMULA 1 AUSTRALIAN GRAND PRIX 2026",
            "EventDate": "2026-03-15",
            "EventName": "Australian Grand Prix",
            "EventFormat": "conventional",
            "Session1": "Practice 1", "Session2": "Practice 2", "Session3": "Practice 3",
            "Session4": "Qualifying", "Session5": "Race",
        },
        # Round 2: China (Sprint)
        {
            "RoundNumber": 2,
            "Country": "China",
            "Location": "Shanghai",
            "OfficialEventName": "FORMULA 1 CHINESE GRAND PRIX 2026",
            "EventDate": "2026-03-29",
            "EventName": "Chinese Grand Prix",
            "EventFormat": "sprint",
            "Session1": "Practice 1", "Session2": "Sprint Qualifying", "Session3": "Sprint",
            "Session4": "Qualifying", "Session5": "Race",
        },
        # Round 3: Japan (Suzuka) - In 2026 Suzuka is Round 3!
        {
            "RoundNumber": 3,
            "Country": "Japan",
            "Location": "Suzuka",
            "OfficialEventName": "FORMULA 1 JAPANESE GRAND PRIX 2026",
            "EventDate": "2026-04-12",
            "EventName": "Japanese Grand Prix",
            "EventFormat": "conventional",
            "Session1": "Practice 1", "Session2": "Practice 2", "Session3": "Practice 3",
            "Session4": "Qualifying", "Session5": "Race",
        },
        # Round 19: Mexico
        {
            "RoundNumber": 19,
            "Country": "Mexico",
            "Location": "Mexico City",
            "OfficialEventName": "FORMULA 1 GRAN PREMIO DE LA CIUDAD DE MÉXICO 2026",
            "EventDate": "2026-11-01",
            "EventName": "Mexico City Grand Prix",
            "EventFormat": "conventional",
            "Session1": "Practice 1", "Session2": "Practice 2", "Session3": "Practice 3",
            "Session4": "Qualifying", "Session5": "Race",
        },
    ]
    return pd.DataFrame(data)


# =============================================================================
# 1. VALID SEASON CALENDAR RETRIEVAL & FILTERING
# =============================================================================

def test_valid_season_calendar_retrieval():
    """Test 1: Valid season schedule retrieves only positive rounds, excluding round 0 testing."""
    service = CalendarService(root_dir=ROOT, enable_cache=False)
    mock_df = make_mock_schedule_2024()

    with patch("fastf1.get_event_schedule", return_value=mock_df):
        events = service.get_calendar(2024)

    # Round 0 (Pre-season testing) MUST be excluded
    assert len(events) == 5
    assert all(ev.round_number > 0 for ev in events)
    assert [ev.round_number for ev in events] == [1, 2, 3, 4, 5]
    assert events[0].event_name == "Bahrain Grand Prix"
    assert events[0].round_number == 1


# =============================================================================
# 2. MULTIPLE DIFFERENT SEASONS & CACHING
# =============================================================================

def test_multiple_seasons_retrieval_and_caching():
    """Test 2: Retrieving multiple seasons caches in memory and avoids duplicate API calls."""
    service = CalendarService(root_dir=ROOT, enable_cache=False)
    sched_2024 = make_mock_schedule_2024()
    sched_2026 = make_mock_schedule_2026()

    with patch("fastf1.get_event_schedule") as mock_get_sched:
        mock_get_sched.side_effect = lambda yr: sched_2024 if yr == 2024 else sched_2026

        events_2024 = service.get_calendar(2024)
        assert len(events_2024) == 5
        assert mock_get_sched.call_count == 1

        # Second call for 2024 should hit memory cache without calling fastf1 again
        cached_2024 = service.get_calendar(2024)
        assert len(cached_2024) == 5
        assert mock_get_sched.call_count == 1  # Still 1!

        # Call for 2026 should retrieve 2026
        events_2026 = service.get_calendar(2026)
        assert len(events_2026) == 4
        assert mock_get_sched.call_count == 2

        # Force refresh bypasses cache
        refreshed_2024 = service.get_calendar(2024, force_refresh=True)
        assert len(refreshed_2024) == 5
        assert mock_get_sched.call_count == 3


# =============================================================================
# 3. CORRECT EVENT AND ROUND MAPPING
# =============================================================================

def test_correct_event_and_round_mapping():
    """Test 3: Normalizes all fields into a structured CalendarEvent."""
    service = CalendarService(root_dir=ROOT, enable_cache=False)
    mock_df = make_mock_schedule_2024()

    with patch("fastf1.get_event_schedule", return_value=mock_df):
        events = service.get_calendar(2024)

    aus = events[2]
    assert aus.round_number == 3
    assert aus.year == 2024
    assert aus.event_name == "Australian Grand Prix"
    assert "AUSTRALIAN GRAND PRIX" in aus.official_event_name
    assert aus.country == "Australia"
    assert aus.location == "Melbourne"
    assert aus.circuit_key == "australia"
    assert aus.event_date == "2024-03-24"
    assert aus.is_sprint is False
    assert "Practice 1" in aus.sessions
    assert "Qualifying" in aus.sessions
    assert "Race" in aus.sessions

    # Test to_dict() method
    d = aus.to_dict()
    assert isinstance(d, dict)
    assert d["round_number"] == 3
    assert d["circuit_key"] == "australia"


# =============================================================================
# 4. DIFFERENT ROUND NUMBERS ACROSS SEASONS
# =============================================================================

def test_different_round_numbers_across_seasons():
    """Test 4: Same circuit accurately maps to different round numbers across seasons."""
    service = CalendarService(root_dir=ROOT, enable_cache=False)
    sched_2024 = make_mock_schedule_2024()
    sched_2026 = make_mock_schedule_2026()

    with patch("fastf1.get_event_schedule") as mock_get_sched:
        mock_get_sched.side_effect = lambda yr: sched_2024 if yr == 2024 else sched_2026

        # In 2024, Australia was Round 3; Japan was Round 4
        ev_aus_2024 = service.get_event_by_circuit(2024, "australia")
        ev_jpn_2024 = service.get_event_by_circuit(2024, "japan")
        assert ev_aus_2024.round_number == 3
        assert ev_jpn_2024.round_number == 4

        # In 2026, Australia was Round 1; Japan was Round 3
        ev_aus_2026 = service.get_event_by_circuit(2026, "australia")
        ev_jpn_2026 = service.get_event_by_circuit(2026, "japan")
        assert ev_aus_2026.round_number == 1
        assert ev_jpn_2026.round_number == 3


# =============================================================================
# 5. SPRINT AND NON-SPRINT WEEKENDS
# =============================================================================

def test_sprint_and_non_sprint_weekends():
    """Test 5: Accurately identifies Sprint format vs Standard format from schedule."""
    service = CalendarService(root_dir=ROOT, enable_cache=False)
    mock_df = make_mock_schedule_2024()

    with patch("fastf1.get_event_schedule", return_value=mock_df):
        events = service.get_calendar(2024)

    # Bahrain is standard
    bah = events[0]
    assert bah.is_sprint is False
    assert bah.event_format == "conventional"

    # China is sprint
    chn = events[4]
    assert chn.is_sprint is True
    assert chn.event_format == "sprint"
    assert "Sprint Qualifying" in chn.sessions
    assert "Sprint" in chn.sessions


# =============================================================================
# 6. EMPTY CALENDAR HANDLING
# =============================================================================

def test_empty_calendar_handling():
    """Test 6: Handles empty DataFrame or schedule without championship rounds cleanly."""
    service = CalendarService(root_dir=ROOT, enable_cache=False)

    with patch("fastf1.get_event_schedule", return_value=pd.DataFrame()):
        events = service.get_calendar(2024)
        assert events == []

    # Schedule with only testing round (RoundNumber == 0)
    testing_only = pd.DataFrame([{"RoundNumber": 0, "EventName": "Testing"}])
    with patch("fastf1.get_event_schedule", return_value=testing_only):
        events = service.get_calendar(2024, force_refresh=True)
        assert events == []


# =============================================================================
# 7. FAILED SCHEDULE RETRIEVAL
# =============================================================================

def test_failed_schedule_retrieval_raises_calendar_retrieval_error():
    """Test 7: Raises CalendarRetrievalError when FastF1 network/API call fails."""
    service = CalendarService(root_dir=ROOT, enable_cache=False)

    with patch("fastf1.get_event_schedule", side_effect=Exception("Network connection failed")):
        with pytest.raises(CalendarRetrievalError, match="Failed to retrieve FastF1 schedule"):
            service.get_calendar(2024)


# =============================================================================
# 8. INVALID YEAR INPUT
# =============================================================================

def test_invalid_year_input_raises_value_error():
    """Test 8: Rejects invalid year values (out of range, non-numeric)."""
    service = CalendarService(root_dir=ROOT, enable_cache=False)

    with pytest.raises(ValueError, match="Invalid championship year"):
        service.get_calendar("abc")

    with pytest.raises(ValueError, match="out of valid championship range"):
        service.get_calendar(1899)

    with pytest.raises(ValueError, match="out of valid championship range"):
        service.get_calendar(2099)


# =============================================================================
# 9. CIRCUIT PRESENT IN ONE SEASON BUT ABSENT IN ANOTHER
# =============================================================================

def test_circuit_present_in_one_season_absent_in_another():
    """Test 9: Successfully finds a circuit in season A but returns None in season B where absent."""
    service = CalendarService(root_dir=ROOT, enable_cache=False)
    sched_2024 = make_mock_schedule_2024()  # Contains Saudi Arabia
    sched_2026 = make_mock_schedule_2026()  # Does NOT contain Saudi Arabia

    with patch("fastf1.get_event_schedule") as mock_get_sched:
        mock_get_sched.side_effect = lambda yr: sched_2024 if yr == 2024 else sched_2026

        # Present in 2024
        ev_jed_2024 = service.get_event_by_circuit(2024, "jeddah")
        assert ev_jed_2024 is not None
        assert ev_jed_2024.round_number == 2

        # Absent in 2026
        ev_jed_2026 = service.get_event_by_circuit(2026, "jeddah")
        assert ev_jed_2026 is None


# =============================================================================
# 10. EXISTING 2026 COMPATIBILITY
# =============================================================================

def test_existing_2026_compatibility():
    """Test 10: 2026 schedule retrieval correctly returns 2026 rounds matching tracks.json."""
    service = CalendarService(root_dir=ROOT, enable_cache=False)
    sched_2026 = make_mock_schedule_2026()

    with patch("fastf1.get_event_schedule", return_value=sched_2026):
        events = service.get_calendar(2026)

    assert len(events) == 4
    mex = next(e for e in events if e.circuit_key == "mexico")
    assert mex.round_number == 19
    assert mex.is_sprint is False


# =============================================================================
# 11. NO 2026 FALLBACK FOR ANOTHER YEAR
# =============================================================================

def test_no_2026_fallback_for_another_year():
    """Test 11: When calendar for 2024 is requested, 2026 calendar is never substituted."""
    service = CalendarService(root_dir=ROOT, enable_cache=False)
    sched_2024 = make_mock_schedule_2024()
    sched_2026 = make_mock_schedule_2026()

    with patch("fastf1.get_event_schedule") as mock_get_sched:
        mock_get_sched.side_effect = lambda yr: sched_2024 if yr == 2024 else sched_2026

        events_2024 = service.get_calendar(2024)
        rounds_2024 = [e.round_number for e in events_2024]
        # 2024 rounds must be 1, 2, 3, 4, 5 - NOT 2026 rounds [1, 2, 3, 19]
        assert rounds_2024 == [1, 2, 3, 4, 5]
        assert not any(e.round_number == 19 for e in events_2024)

    # When 2024 raises an error, it must NOT silently return 2026
    with patch("fastf1.get_event_schedule", side_effect=Exception("Failed")):
        with pytest.raises(CalendarRetrievalError):
            service.get_calendar(2024, force_refresh=True)


# =============================================================================
# 12. PREDICTION PIPELINE INTEGRATION & BACKWARD COMPATIBILITY
# =============================================================================

def test_pipeline_integration_with_explicit_round():
    """Test 12A: Pipeline accepts explicit round_num without requiring tracks.json entry."""
    pipeline = RacePredictionPipeline(root_dir=ROOT)

    # If round 1 is passed for Australia 2024, it must recognize Round 1 (and raise early season DEF-01 error)
    with pytest.raises(ValueError) as excinfo:
        pipeline.run(track_key="australia", year=2024, round_num=1, stage="pre_fp1")
    assert "Round 1" in str(excinfo.value)
    assert "No prior same-season historical rounds exist" in str(excinfo.value)


def test_pipeline_backward_compatibility_without_round_num():
    """Test 12B: Calling pipeline.run without round_num preserves existing 2026 behavior."""
    pipeline = RacePredictionPipeline(root_dir=ROOT)

    # In 2026, Australia is round 1 in tracks.json
    with pytest.raises(ValueError) as excinfo:
        pipeline.run(track_key="australia", year=2026, stage="pre_fp1")
    assert "Round 1" in str(excinfo.value)

    # For non-2026 year without round_num and without round in tracks.json, raises descriptive error
    with pytest.raises(ValueError) as excinfo:
        pipeline.run(track_key="australia", year=2024, stage="pre_fp1")
    assert "Calendar round for track 'australia' is not configured for year 2024" in str(excinfo.value)


# =============================================================================
# 13. INTERACTIVE MENU CALENDAR-DRIVEN SELECTION WORKFLOW
# =============================================================================

def test_interactive_menu_calendar_driven_workflow(capsys):
    """
    Test 13: Interactive menu prompts for year, retrieves calendar,
    displays available events, and passes selected round_num to the pipeline.
    """
    menu = InteractiveMenu(root_dir=ROOT)
    sched_2024 = make_mock_schedule_2024()

    with patch("fastf1.get_event_schedule", return_value=sched_2024):
        # User input sequence:
        # 1. "2024" (Season Year)
        # 2. "3"    (Select Australian Grand Prix - Round 3)
        # 3. "1"    (Select stage 'pre_fp1')
        # 4. "0"    (Cancel at model selection / back)
        inputs = iter(["2024", "3", "1", "0"])
        with patch("builtins.input", lambda *args, **kwargs: next(inputs)):
            menu.menu_predict_race()

    captured = capsys.readouterr().out
    assert "Discovering 2024 F1 calendar..." in captured
    assert "Available Grand Prix events:" in captured
    assert "Bahrain Grand Prix" in captured
    assert "Australian Grand Prix" in captured
    assert "Round 3" in captured


def test_interactive_menu_calendar_failure_handling(capsys):
    """Test 14: Interactive menu handles FastF1 calendar retrieval failure gracefully without crashing."""
    menu = InteractiveMenu(root_dir=ROOT)

    with patch("fastf1.get_event_schedule", side_effect=Exception("API connection refused")):
        # User input sequence:
        # 1. "2024" (Season Year)
        # 2. ""     (Press Enter to return)
        inputs = iter(["2024", ""])
        with patch("builtins.input", lambda *args, **kwargs: next(inputs)):
            menu.menu_predict_race()

    captured = capsys.readouterr().out
    assert "Failed to retrieve F1 calendar for season 2024" in captured
    # Must NOT report success or pretend 2026 data is 2024
    assert "Available Grand Prix events:" not in captured
