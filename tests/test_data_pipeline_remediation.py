"""
Phase 3 Regression Test Suite: Controlled Data Pipeline Remediation
===================================================================
Validates:
1. Valid FastF1 qualifying identifiers (strictly 'Qualifying', 'Q', 'Sprint Qualifying', 'Sprint Shootout', 'SQ').
2. Rejection / sanitization of unsupported 'Quali' / 'Sprint Quali' identifiers and graceful ValueError handling.
3. Modern qualifying session collection loads with messages=True.
4. Missing race-control messages handling without crashing or fabricating results.
5. Missing official qualifying results rejection (no silent fabrication from raw lap times).
6. Strict 'after_qualifying' pipeline rejection when official qualifying results are missing.
7. Permitted provisional prediction behavior under appropriate pre-qualifying stages.
8. Future race session classification (FUTURE_SESSION without false failure logging).
9. Empty race-session response handling (DATA_MISSING).
10. OpenF1 timeout and connection error handling.
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
import urllib.error
import pytest
import pandas as pd
import numpy as np

from src.data.collect_season import (
    detect_event_format_and_sessions,
    validate_and_format_session_results,
    collect_round_data,
    resolve_session_scheduled_time,
)
from src.data.grid_retriever import StartingGridRetriever
from src.pipeline.race_prediction_pipeline import RacePredictionPipeline


# =============================================================================
# 1. Valid FastF1 Qualifying Identifiers
# =============================================================================

def test_valid_fastf1_qualifying_identifiers():
    """Verify session plans contain only supported FastF1 session identifiers."""
    # Standard schedule row
    standard_row = pd.Series({
        "RoundNumber": 1,
        "EventName": "Australian Grand Prix",
        "EventFormat": "conventional",
        "Session1": "Practice 1",
        "Session2": "Practice 2",
        "Session3": "Practice 3",
        "Session4": "Qualifying",
        "Session5": "Race",
    })
    plan_std = detect_event_format_and_sessions(standard_row, session_mode="prediction_only")["session_plan"]
    q_plan_std = next(item for item in plan_std if item["slug"] == "qualifying")
    assert "Quali" not in q_plan_std["candidates"], "Colloquial 'Quali' must NOT appear in qualifying candidates."
    assert set(q_plan_std["candidates"]).issubset({"Qualifying", "Q"})

    # Sprint schedule row
    sprint_row = pd.Series({
        "RoundNumber": 2,
        "EventName": "Chinese Grand Prix",
        "EventFormat": "sprint",
        "Session1": "Practice 1",
        "Session2": "Sprint Qualifying",
        "Session3": "Sprint",
        "Session4": "Qualifying",
        "Session5": "Race",
    })
    plan_sprint = detect_event_format_and_sessions(sprint_row, session_mode="prediction_only")["session_plan"]
    sq_plan = next(item for item in plan_sprint if item["slug"] == "sprint_qualifying")
    assert "Sprint Quali" not in sq_plan["candidates"], "'Sprint Quali' must NOT appear in sprint qualifying candidates."
    assert set(sq_plan["candidates"]).issubset({"Sprint Qualifying", "Sprint Shootout", "SQ"})


# =============================================================================
# 2. Unsupported 'Quali' Rejection & Error Handling
# =============================================================================

def test_unsupported_quali_rejection_and_error_handling(tmp_path):
    """Verify declared colloquial 'Quali' is sanitized, and FastF1 ValueErrors do not crash collection."""
    colloquial_row = pd.Series({
        "RoundNumber": 20,
        "EventName": "São Paulo Grand Prix",
        "EventFormat": "conventional",
        "Session1": "FP1",
        "Session2": "FP2",
        "Session3": "FP3",
        "Session4": "Quali",  # Colloquial declaration
        "Session5": "Race",
    })
    format_info = detect_event_format_and_sessions(colloquial_row, session_mode="prediction_only")
    q_plan = next(item for item in format_info["session_plan"] if item["slug"] == "qualifying")
    assert "Quali" not in q_plan["candidates"], "Colloquial 'Quali' must be sanitized and excluded."
    assert "Qualifying" in q_plan["candidates"]

    # Verify that if fastf1.get_session raises ValueError, collect_round_data handles it gracefully
    with patch("fastf1.get_session", side_effect=ValueError("Invalid session type 'Quali'")):
        summary = collect_round_data(
            year=2026,
            round_num=20,
            event_row=colloquial_row,
            output_dir=tmp_path,
            force=True,
            session_mode="prediction_only",
        )
        # Should not crash; should return summary
        assert isinstance(summary, dict)
        assert "round" in summary


# =============================================================================
# 3. Qualifying Session Loading With Messages Enabled
# =============================================================================

def test_qualifying_session_loading_with_messages_enabled(tmp_path):
    """Verify modern qualifying collection requests messages=True for track-limits deleted lap resolution."""
    event_row = pd.Series({
        "RoundNumber": 20,
        "EventName": "São Paulo Grand Prix",
        "Location": "São Paulo",
        "EventFormat": "conventional",
        "EventDate": "2024-11-03",
        "F1ApiSupport": True,
    })

    mock_sess = MagicMock()
    mock_sess.f1_api_support = True
    mock_sess.results = pd.DataFrame({
        "Position": [1.0, 2.0],
        "Abbreviation": ["VER", "NOR"],
        "FullName": ["Max Verstappen", "Lando Norris"],
        "TeamName": ["Red Bull Racing", "McLaren"],
        "DriverNumber": [1, 4],
    })

    with patch("fastf1.get_session", return_value=mock_sess):
        collect_round_data(
            year=2024,
            round_num=20,
            event_row=event_row,
            output_dir=tmp_path,
            force=True,
            session_mode="prediction_only",
        )
        # Verify sess.load call arguments
        calls = mock_sess.load.call_args_list
        assert len(calls) > 0

        # Check call for qualifying session
        q_call = calls[0]
        assert q_call.kwargs.get("messages") is True, "Qualifying session MUST load with messages=True"
        assert q_call.kwargs.get("telemetry") is False, "Telemetry should remain False"
        assert q_call.kwargs.get("weather") is False, "Weather should remain False"


# =============================================================================
# 4. Missing Race-Control Messages Handling
# =============================================================================

def test_qualifying_missing_race_control_messages(tmp_path):
    """Verify that if FastF1 fails due to missing deleted-laps telemetry, it is caught cleanly without crash."""
    event_row = pd.Series({
        "RoundNumber": 20,
        "EventName": "São Paulo Grand Prix",
        "Location": "São Paulo",
        "EventDate": "2024-11-03",
        "F1ApiSupport": True,
    })

    mock_sess = MagicMock()
    mock_sess.f1_api_support = True
    mock_sess.load.side_effect = Exception("Cannot calculate qualifying results: missing information about deleted laps.")

    with patch("fastf1.get_session", return_value=mock_sess):
        summary = collect_round_data(
            year=2024,
            round_num=20,
            event_row=event_row,
            output_dir=tmp_path,
            force=True,
            session_mode="prediction_only",
        )
        assert len(summary["successes"]) == 0
        assert summary["session_statuses"]["qualifying"]["status"] == "DATA_MISSING"


# =============================================================================
# 5. Missing Official Qualifying Results (No Silent Fabrication)
# =============================================================================

def test_missing_official_qualifying_results_not_fabricated():
    """Verify validate_and_format_session_results strictly rejects empty results without manufacturing positions."""
    mock_session = MagicMock()
    mock_session.results = pd.DataFrame()  # Empty results

    valid, msg, df = validate_and_format_session_results(
        session=mock_session,
        expected_year=2026,
        expected_round=20,
        event_name="São Paulo Grand Prix",
        circuit_name="Interlagos",
        country="Brazil",
        canonical_type="Qualifying",
    )
    assert valid is False
    assert df is None
    assert "empty" in msg.lower()


# =============================================================================
# 6. Strict after_qualifying Pipeline Rejection
# =============================================================================

def test_strict_after_qualifying_pipeline_rejection():
    """Verify RacePredictionPipeline raises ValueError for stage 'after_qualifying' when real qualifying is missing."""
    pipeline = RacePredictionPipeline()

    # Mock DataLoader so target qualifying returns None
    with patch.object(pipeline.data_loader, "load_target_qualifying", return_value=None), \
         patch.object(pipeline.data_loader, "load_starting_grid", return_value=(None, {"provenance": "missing_data"})):

        with pytest.raises(ValueError) as exc_info:
            pipeline.run(track_key="sao_paulo", year=2026, stage="after_qualifying")

        err = str(exc_info.value)
        assert "after_qualifying" in err
        assert "Official qualifying results are unavailable" in err

    # Also test predict_qualifying directly rejects after_qualifying
    with pytest.raises(ValueError) as exc_info:
        pipeline.predict_qualifying(track_key="sao_paulo", year=2026, stage="after_qualifying")
    assert "after_qualifying" in str(exc_info.value)


# =============================================================================
# 7. Provisional Prediction Under Permitted Stages
# =============================================================================

def test_provisional_prediction_under_permitted_stages():
    """Verify provisional qualifying prediction is permitted under appropriate pre-qualifying stages."""
    pipeline = RacePredictionPipeline()

    mock_provisional = {
        "predictions": pd.DataFrame({
            "Abbreviation": ["VER", "NOR"],
            "TeamName": ["Red Bull Racing", "McLaren"],
            "PredictedPosition": [1.0, 2.0],
        })
    }

    # Under stage='after_fp3', missing qualifying results triggers predict_qualifying rather than raising ValueError
    with patch.object(pipeline.data_loader, "load_target_qualifying", return_value=None), \
         patch.object(pipeline.data_loader, "load_starting_grid", return_value=(None, {"provenance": "unpublished"})), \
         patch.object(pipeline, "predict_qualifying", return_value=mock_provisional) as mock_predict_q, \
         patch.object(pipeline.feature_engineer, "build_training_dataset", side_effect=RuntimeError("HALT_AFTER_PROVISIONAL")):

        with pytest.raises(RuntimeError) as exc_info:
            pipeline.run(track_key="sao_paulo", year=2026, stage="after_fp3")

        assert "HALT_AFTER_PROVISIONAL" in str(exc_info.value)
        assert mock_predict_q.called, "predict_qualifying MUST be invoked under stage 'after_fp3' when quali is missing"

    # Also verify predict_qualifying directly accepts permitted pre-qualifying stages
    with patch.object(pipeline.data_loader, "load_season_rounds", return_value=(pd.DataFrame(), pd.DataFrame())), \
         patch.object(pipeline.data_loader, "load_all_practice_sessions", return_value={}), \
         patch.object(pipeline.feature_engineer, "build_training_dataset", side_effect=RuntimeError("HALT_AT_TRAIN")):
        with pytest.raises(RuntimeError) as exc_info:
            pipeline.predict_qualifying(track_key="sao_paulo", year=2026, stage="after_fp3")
        assert "HALT_AT_TRAIN" in str(exc_info.value)


# =============================================================================
# 8. Future Race Session Classification
# =============================================================================

def test_future_race_session_classification(tmp_path):
    """Verify a race scheduled in the future is classified as FUTURE_SESSION, not COLLECTION_FAILED."""
    future_time_utc = (datetime.now(timezone.utc) + timedelta(days=14)).strftime("%Y-%m-%d %H:%M:%S")

    event_row = pd.Series({
        "RoundNumber": 22,
        "EventName": "Future Grand Prix",
        "Location": "Lusail",
        "Country": "Qatar",
        "EventFormat": "conventional",
        "Session5": "Race",
        "Session5DateUtc": future_time_utc,
        "EventDate": future_time_utc,
        "F1ApiSupport": True,
    })

    # When FastF1 has no data because session hasn't occurred
    mock_sess = MagicMock()
    mock_sess.f1_api_support = True
    mock_sess.results = None

    with patch("fastf1.get_session", return_value=mock_sess):
        summary = collect_round_data(
            year=2026,
            round_num=22,
            event_row=event_row,
            output_dir=tmp_path,
            force=True,
            session_mode="prediction_only",
        )

        # Race session must be classified as FUTURE_SESSION
        race_status = summary["session_statuses"]["race"]
        assert race_status["status"] == "FUTURE_SESSION"
        assert "has not yet taken place" in race_status["detail"]

        # MUST NOT be placed into failures!
        assert len(summary["failures"]) == 0, f"Future race must NOT be in failures: {summary['failures']}"
        assert any("[FUTURE_SESSION]" in u for u in summary["unavailable"])


# =============================================================================
# 9. Empty Race-Session Response Handling
# =============================================================================

def test_empty_race_session_response_handling(tmp_path):
    """Verify past session with empty archive data is classified as DATA_MISSING, not failure."""
    past_time_utc = (datetime.now(timezone.utc) - timedelta(days=100)).strftime("%Y-%m-%d %H:%M:%S")

    event_row = pd.Series({
        "RoundNumber": 1,
        "EventName": "Past Grand Prix",
        "Location": "Silverstone",
        "Country": "Great Britain",
        "EventFormat": "conventional",
        "Session5": "Race",
        "Session5DateUtc": past_time_utc,
        "EventDate": past_time_utc,
        "F1ApiSupport": True,
    })

    mock_sess = MagicMock()
    mock_sess.f1_api_support = True
    mock_sess.results = pd.DataFrame()  # Empty dataframe from archive

    with patch("fastf1.get_session", return_value=mock_sess):
        summary = collect_round_data(
            year=2020,
            round_num=1,
            event_row=event_row,
            output_dir=tmp_path,
            force=True,
            session_mode="prediction_only",
        )

        race_status = summary["session_statuses"]["race"]
        assert race_status["status"] == "DATA_MISSING"
        assert len(summary["failures"]) == 0


# =============================================================================
# 10. OpenF1 Timeout Handling
# =============================================================================

def test_openf1_timeout_handling(tmp_path):
    """Verify StartingGridRetriever handles OpenF1 timeouts gracefully without unhandled exceptions."""
    retriever = StartingGridRetriever(root_dir=tmp_path)

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timed out")):
        grid_df, meta = retriever._retrieve_openf1(
            track_key="sao_paulo",
            year=2026,
            round_num=20,
            stage="after_qualifying",
        )
        assert grid_df is None
        assert meta == {}
