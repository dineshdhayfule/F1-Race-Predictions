from datetime import datetime, timedelta, timezone

import pandas as pd

from apexgrid.data.collect_season import detect_event_format_and_sessions, get_allowed_stage_sessions
from apexgrid.data.collect_track import collect_track_data


def _standard_row():
    return pd.Series({
        "EventFormat": "conventional",
        "Session1": "Practice 1",
        "Session2": "Practice 2",
        "Session3": "Practice 3",
        "Session4": "Qualifying",
        "Session5": "Race",
    })


def _sprint_row():
    return pd.Series({
        "EventFormat": "sprint",
        "Session1": "Practice 1",
        "Session2": "Sprint Qualifying",
        "Session3": "Sprint",
        "Session4": "Qualifying",
        "Session5": "Race",
    })


def test_standard_weekend_session_plan():
    detected = detect_event_format_and_sessions(_standard_row(), session_mode="prediction_only")
    assert detected["is_sprint"] is False
    assert [item["slug"] for item in detected["session_plan"]] == ["qualifying", "race"]


def test_sprint_weekend_session_plan():
    detected = detect_event_format_and_sessions(_sprint_row(), session_mode="prediction_only")
    assert detected["is_sprint"] is True
    assert [item["slug"] for item in detected["session_plan"]] == [
        "sprint_qualifying", "sprint_race", "qualifying", "race",
    ]


def test_full_mode_includes_standard_practice():
    detected = detect_event_format_and_sessions(_standard_row(), session_mode="full")
    slugs = [item["slug"] for item in detected["session_plan"]]
    assert slugs == ["fp1", "fp2", "fp3", "qualifying", "race"]


def test_sprint_full_mode_includes_only_fp1():
    detected = detect_event_format_and_sessions(_sprint_row(), session_mode="full")
    slugs = [item["slug"] for item in detected["session_plan"]]
    assert "fp1" in slugs
    assert "fp2" not in slugs
    assert "fp3" not in slugs


def test_stage_pre_fp1_collects_no_sessions():
    assert get_allowed_stage_sessions("pre_fp1", False) == []
    assert get_allowed_stage_sessions("pre_fp1", True) == []


def test_stage_after_fp1_collects_only_fp1():
    assert get_allowed_stage_sessions("after_fp1", False) == ["fp1"]
    assert get_allowed_stage_sessions("after_fp1", True) == ["fp1"]


def test_stage_after_qualifying_standard_plan():
    assert get_allowed_stage_sessions("after_qualifying", False) == [
        "fp1", "fp2", "fp3", "qualifying",
    ]


def test_stage_after_sprint_race_plan():
    assert get_allowed_stage_sessions("after_sprint_race", True) == [
        "fp1", "sprint_qualifying", "sprint_race",
    ]


def test_stage_aliases_are_normalized():
    assert get_allowed_stage_sessions(" AFTER_FP2 ", False) == ["fp1", "fp2"]
    assert get_allowed_stage_sessions("after_sprint", True) == [
        "fp1", "sprint_qualifying", "sprint_race",
    ]


def test_track_collector_delegates_stage_and_mode(monkeypatch):
    import apexgrid.data.collect_track as track_module
    import apexgrid.data.collect_season as season_module
    captured = {}

    def fake_collect_grand_prix(**kwargs):
        captured.update(kwargs)
        return {"successes": []}

    monkeypatch.setattr(season_module, "collect_grand_prix", fake_collect_grand_prix)
    track_module.collect_track_data(
        "austin", year=2026, force=True,
        session_mode="practice_only", stage="after_fp1",
    )
    assert captured["track_id"] == "austin"
    assert captured["force"] is True
    assert captured["session_mode"] == "practice_only"
    assert captured["stage"] == "after_fp1"


def test_future_schedule_is_detected_before_download():
    from apexgrid.data.collect_season import resolve_session_scheduled_time
    event_row = pd.Series({
        "Session1DateUtc": "2099-05-15 10:00:00",
        "Session1": "Practice 1",
    })
    sched_dt = resolve_session_scheduled_time(event_row, "Practice 1", "fp1", ["FP1", "Practice 1"])
    assert sched_dt is not None
    ref_now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert sched_dt > ref_now


def test_before_race_session_plan():
    assert get_allowed_stage_sessions("before_race", False) == [
        "fp1", "fp2", "fp3", "qualifying",
    ]
    assert get_allowed_stage_sessions("before_race", True) == [
        "fp1", "sprint_qualifying", "sprint_race", "qualifying",
    ]


def test_post_race_session_plan_includes_only_observed_results():
    assert get_allowed_stage_sessions("post_race", False) == [
        "fp1", "fp2", "fp3", "qualifying", "race",
    ]
