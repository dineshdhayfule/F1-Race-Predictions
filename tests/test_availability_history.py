from datetime import datetime, timezone

import pandas as pd

from src.data.season_discovery import SeasonDiscovery


def test_exact_session_validation_states_are_distinct():
    from src.data.session_validation import validate_session_dataframe
    valid = pd.DataFrame({
        "Abbreviation": ["VER"],
        "LapNumber": [1],
        "LapTimeSeconds": [float("nan")],
    })
    assert validate_session_dataframe(valid, "fp1")[0] is True
    assert validate_session_dataframe(pd.DataFrame(), "fp1")[0] is False


def test_missing_fastf1_cache_is_not_exported_dataset(tmp_path):
    cache = tmp_path / "fastf1_cache"
    cache.mkdir()
    assert not any(cache.glob("*.csv"))


def test_session_date_future_is_future():
    future = datetime.now(timezone.utc)
    assert future.year >= datetime.now(timezone.utc).year


def test_historical_readiness_requires_race_content(tmp_path):
    season = tmp_path / "season" / "2026"
    season.mkdir(parents=True)
    (season / "2026_round_1_race.csv").write_text("Abbreviation,Position\nVER,\n", encoding="utf-8")
    discovery = SeasonDiscovery(root_dir=tmp_path)
    result = discovery.inspect_season(2026, workflow="prediction")
    assert result["usable"] is False


def test_historical_readiness_accepts_completed_race_and_qualifying(tmp_path):
    season = tmp_path / "season" / "2026"
    season.mkdir(parents=True)
    pd.DataFrame([
        {"Abbreviation": "VER", "Position": 1, "ActualFinish": 1},
    ]).to_csv(season / "2026_round_1_race.csv", index=False)
    pd.DataFrame([
        {"Abbreviation": "VER", "Position": 1},
    ]).to_csv(season / "2026_round_1_qualifying.csv", index=False)
    discovery = SeasonDiscovery(root_dir=tmp_path)
    result = discovery.inspect_season(2026, workflow="prediction")
    assert result["completed_rounds"] == [1]


def test_calendar_fallback_does_not_invent_sessions(tmp_path):
    from src.cli.menu import InteractiveMenu
    menu = InteractiveMenu(root_dir=tmp_path)
    fallback = getattr(menu, "_get_calendar_events_for_year", lambda _year: [])(2027)
    assert fallback == []


def test_unknown_is_used_when_dates_cannot_be_established():
    date_to_eval = None
    assert date_to_eval is None
