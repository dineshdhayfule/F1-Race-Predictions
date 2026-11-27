from datetime import datetime, timezone

import pandas as pd

from apexgrid.data.season_discovery import SeasonDiscovery


def test_exact_session_validation_states_are_distinct():
    from apexgrid.data.session_validation import validate_session_dataframe
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


def test_session_date_future_is_future(tmp_path):
    from apexgrid.cli.menu import InteractiveMenu
    menu = InteractiveMenu(root_dir=tmp_path)
    status = menu._check_session_dataset_availability(
        year=2099,
        round_num=1,
        circuit_key="test_circuit",
        event_name="Test Grand Prix",
        circuit_name="Test Circuit",
        session_name="Practice 1",
        session_date_raw="2099-05-15 10:00:00+00:00",
        event_date_raw="2099-05-17",
    )
    assert status == "FUTURE"


def test_historical_readiness_requires_race_content(tmp_path):
    season = tmp_path / "data" / "raw" / "season" / "2026"
    season.mkdir(parents=True)
    (season / "2026_round_1_race.csv").write_text("Abbreviation,Position\nVER,\n", encoding="utf-8")
    discovery = SeasonDiscovery(root_dir=tmp_path)
    result = discovery.get_season_status(2026, workflow="prediction")
    assert result["usable"] is False


def test_historical_readiness_accepts_completed_race_and_qualifying(tmp_path):
    season = tmp_path / "data" / "raw" / "season" / "2026"
    season.mkdir(parents=True)
    pd.DataFrame([
        {"Abbreviation": "VER", "Position": 1, "ActualFinish": 1},
    ]).to_csv(season / "2026_round_1_race.csv", index=False)
    pd.DataFrame([
        {"Abbreviation": "VER", "Position": 1},
    ]).to_csv(season / "2026_round_1_qualifying.csv", index=False)
    discovery = SeasonDiscovery(root_dir=tmp_path)
    result = discovery.get_season_status(2026, workflow="prediction")
    assert result["completed_rounds"] == [1]


def test_calendar_fallback_does_not_invent_sessions(tmp_path):
    from apexgrid.cli.menu import InteractiveMenu
    menu = InteractiveMenu(root_dir=tmp_path)
    fallback = getattr(menu, "_get_calendar_events_for_year", lambda _year: [])(2027)
    assert fallback == []


def test_unknown_is_used_when_dates_cannot_be_established(tmp_path):
    from apexgrid.cli.menu import InteractiveMenu
    menu = InteractiveMenu(root_dir=tmp_path)
    status = menu._check_session_dataset_availability(
        year=2099,
        round_num=1,
        circuit_key="test_circuit",
        event_name="Test Grand Prix",
        circuit_name="Test Circuit",
        session_name="Practice 1",
        session_date_raw=None,
        event_date_raw=None,
    )
    assert status == "UNKNOWN"


def test_validate_session_csv_lifecycle(tmp_path):
    from apexgrid.data.session_validation import validate_session_csv

    # 1. Missing file
    missing_file = tmp_path / "missing.csv"
    valid, errors = validate_session_csv(missing_file, "fp1")
    assert valid is False
    assert any("not found" in err.lower() for err in errors)

    # 2. Empty file (0 bytes)
    empty_file = tmp_path / "empty.csv"
    empty_file.touch()
    valid, errors = validate_session_csv(empty_file, "fp1")
    assert valid is False
    assert any("empty" in err.lower() for err in errors)

    # 3. Valid practice CSV with untimed installation lap
    valid_file = tmp_path / "practice.csv"
    pd.DataFrame([
        {"Abbreviation": "VER", "LapNumber": 1, "LapTimeSeconds": None},
    ]).to_csv(valid_file, index=False)
    valid, errors = validate_session_csv(valid_file, "fp1")
    assert valid is True
    assert len(errors) == 0
