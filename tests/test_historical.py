from datetime import datetime, timezone

from apexgrid.historical.in_memory_cache import InMemoryTTLCache
from apexgrid.historical.events_service import HistoricalEventsService
from apexgrid.historical.models import interpret_grid_position
from apexgrid.historical.classification_service import HistoricalClassificationService


def test_historical_api_cache_is_process_scoped():
    cache = InMemoryTTLCache()
    assert cache.size() == 0
    cache.set("key", {"value": 1})
    assert cache.get("key") == {"value": 1}


def test_event_status_unknown_for_missing_date():
    service = HistoricalEventsService()
    is_completed, is_upcoming, is_cancelled, label = service._evaluate_event_status(
        date_str="", time_str="", race_name="2026 Grand Prix",
        now_utc=datetime.now(timezone.utc),
    )
    assert is_completed is False
    assert is_upcoming is False
    assert is_cancelled is False
    assert label == "Date Unknown"


def test_event_cancelled_label_is_advisory_only():
    service = HistoricalEventsService()
    is_completed, is_upcoming, is_cancelled, label = service._evaluate_event_status(
        date_str="2026-03-01", time_str="12:00",
        race_name="2026 Grand Prix [Cancelled]",
        now_utc=datetime.now(timezone.utc),
    )
    assert is_completed is False
    assert is_upcoming is False
    assert is_cancelled is False
    assert "Cancelled" in label


def test_grid_position_parser_preserves_pit_lane():
    position, status = interpret_grid_position("0")
    assert position is None
    assert status == "Unclassified / Non-standard Grid Position (0)"


def test_grid_position_parser_rejects_missing_value():
    position, status = interpret_grid_position(None)
    assert position is None
    assert status == "Grid position unrecorded"


def test_classification_parser_preserves_dns_status():
    service = HistoricalClassificationService()
    entries = service._parse_race_results([{
        "position": None,
        "positionText": "R",
        "grid": "10",
        "laps": 3,
        "status": "Did not start",
        "Driver": {"driverId": "test", "givenName": "Test", "familyName": "Driver"},
        "Constructor": {"constructorId": "test_team", "name": "Test Team"},
    }])
    assert len(entries) == 1
    assert entries[0].status == "Did not start"
