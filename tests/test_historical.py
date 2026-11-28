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


def test_view_circuit_history_numeric_selection_qatar(monkeypatch):
    from unittest.mock import MagicMock
    from apexgrid.cli.historical_menu import HistoricalStatsMenu
    from apexgrid.historical.analytics.models import CircuitAnalyticsSummary

    menu = HistoricalStatsMenu()
    mock_summary = CircuitAnalyticsSummary(
        circuit_id="losail",
        circuit_name="Lusail International Circuit",
        locality="Lusail",
        country="Qatar",
        first_gp_year=2021,
        last_gp_year=2025,
        total_races_hosted=4,
        top_winning_drivers=[("Max Verstappen", 2)],
        top_winning_constructors=[("Red Bull", 2)],
        top_podium_drivers=[("Max Verstappen", 3)],
        recent_winners=[(2025, "Max Verstappen", "Red Bull")],
    )
    menu.circuit_service.get_circuit_history = MagicMock(return_value=mock_summary)

    # Round 22 in tracks.json corresponds to Qatar -> 'losail'
    prompts = iter(["22", ""])
    monkeypatch.setattr("apexgrid.cli.historical_menu.prompt_str", lambda *args, **kwargs: next(prompts))

    menu.view_circuit_history()
    menu.circuit_service.get_circuit_history.assert_called_once_with("losail")


def test_render_circuit_spec_card_handles_partial_metadata():
    from apexgrid.cli.historical_menu import HistoricalStatsMenu

    menu = HistoricalStatsMenu()
    # Partial dict with missing turns, drs_zones, lap_record, etc. must not raise KeyError
    menu._render_circuit_spec_card({"circuit_name": "Partial Circuit", "circuit_type": "permanent"})
