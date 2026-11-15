"""
Unit tests for ApexGrid Historical Analytics Subsystem — Phase 2 Events & Circuits
===================================================================================
Tests HistoricalEventsService and HistoricalCircuitsService using mocked API responses.
Enforces:
- Separation of calendar scheduling from confirmed race completion.
- Missing circuit names remain None (zero synthetic title-case generation).
- Missing event dates produce 'Date Unknown' (not upcoming or completed).
- Past scheduled dates do not inflate completed_rounds (completed_rounds = 0 for calendar).
- Advisory cancellation heuristic is clearly labeled.
- Zero filesystem persistence invariant.
"""

from datetime import datetime, timezone
import pytest
from unittest.mock import MagicMock, patch

from src.historical.models import HistoricalCircuit, HistoricalEvent, SeasonCalendar
from src.historical.in_memory_cache import InMemoryTTLCache
from src.historical.api_client import HistoricalAPIClient
from src.historical.events_service import HistoricalEventsService
from src.historical.circuits_service import HistoricalCircuitsService


# -----------------------------------------------------------------------------
# Mock Data Fixtures
# -----------------------------------------------------------------------------

MOCK_1950_CALENDAR_RESPONSE = {
    "MRData": {
        "total": "2",
        "RaceTable": {
            "season": "1950",
            "Races": [
                {
                    "season": "1950",
                    "round": "1",
                    "url": "https://en.wikipedia.org/wiki/1950_British_Grand_Prix",
                    "raceName": "British Grand Prix",
                    "Circuit": {
                        "circuitId": "silverstone",
                        "circuitName": "Silverstone Circuit",
                        "Location": {"locality": "Silverstone", "country": "UK", "lat": "52.0786", "long": "-1.01694"}
                    },
                    "date": "1950-05-13"
                    # Note: 'time' is omitted for 1950
                },
                {
                    "season": "1950",
                    "round": "2",
                    "url": "https://en.wikipedia.org/wiki/1950_Monaco_Grand_Prix",
                    "raceName": "Monaco Grand Prix",
                    "Circuit": {
                        "circuitId": "monaco",
                        "circuitName": "Circuit de Monaco",
                        "Location": {"locality": "Monte-Carlo", "country": "Monaco", "lat": "43.7347", "long": "7.42056"}
                    },
                    "date": "1950-05-21"
                }
            ]
        }
    }
}

MOCK_2026_CALENDAR_RESPONSE = {
    "MRData": {
        "total": "4",
        "RaceTable": {
            "season": "2026",
            "Races": [
                {
                    "season": "2026",
                    "round": "1",
                    "raceName": "Australian Grand Prix",
                    "Circuit": {"circuitId": "albert_park", "circuitName": "Albert Park Grand Prix Circuit"},
                    "date": "2026-03-15",
                    "time": "05:00:00Z"
                },
                {
                    "season": "2026",
                    "round": "2",
                    "raceName": "Chinese Grand Prix",
                    "Circuit": {"circuitId": "shanghai", "circuitName": "Shanghai International Circuit"},
                    "date": "2026-03-22",
                    "time": "07:00:00Z"
                },
                {
                    "season": "2026",
                    "round": "3",
                    "raceName": "Emilia Romagna Grand Prix [Cancelled]",
                    "Circuit": {"circuitId": "imola", "circuitName": "Autodromo Enzo e Dino Ferrari"},
                    "date": "2026-05-17",
                    "time": "13:00:00Z"
                },
                {
                    "season": "2026",
                    "round": "4",
                    "raceName": "Monaco Grand Prix",
                    "Circuit": {"circuitId": "monaco", "circuitName": "Circuit de Monaco"},
                    "date": "2026-05-24",
                    "time": "13:00:00Z"
                }
            ]
        }
    }
}

MOCK_CIRCUITS_PAGE_RESPONSE = {
    "MRData": {
        "total": "3",
        "limit": "100",
        "offset": "0",
        "CircuitTable": {
            "Circuits": [
                {
                    "circuitId": "monza",
                    "circuitName": "Autodromo Nazionale di Monza",
                    "url": "https://en.wikipedia.org/wiki/Monza_Circuit",
                    "Location": {
                        "lat": "45.6156",
                        "long": "9.28111",
                        "locality": "Monza",
                        "country": "Italy"
                    }
                },
                {
                    "circuitId": "bremgarten",
                    "circuitName": "Circuit Bremgarten",
                    "url": "https://en.wikipedia.org/wiki/Circuit_Bremgarten",
                    "Location": {
                        # Coordinates omitted or unrecorded in historical entry
                        "locality": "Bern",
                        "country": "Switzerland"
                    }
                },
                {
                    # Circuit entry with missing circuitName
                    "circuitId": "watkins_glen",
                    # circuitName omitted by provider
                    "url": "https://en.wikipedia.org/wiki/Watkins_Glen_International",
                    "Location": {
                        "locality": "Watkins Glen",
                        "country": "USA",
                        "lat": "42.3369",
                        "long": "-76.9272"
                    }
                }
            ]
        }
    }
}


# -----------------------------------------------------------------------------
# 1. Historical Events Service Tests
# -----------------------------------------------------------------------------

def test_events_service_past_scheduled_dates_do_not_inflate_completed_count():
    """Verify past scheduled dates are NOT treated as completed races (completed_rounds = 0)."""
    cache = InMemoryTTLCache()
    client = HistoricalAPIClient(cache=cache)
    service = HistoricalEventsService(api_client=client)

    with patch.object(client, "get_json", return_value=MOCK_1950_CALENDAR_RESPONSE):
        # Freeze reference date well after 1950
        calendar = service.get_season_calendar(1950, ref_date=datetime(2026, 1, 1, tzinfo=timezone.utc))

        assert calendar.year == 1950
        assert calendar.total_rounds == 2
        # CRITICAL: Past scheduled dates do NOT inflate completed_rounds!
        assert calendar.completed_rounds == 0
        assert calendar.past_scheduled_rounds == 2
        assert calendar.upcoming_rounds == 0
        assert not calendar.is_in_progress
        # Must NOT claim "Completed Championship" from a calendar query
        assert "Schedule Concluded" in calendar.status_display
        assert "Results Required to Confirm Completion" in calendar.status_display

        # Check round 1 event attributes
        ev1 = calendar.events[0]
        assert ev1.round == 1
        assert ev1.event_name == "British Grand Prix"
        assert ev1.circuit_id == "silverstone"
        assert ev1.circuit_name == "Silverstone Circuit"
        assert ev1.date == "1950-05-13"
        assert ev1.time is None  # Zero fabrication
        # Crucial status invariants:
        assert ev1.is_completed is False
        assert ev1.is_upcoming is False
        assert ev1.is_cancelled is False
        assert ev1.status_label == "Scheduled (Past Date)"


def test_events_service_current_season_2026_in_progress():
    """Verify 2026 season distinguishes past dates, upcoming rounds, and advisory cancellations."""
    cache = InMemoryTTLCache()
    client = HistoricalAPIClient(cache=cache)
    service = HistoricalEventsService(api_client=client)

    # Freeze reference date to March 20, 2026:
    # Round 1 (March 15) date has elapsed.
    # Round 2 (March 22) is upcoming.
    # Round 3 (May 17) has [Cancelled] in name.
    # Round 4 (May 24) is upcoming.
    ref_date = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)

    with patch.object(client, "get_json", return_value=MOCK_2026_CALENDAR_RESPONSE):
        calendar = service.get_season_calendar(2026, ref_date=ref_date)

        assert calendar.year == 2026
        assert calendar.total_rounds == 4
        assert calendar.completed_rounds == 0  # Unconfirmed from calendar alone
        assert calendar.past_scheduled_rounds == 1
        assert calendar.upcoming_rounds == 2
        assert calendar.is_in_progress is True
        assert "In-Progress: 1 Dates Elapsed, 2 Upcoming of 4 Scheduled" in calendar.status_display

        # Round 1: Past scheduled date (not completed!)
        assert calendar.events[0].is_completed is False
        assert calendar.events[0].is_upcoming is False
        assert calendar.events[0].status_label == "Scheduled (Past Date)"

        # Round 2: Upcoming
        assert calendar.events[1].is_upcoming is True
        assert calendar.events[1].status_label == "Upcoming"

        # Round 3: Advisory cancellation heuristic (is_cancelled must remain False!)
        assert calendar.events[2].is_cancelled is False
        assert calendar.events[2].is_completed is False
        assert calendar.events[2].is_upcoming is False
        assert calendar.events[2].status_label == "Advisory: Likely Cancelled"

        # Round 4: Upcoming
        assert calendar.events[3].is_upcoming is True


def test_advisory_cancellation_does_not_set_is_cancelled_true():
    """Verify that a race name containing '[Cancelled]' does NOT set is_cancelled=True, but keeps advisory label."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = HistoricalEventsService(api_client=client)

    cancelled_event_payload = {
        "MRData": {
            "total": "1",
            "RaceTable": {
                "season": "2023",
                "Races": [
                    {
                        "round": "6",
                        "raceName": "Emilia Romagna Grand Prix [Cancelled]",
                        "Circuit": {"circuitId": "imola", "circuitName": "Autodromo Enzo e Dino Ferrari"},
                        "date": "2023-05-21",
                        "time": "13:00:00Z"
                    }
                ]
            }
        }
    }

    with patch.object(client, "get_json", return_value=cancelled_event_payload):
        calendar = service.get_season_calendar(2023)
        assert len(calendar.events) == 1
        ev = calendar.events[0]
        # CRITICAL INVARIANT: Confirmed cancellation cannot be established from provider calendar alone!
        assert ev.is_cancelled is False
        assert ev.is_completed is False
        assert ev.is_upcoming is False
        assert ev.status_label == "Advisory: Likely Cancelled"


def test_events_service_missing_event_date_produces_unknown_status():
    """Verify missing or invalid event date produces Date Unknown (neither completed nor upcoming)."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = HistoricalEventsService(api_client=client)

    mock_resp = {
        "MRData": {
            "total": "2",
            "RaceTable": {
                "season": "2026",
                "Races": [
                    {
                        "round": "1",
                        "raceName": "TBD Grand Prix",
                        "Circuit": {"circuitId": "unknown_venue"},
                        "date": None  # Missing date
                    },
                    {
                        "round": "2",
                        "raceName": "Pending Date Grand Prix",
                        "Circuit": {"circuitId": "unknown_venue_2"},
                        "date": "   "  # Whitespace date
                    }
                ]
            }
        }
    }

    with patch.object(client, "get_json", return_value=mock_resp):
        calendar = service.get_season_calendar(2026)
        assert len(calendar.events) == 2
        for ev in calendar.events:
            assert ev.is_completed is False
            assert ev.is_upcoming is False
            assert ev.is_cancelled is False
            assert ev.status_label == "Date Unknown"


def test_events_service_partial_calendar_not_labelled_completed_championship():
    """Verify a partial calendar where all listed dates have passed is not labelled a completed championship."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = HistoricalEventsService(api_client=client)

    partial_calendar = {
        "MRData": {
            "total": "3",
            "RaceTable": {
                "season": "2026",
                "Races": [
                    {"round": "1", "raceName": "GP 1", "Circuit": {"circuitId": "c1"}, "date": "2026-03-01"},
                    {"round": "2", "raceName": "GP 2", "Circuit": {"circuitId": "c2"}, "date": "2026-03-08"},
                    {"round": "3", "raceName": "GP 3", "Circuit": {"circuitId": "c3"}, "date": "2026-03-15"}
                ]
            }
        }
    }

    # Reference date after round 3
    ref_date = datetime(2026, 4, 1, tzinfo=timezone.utc)

    with patch.object(client, "get_json", return_value=partial_calendar):
        calendar = service.get_season_calendar(2026, ref_date=ref_date)
        # Even though all 3 dates have elapsed, it must NOT claim completed championship
        assert "Schedule Concluded" in calendar.status_display
        assert "Completed Championship" not in calendar.status_display
        assert calendar.completed_rounds == 0
        assert calendar.past_scheduled_rounds == 3


def test_events_service_empty_and_unavailable_season():
    """Verify handling when provider returns empty or missing calendar."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = HistoricalEventsService(api_client=client)

    empty_response = {"MRData": {"total": "0", "RaceTable": {"season": "1940", "Races": []}}}

    with patch.object(client, "get_json", return_value=empty_response):
        calendar = service.get_season_calendar(1940)
        assert calendar.total_rounds == 0
        assert calendar.completed_rounds == 0
        assert not calendar.is_in_progress
        assert "Calendar Pending" in calendar.status_display


# -----------------------------------------------------------------------------
# 2. Historical Circuits Service Tests
# -----------------------------------------------------------------------------

def test_circuits_service_missing_name_remains_none_and_preserves_id():
    """Verify missing circuit name remains None (no title-case synthesis from ID)."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = HistoricalCircuitsService(api_client=client)

    with patch.object(client, "get_all_pages", return_value=MOCK_CIRCUITS_PAGE_RESPONSE["MRData"]["CircuitTable"]["Circuits"]):
        circuits = service.get_all_circuits()
        assert len(circuits) == 3

        # Circuit 3: Watkins Glen (circuitName omitted in mock response)
        c3 = circuits[2]
        assert c3.circuit_id == "watkins_glen"
        # Zero synthetic name generation: must be None!
        assert c3.name is None
        assert c3.locality == "Watkins Glen"
        assert c3.country == "USA"
        assert c3.latitude == 42.3369


def test_circuits_service_all_circuits_and_missing_coordinates():
    """Verify circuit metadata parsing, URL retention, and graceful missing coordinate handling."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = HistoricalCircuitsService(api_client=client)

    with patch.object(client, "get_all_pages", return_value=MOCK_CIRCUITS_PAGE_RESPONSE["MRData"]["CircuitTable"]["Circuits"]):
        circuits = service.get_all_circuits()

        # Circuit 1: Monza (all coordinates present)
        c1 = circuits[0]
        assert c1.circuit_id == "monza"
        assert c1.name == "Autodromo Nazionale di Monza"
        assert c1.locality == "Monza"
        assert c1.country == "Italy"
        assert c1.latitude == 45.6156
        assert c1.longitude == 9.28111
        assert "wikipedia" in c1.url

        # Circuit 2: Bremgarten (historical venue with missing coordinates in mock)
        c2 = circuits[1]
        assert c2.circuit_id == "bremgarten"
        assert c2.name == "Circuit Bremgarten"
        assert c2.locality == "Bern"
        assert c2.country == "Switzerland"
        assert c2.latitude is None  # Zero fabrication: must be None
        assert c2.longitude is None


def test_circuits_service_get_single_circuit():
    """Verify specific circuit lookup by ID."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = HistoricalCircuitsService(api_client=client)

    mock_monza_resp = {
        "MRData": {
            "CircuitTable": {
                "Circuits": [MOCK_CIRCUITS_PAGE_RESPONSE["MRData"]["CircuitTable"]["Circuits"][0]]
            }
        }
    }

    with patch.object(client, "get_json", return_value=mock_monza_resp):
        c = service.get_circuit("monza")
        assert c is not None
        assert c.circuit_id == "monza"
        assert c.name == "Autodromo Nazionale di Monza"

    with patch.object(client, "get_json", return_value={"MRData": {"CircuitTable": {"Circuits": []}}}):
        nonexistent = service.get_circuit("nonexistent_circuit_id")
        assert nonexistent is None


# -----------------------------------------------------------------------------
# 3. Integration & Zero Disk Persistence Tests
# -----------------------------------------------------------------------------

def test_cache_reuse_across_services():
    """Verify that multiple service calls leverage the in-memory cache to avoid duplicate HTTP requests."""
    cache = InMemoryTTLCache()
    client = HistoricalAPIClient(cache=cache)
    events_service = HistoricalEventsService(api_client=client)

    with patch.object(client, "_fetch_with_retries", return_value=MOCK_1950_CALENDAR_RESPONSE) as mock_fetch:
        cal1 = events_service.get_season_calendar(1950)
        assert mock_fetch.call_count == 1

        cal2 = events_service.get_season_calendar(1950)
        assert mock_fetch.call_count == 1
        assert len(cal1.events) == len(cal2.events)


def test_phase2_zero_disk_persistence():
    """Assert that no file writes, database creations, or persistent files occur during Phase 2 services."""
    import builtins
    original_open = builtins.open
    write_calls = []

    def guard_open(file, mode="r", *args, **kwargs):
        if any(m in mode for m in ("w", "a", "x", "+")):
            write_calls.append((file, mode))
            raise AssertionError(f"PROHIBITED DISK WRITE: {file}")
        return original_open(file, mode, *args, **kwargs)

    with patch("builtins.open", side_effect=guard_open):
        cache = InMemoryTTLCache()
        client = HistoricalAPIClient(cache=cache)
        events_service = HistoricalEventsService(api_client=client)
        circuits_service = HistoricalCircuitsService(api_client=client)

        with patch.object(client, "get_json", return_value=MOCK_2026_CALENDAR_RESPONSE):
            _ = events_service.get_season_calendar(2026)

        with patch.object(client, "get_all_pages", return_value=MOCK_CIRCUITS_PAGE_RESPONSE["MRData"]["CircuitTable"]["Circuits"]):
            _ = circuits_service.get_all_circuits()

    assert len(write_calls) == 0, f"Detected illegal disk write operations: {write_calls}"
