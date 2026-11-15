"""
ApexGrid — Historical Analytics Data Completeness Regression Tests
===================================================================
Tests for verified completeness across multi-page nested endpoints,
round merging across page boundaries, circuit analytics reconciliation,
and season/teammate aggregation integrity.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.historical.api_client import HistoricalAPIClient, HistoricalAPIError
from src.historical.in_memory_cache import InMemoryTTLCache
from src.historical.classification_service import HistoricalClassificationService
from src.historical.grid_service import HistoricalGridService
from src.historical.analytics.teammate_analytics import TeammateAnalyticsService
from src.historical.analytics.circuit_analytics import CircuitAnalyticsService
from src.historical.analytics.season_analytics import SeasonAnalyticsService
from src.historical.events_service import HistoricalEventsService
from src.historical.qualifying_service import HistoricalQualifyingService
from src.historical.models import (
    RaceResultEntry,
    StartingGridResultEntry,
    HistoricalCircuit,
    SeasonCalendar,
    HistoricalEvent,
)


# -----------------------------------------------------------------------------
# 1. API Client Nested & Flat Pagination Tests
# -----------------------------------------------------------------------------

def test_nested_endpoint_pagination_fewer_races_than_page_size():
    """
    Verify pagination does NOT terminate prematurely when len(Races) < page_size
    but MRData.total counts nested driver results and more remain.
    """
    client = HistoricalAPIClient(cache=InMemoryTTLCache())

    # 250 total results across 6 races. Limit = 100.
    # Page 1: 2 races, 50 results each = 100 results. offset = 0.
    page1 = {
        "MRData": {
            "total": "250",
            "limit": "100",
            "offset": "0",
            "RaceTable": {
                "Races": [
                    {"season": "2016", "round": "1", "Results": [{"position": str(i)} for i in range(1, 51)]},
                    {"season": "2016", "round": "2", "Results": [{"position": str(i)} for i in range(1, 51)]},
                ]
            },
        }
    }
    # Page 2: 2 races, 50 results each = 100 results. offset = 100.
    page2 = {
        "MRData": {
            "total": "250",
            "limit": "100",
            "offset": "100",
            "RaceTable": {
                "Races": [
                    {"season": "2016", "round": "3", "Results": [{"position": str(i)} for i in range(1, 51)]},
                    {"season": "2016", "round": "4", "Results": [{"position": str(i)} for i in range(1, 51)]},
                ]
            },
        }
    }
    # Page 3: 2 races, 25 results each = 50 results. offset = 200.
    page3 = {
        "MRData": {
            "total": "250",
            "limit": "100",
            "offset": "200",
            "RaceTable": {
                "Races": [
                    {"season": "2016", "round": "5", "Results": [{"position": str(i)} for i in range(1, 26)]},
                    {"season": "2016", "round": "6", "Results": [{"position": str(i)} for i in range(1, 26)]},
                ]
            },
        }
    }

    requested_offsets = []

    def mock_get_json(endpoint, params=None, ttl_seconds=None, bypass_cache=False):
        offset = params.get("offset", 0)
        requested_offsets.append(offset)
        if offset == 0:
            return page1
        elif offset == 100:
            return page2
        elif offset == 200:
            return page3
        return {"MRData": {"total": "250", "limit": "100", "offset": str(offset), "RaceTable": {"Races": []}}}

    with patch.object(client, "get_json", side_effect=mock_get_json):
        races = client.get_all_pages(
            endpoint="2016/results",
            table_key="RaceTable",
            items_key="Races",
            page_size=100,
        )

    # Must have made 3 calls advancing by 100
    assert requested_offsets == [0, 100, 200]
    # All 6 races accumulated
    assert len(races) == 6
    total_results = sum(len(r.get("Results", [])) for r in races)
    assert total_results == 250


def test_flat_endpoint_pagination_remains_correct():
    """Verify flat endpoints (e.g. circuits, seasons) continue to paginate and terminate correctly."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())

    page1 = {"MRData": {"total": "50", "limit": "25", "offset": "0", "CircuitTable": {"Circuits": [{"id": f"c{i}"} for i in range(1, 26)]}}}
    page2 = {"MRData": {"total": "50", "limit": "25", "offset": "25", "CircuitTable": {"Circuits": [{"id": f"c{i}"} for i in range(26, 51)]}}}

    offsets = []

    def mock_get_json(endpoint, params=None, **kwargs):
        off = params.get("offset", 0)
        offsets.append(off)
        return page1 if off == 0 else page2

    with patch.object(client, "get_json", side_effect=mock_get_json):
        circuits = client.get_all_pages("circuits", "CircuitTable", "Circuits", page_size=25)

    assert offsets == [0, 25]
    assert len(circuits) == 50
    assert circuits[0]["id"] == "c1"
    assert circuits[-1]["id"] == "c50"


def test_api_failure_in_middle_of_pagination_does_not_return_partial_dataset():
    """Verify that an API error during pagination raises HistoricalAPIError and does NOT return partial data."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())

    page1 = {
        "MRData": {
            "total": "200",
            "limit": "100",
            "offset": "0",
            "RaceTable": {"Races": [{"season": "2023", "round": "1"}]},
        }
    }

    def mock_get_json(endpoint, params=None, **kwargs):
        off = params.get("offset", 0)
        if off == 0:
            return page1
        raise HistoricalAPIError("Network timeout on page 2")

    with patch.object(client, "get_json", side_effect=mock_get_json):
        with pytest.raises(HistoricalAPIError):
            client.get_all_pages("2023/results", "RaceTable", "Races", page_size=100)


def test_driver_specific_pagination_still_works():
    """Verify that driver career pagination (1 race per driver) continues to paginate across multiple pages."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())

    page1 = {
        "MRData": {
            "total": "150",
            "limit": "100",
            "offset": "0",
            "RaceTable": {"Races": [{"round": str(i)} for i in range(1, 101)]},
        }
    }
    page2 = {
        "MRData": {
            "total": "150",
            "limit": "100",
            "offset": "100",
            "RaceTable": {"Races": [{"round": str(i)} for i in range(101, 151)]},
        }
    }

    with patch.object(client, "get_json", side_effect=[page1, page2]):
        all_races = client.get_all_pages("drivers/alonso/results", "RaceTable", "Races", page_size=100)

    assert len(all_races) == 150
    assert all_races[0]["round"] == "1"
    assert all_races[-1]["round"] == "150"


# -----------------------------------------------------------------------------
# 2. Split Round Merging Tests (Classification & Grid)
# -----------------------------------------------------------------------------

def test_split_classification_rounds_merged_not_overwritten():
    """
    Verify that when an Ergast page boundary splits a round across multiple pages,
    HistoricalClassificationService merges all records rather than overwriting.
    """
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = HistoricalClassificationService(api_client=client)

    # Page 1 contains Round 5 positions 1-12.
    # Page 2 contains Round 5 positions 13-22.
    mock_races = [
        {
            "season": "2016",
            "round": "5",
            "Results": [
                {
                    "position": str(i),
                    "positionText": str(i),
                    "points": "25" if i == 1 else "0",
                    "status": "Finished",
                    "Driver": {"driverId": f"driver_{i}", "givenName": "Driver", "familyName": str(i)},
                    "Constructor": {"constructorId": "team_a", "name": "Team A"},
                    "grid": str(i),
                    "laps": "66",
                }
                for i in range(1, 13)
            ],
        },
        {
            "season": "2016",
            "round": "5",
            "Results": [
                {
                    "position": str(i),
                    "positionText": str(i),
                    "points": "0",
                    "status": "Finished" if i < 21 else "Collision",
                    "Driver": {"driverId": f"driver_{i}", "givenName": "Driver", "familyName": str(i)},
                    "Constructor": {"constructorId": "team_a", "name": "Team A"},
                    "grid": str(i),
                    "laps": "65" if i < 21 else "0",
                }
                for i in range(13, 23)
            ],
        },
    ]

    with patch.object(client, "get_all_pages", return_value=mock_races):
        season_res = service.get_season_classifications(2016)

    assert 5 in season_res
    round_5_entries = season_res[5]
    # Must contain all 22 entries, not only the 10 from the second page!
    assert len(round_5_entries) == 22
    assert round_5_entries[0].driver_id == "driver_1"
    assert round_5_entries[0].position == 1
    assert round_5_entries[-1].driver_id == "driver_22"
    assert round_5_entries[-1].position == 22


def test_split_starting_grids_merged_not_overwritten():
    """Verify that split starting grids across page boundaries are safely merged."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = HistoricalGridService(api_client=client)

    mock_races = [
        {
            "season": "2016",
            "round": "5",
            "Results": [
                {"grid": str(i), "Driver": {"driverId": f"d_{i}", "givenName": "D", "familyName": str(i)}, "Constructor": {"name": "Team"}}
                for i in range(1, 13)
            ],
        },
        {
            "season": "2016",
            "round": "5",
            "Results": [
                {"grid": str(i), "Driver": {"driverId": f"d_{i}", "givenName": "D", "familyName": str(i)}, "Constructor": {"name": "Team"}}
                for i in range(13, 23)
            ],
        },
    ]

    with patch.object(client, "get_all_pages", return_value=mock_races):
        grids = service.get_season_starting_grids(2016)

    assert 5 in grids
    assert len(grids[5]) == 22
    assert grids[5][0].driver_id == "d_1"
    assert grids[5][-1].driver_id == "d_22"


def test_classification_and_grid_deduplication_on_repeated_records():
    """Verify that repeated records from overlapping page responses do not create duplicate entries."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = HistoricalClassificationService(api_client=client)

    # Round 1 repeated with overlapping driver_2
    mock_races = [
        {
            "season": "2024",
            "round": "1",
            "Results": [
                {"position": "1", "positionText": "1", "points": "25", "status": "Finished", "Driver": {"driverId": "d1"}, "Constructor": {"constructorId": "c1"}},
                {"position": "2", "positionText": "2", "points": "18", "status": "Finished", "Driver": {"driverId": "d2"}, "Constructor": {"constructorId": "c1"}},
            ],
        },
        {
            "season": "2024",
            "round": "1",
            "Results": [
                {"position": "2", "positionText": "2", "points": "18", "status": "Finished", "Driver": {"driverId": "d2"}, "Constructor": {"constructorId": "c1"}},  # duplicate
                {"position": "3", "positionText": "3", "points": "15", "status": "Finished", "Driver": {"driverId": "d3"}, "Constructor": {"constructorId": "c1"}},
            ],
        },
    ]

    with patch.object(client, "get_all_pages", return_value=mock_races):
        res = service.get_season_classifications(2024)

    assert len(res[1]) == 3
    assert [e.driver_id for e in res[1]] == ["d1", "d2", "d3"]


# -----------------------------------------------------------------------------
# 3. Teammate H2H Completeness (Hamilton vs Rosberg 2016)
# -----------------------------------------------------------------------------

def test_hamilton_vs_rosberg_2016_full_season_aggregation():
    """Verify that 2016 Hamilton vs Rosberg aggregates all 21 rounds when complete data is retrieved."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    class_svc = HistoricalClassificationService(api_client=client)
    qual_svc = HistoricalQualifyingService(api_client=client)
    teammate_svc = TeammateAnalyticsService(
        api_client=client,
        classification_service=class_svc,
        qualifying_service=qual_svc,
    )

    # Synthesize all 21 rounds for 2016
    mock_classifications = {}
    mock_qualifying = {}

    for rnd in range(1, 22):
        # Round 5 is Spain where both crashed
        if rnd == 5:
            r1 = RaceResultEntry(position=21, position_text="R", driver_id="hamilton", driver_name="Lewis Hamilton", driver_number="44", constructor_id="mercedes", constructor_name="Mercedes", grid_position=1, raw_grid_value="1", positions_gained=None, race_time_or_gap=None, laps_completed=0, points=0.0, status="Collision")
            r2 = RaceResultEntry(position=22, position_text="R", driver_id="rosberg", driver_name="Nico Rosberg", driver_number="6", constructor_id="mercedes", constructor_name="Mercedes", grid_position=2, raw_grid_value="2", positions_gained=None, race_time_or_gap=None, laps_completed=0, points=0.0, status="Collision")
        elif rnd % 2 == 1:
            r1 = RaceResultEntry(position=1, position_text="1", driver_id="hamilton", driver_name="Lewis Hamilton", driver_number="44", constructor_id="mercedes", constructor_name="Mercedes", grid_position=1, raw_grid_value="1", positions_gained=0, race_time_or_gap=None, laps_completed=55, points=25.0, status="Finished")
            r2 = RaceResultEntry(position=2, position_text="2", driver_id="rosberg", driver_name="Nico Rosberg", driver_number="6", constructor_id="mercedes", constructor_name="Mercedes", grid_position=2, raw_grid_value="2", positions_gained=0, race_time_or_gap=None, laps_completed=55, points=18.0, status="Finished")
        else:
            r1 = RaceResultEntry(position=2, position_text="2", driver_id="hamilton", driver_name="Lewis Hamilton", driver_number="44", constructor_id="mercedes", constructor_name="Mercedes", grid_position=2, raw_grid_value="2", positions_gained=0, race_time_or_gap=None, laps_completed=55, points=18.0, status="Finished")
            r2 = RaceResultEntry(position=1, position_text="1", driver_id="rosberg", driver_name="Nico Rosberg", driver_number="6", constructor_id="mercedes", constructor_name="Mercedes", grid_position=1, raw_grid_value="1", positions_gained=0, race_time_or_gap=None, laps_completed=55, points=25.0, status="Finished")

        mock_classifications[rnd] = [r1, r2]

    with patch.object(class_svc, "get_season_classifications", return_value=mock_classifications):
        with patch.object(qual_svc, "get_season_qualifying", return_value={}):
            h2h = teammate_svc.compare_teammates("hamilton", "rosberg", season=2016)

    assert h2h is not None
    assert h2h.races_cohabited == 21
    # Both finished 20 rounds (excluding Round 5 collision)
    assert h2h.both_finished_count == 20


# -----------------------------------------------------------------------------
# 4. Circuit Analytics Optimization & Reconciliation Tests
# -----------------------------------------------------------------------------

def test_circuit_analytics_monaco_full_history_and_reconciliation():
    """
    Verify Monaco circuit history uses races and results/1 endpoints,
    reconciles events by (season, round), and excludes future/scheduled events without winners.
    """
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = CircuitAnalyticsService(api_client=client)

    # 72 historical races (1950 to 2025) + 1 future scheduled 2026 race
    mock_races = [
        {"season": str(y), "round": "6", "raceName": "Monaco Grand Prix"}
        for y in range(1950, 2026)
    ]
    mock_races.append({"season": "2026", "round": "6", "raceName": "Monaco Grand Prix"})  # Scheduled future race

    # Historical winners for 1950-2025 (2026 has no winner)
    mock_winners = [
        {
            "season": str(y),
            "round": "6",
            "Results": [
                {
                    "position": "1",
                    "positionText": "1",
                    "Driver": {"driverId": "senna" if y < 1994 else "leclerc", "givenName": "Ayrton" if y < 1994 else "Charles", "familyName": "Senna" if y < 1994 else "Leclerc"},
                    "Constructor": {"constructorId": "mclaren" if y < 1994 else "ferrari", "name": "McLaren" if y < 1994 else "Ferrari"},
                }
            ],
        }
        for y in range(1950, 2026)
    ]

    mock_circuit = HistoricalCircuit(circuit_id="monaco", name="Circuit de Monaco", locality="Monte Carlo", country="Monaco")

    def mock_get_all_pages(endpoint, **kwargs):
        if "results/1" in endpoint:
            return mock_winners
        elif "races" in endpoint:
            return mock_races
        return []

    with patch.object(service.circuits_service, "get_circuit", return_value=mock_circuit):
        with patch.object(client, "get_all_pages", side_effect=mock_get_all_pages):
            c_hist = service.get_circuit_history("monaco")

    assert c_hist is not None
    assert c_hist.circuit_name == "Circuit de Monaco"
    # Reconciled completed races = 76 (1950..2025 = 76 events). Future 2026 race is NOT counted in completed results.
    assert c_hist.total_races_hosted == len(mock_winners)
    assert c_hist.first_gp_year == 1950
    assert c_hist.last_gp_year == 2025
    assert len(c_hist.recent_winners) == 5
    assert c_hist.recent_winners[0][0] == 2025


# -----------------------------------------------------------------------------
# 5. Season 2023 Completed Championship Integrity Tests
# -----------------------------------------------------------------------------

def test_season_analytics_2023_all_22_rounds_completed():
    """Verify Season 2023 retrieves all 22 rounds and is correctly marked Completed Championship."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    events_svc = HistoricalEventsService(api_client=client)
    class_svc = HistoricalClassificationService(api_client=client)
    season_svc = SeasonAnalyticsService(
        api_client=client,
        events_service=events_svc,
        classification_service=class_svc,
    )

    mock_cal = SeasonCalendar(
        year=2023,
        total_rounds=22,
        completed_rounds=22,
        upcoming_rounds=0,
        past_scheduled_rounds=22,
        is_in_progress=False,
    )

    mock_classifications = {
        rnd: [
            RaceResultEntry(
                position=1,
                position_text="1",
                driver_id="verstappen",
                driver_name="Max Verstappen",
                driver_number="1",
                constructor_id="red_bull",
                constructor_name="Red Bull",
                grid_position=1,
                raw_grid_value="1",
                positions_gained=0,
                race_time_or_gap=None,
                laps_completed=57,
                points=25.0,
                status="Finished",
            )
        ]
        for rnd in range(1, 23)
    }

    with patch.object(events_svc, "get_season_calendar", return_value=mock_cal):
        with patch.object(class_svc, "get_season_classifications", return_value=mock_classifications):
            summary = season_svc.get_season_summary(2023)

    assert summary is not None
    assert summary.completed_rounds == 22
    assert summary.total_scheduled_rounds == 22
    assert summary.is_in_progress is False
    assert "Completed Championship: 22 Rounds" in summary.status_display
