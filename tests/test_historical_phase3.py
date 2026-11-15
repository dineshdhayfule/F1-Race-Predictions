"""
Unit tests for ApexGrid Historical Analytics Subsystem — Phase 3
================================================================
Tests HistoricalQualifyingService, HistoricalGridService, and HistoricalClassificationService.
Enforces:
- Separation of qualifying classification, starting grids, and race results.
- Session format handling (1994 single Q1 session vs 2006+ knockout).
- Raw grid value preservation (grid == '0' preserved as unclassified/non-standard).
- Fractional points (8.14 pts) and shared-drive record preservation.
- Grand Prix classification strictly excludes Sprint sessions.
- Zero filesystem persistence.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.historical.models import (
    QualifyingResultEntry,
    StartingGridResultEntry,
    RaceResultEntry,
)
from src.historical.in_memory_cache import InMemoryTTLCache
from src.historical.api_client import HistoricalAPIClient
from src.historical.qualifying_service import HistoricalQualifyingService
from src.historical.grid_service import HistoricalGridService
from src.historical.classification_service import HistoricalClassificationService


# -----------------------------------------------------------------------------
# Mock Data Fixtures
# -----------------------------------------------------------------------------

MOCK_2023_QUALIFYING_RESPONSE = {
    "MRData": {
        "total": "3",
        "RaceTable": {
            "season": "2023",
            "round": "1",
            "Races": [
                {
                    "season": "2023",
                    "round": "1",
                    "raceName": "Bahrain Grand Prix",
                    "QualifyingResults": [
                        {
                            "number": "1",
                            "position": "1",
                            "Driver": {"driverId": "max_verstappen", "code": "VER", "givenName": "Max", "familyName": "Verstappen"},
                            "Constructor": {"constructorId": "red_bull", "name": "Red Bull"},
                            "Q1": "1:31.295",
                            "Q2": "1:30.503",
                            "Q3": "1:29.708"
                        },
                        {
                            "number": "11",
                            "position": "2",
                            "Driver": {"driverId": "perez", "code": "PER", "givenName": "Sergio", "familyName": "Perez"},
                            "Constructor": {"constructorId": "red_bull", "name": "Red Bull"},
                            "Q1": "1:31.479",
                            "Q2": "1:30.746",
                            "Q3": "1:29.846"
                        },
                        {
                            "number": "2",
                            "position": "16",
                            "Driver": {"driverId": "sargeant", "code": "SAR", "givenName": "Logan", "familyName": "Sargeant"},
                            "Constructor": {"constructorId": "williams", "name": "Williams"},
                            "Q1": "1:31.652"
                            # Q2 and Q3 missing due to Q1 elimination
                        }
                    ]
                }
            ]
        }
    }
}

MOCK_1994_QUALIFYING_RESPONSE = {
    "MRData": {
        "total": "2",
        "RaceTable": {
            "season": "1994",
            "round": "1",
            "Races": [
                {
                    "season": "1994",
                    "round": "1",
                    "raceName": "Brazilian Grand Prix",
                    "QualifyingResults": [
                        {
                            "number": "2",
                            "position": "1",
                            "Driver": {"driverId": "senna", "givenName": "Ayrton", "familyName": "Senna"},
                            "Constructor": {"constructorId": "williams", "name": "Williams"},
                            "Q1": "1:15.962"
                            # 1994 format: single best time under Q1; no Q2 or Q3
                        },
                        {
                            "number": "5",
                            "position": "2",
                            "Driver": {"driverId": "michael_schumacher", "givenName": "Michael", "familyName": "Schumacher"},
                            "Constructor": {"constructorId": "benetton", "name": "Benetton"},
                            "Q1": "1:16.290"
                        }
                    ]
                }
            ]
        }
    }
}

MOCK_RACE_RESULTS_WITH_GRID_AND_SHARED_DRIVES = {
    "MRData": {
        "total": "5",
        "RaceTable": {
            "season": "1954",
            "round": "5",
            "Races": [
                {
                    "season": "1954",
                    "round": "5",
                    "raceName": "British Grand Prix",
                    "Results": [
                        {
                            "number": "12",
                            "position": "1",
                            "positionText": "1",
                            "points": "8.14",  # Fractional point split 7 ways!
                            "Driver": {"driverId": "gonzalez", "givenName": "Jose Froilan", "familyName": "Gonzalez"},
                            "Constructor": {"constructorId": "ferrari", "name": "Ferrari"},
                            "grid": "2",
                            "laps": "90",
                            "status": "Finished",
                            "Time": {"time": "2:56:14"}
                        },
                        {
                            # Shared drive row 1
                            "number": "1",
                            "position": "2",
                            "positionText": "2",
                            "points": "3.0",
                            "Driver": {"driverId": "hawthorn", "givenName": "Mike", "familyName": "Hawthorn"},
                            "Constructor": {"constructorId": "ferrari", "name": "Ferrari"},
                            "grid": "3",
                            "laps": "90",
                            "status": "Finished"
                        },
                        {
                            # Driver starting from non-standard grid slot (grid == '0')
                            "number": "20",
                            "position": "10",
                            "positionText": "10",
                            "points": "0",
                            "Driver": {"driverId": "reserve_driver", "givenName": "Test", "familyName": "Driver"},
                            "Constructor": {"constructorId": "maserati", "name": "Maserati"},
                            "grid": "0",  # Value 0 in provider data
                            "laps": "80",
                            "status": "+10 Laps"
                        },
                        {
                            # Retired driver with mechanical failure
                            "number": "5",
                            "position": "15",
                            "positionText": "R",
                            "points": "0",
                            "Driver": {"driverId": "ascari", "givenName": "Alberto", "familyName": "Ascari"},
                            "Constructor": {"constructorId": "maserati", "name": "Maserati"},
                            "grid": "1",
                            "laps": "21",
                            "status": "Engine"
                        },
                        {
                            # Driver with missing grid value
                            "number": "99",
                            "position": "16",
                            "positionText": "W",
                            "points": "0",
                            "Driver": {"driverId": "withdrawn_driver", "givenName": "Withdrawn", "familyName": "Driver"},
                            "Constructor": {"constructorId": "cooper", "name": "Cooper"},
                            "grid": None,  # Missing grid position
                            "laps": "0",
                            "status": "Withdrew"
                        }
                    ]
                }
            ]
        }
    }
}


# -----------------------------------------------------------------------------
# 1. Historical Qualifying Service Tests
# -----------------------------------------------------------------------------

def test_qualifying_service_modern_knockout():
    """Verify modern 2023 knockout qualifying with Q1, Q2, Q3, and Q1-eliminated drivers."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = HistoricalQualifyingService(api_client=client)

    with patch.object(client, "get_json", return_value=MOCK_2023_QUALIFYING_RESPONSE):
        q_results = service.get_round_qualifying(2023, round_num=1)
        assert len(q_results) == 3

        # P1: Pole sitter
        p1 = q_results[0]
        assert p1.position == 1
        assert p1.driver_name == "Max Verstappen"
        assert p1.driver_code == "VER"
        assert p1.q1 == "1:31.295"
        assert p1.q2 == "1:30.503"
        assert p1.q3 == "1:29.708"
        assert p1.best_lap_time == "1:29.708"
        assert p1.gap_to_pole is None  # Pole sitter has no gap
        assert p1.session_format == "Knockout"

        # P2: Gap to pole calculated correctly (+0.138s)
        p2 = q_results[1]
        assert p2.position == 2
        assert p2.best_lap_time == "1:29.846"
        assert p2.gap_to_pole == "+0.138"

        # P16: Eliminated in Q1 (q2 and q3 must be None, never fabricated)
        p16 = q_results[2]
        assert p16.position == 16
        assert p16.q1 == "1:31.652"
        assert p16.q2 is None
        assert p16.q3 is None
        assert p16.best_lap_time == "1:31.652"


def test_qualifying_service_historical_1994_single_session():
    """Verify 1994 historical format with single Q1 session and no Q2/Q3."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = HistoricalQualifyingService(api_client=client)

    with patch.object(client, "get_json", return_value=MOCK_1994_QUALIFYING_RESPONSE):
        q_results = service.get_round_qualifying(1994, round_num=1)
        assert len(q_results) == 2

        p1 = q_results[0]
        assert p1.position == 1
        assert p1.driver_name == "Ayrton Senna"
        assert p1.q1 == "1:15.962"
        assert p1.q2 is None  # Format has no Q2/Q3
        assert p1.q3 is None
        assert "Qualifying Best" in p1.session_format
        assert p1.session_code == "QB"
        assert p1.best_lap_time == "1:15.962"

        p2 = q_results[1]
        assert p2.position == 2
        assert p2.gap_to_pole == "+0.328"


def test_qualifying_alpha_historical_session_formats_qa_and_qo():
    """Verify Alpha Results historical formats (QA aggregate and QO shootout) preserve original session types."""
    # 1. 2005 QA (Qualifying Aggregate)
    alpha_qa_item = {
        "position": 1,
        "time": "3:01.460",
        "driver": {"id": "driver_fisichella", "given_name": "Giancarlo", "family_name": "Fisichella", "abbreviation": "FIS"},
        "team": {"id": "team_renault", "name": "Renault"},
        "components": {
            "Q1": {"time": "1:33.171"},
            "Q2": {"time": "1:28.289"}
        }
    }

    q_qa = HistoricalQualifyingService.parse_alpha_qualifying_result(alpha_qa_item, session_type="QA")
    assert q_qa is not None
    assert q_qa.position == 1
    assert q_qa.driver_name == "Giancarlo Fisichella"
    assert q_qa.session_code == "QA"
    assert "Qualifying Aggregate" in q_qa.session_format
    assert q_qa.best_lap_time == "3:01.460"
    assert q_qa.q1 == "1:33.171"
    assert q_qa.q2 == "1:28.289"

    # 2. 2003 QO (Qualifying Shootout Order)
    alpha_qo_item = {
        "position": 1,
        "time": "1:27.173",
        "driver": {"id": "driver_schumacher", "given_name": "Michael", "family_name": "Schumacher", "abbreviation": "MSC"},
        "team": {"id": "team_ferrari", "name": "Ferrari"},
        "components": {"Q1": {"time": "1:27.173"}}
    }

    q_qo = HistoricalQualifyingService.parse_alpha_qualifying_result(alpha_qo_item, session_type="QO")
    assert q_qo is not None
    assert q_qo.session_code == "QO"
    assert "Qualifying Shootout Order" in q_qo.session_format
    assert q_qo.best_lap_time == "1:27.173"


def test_qualifying_missing_metadata_and_missing_lap_times():
    """Verify handling when driver/constructor names or lap times are missing."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = HistoricalQualifyingService(api_client=client)

    sparse_qualifying_resp = {
        "MRData": {
            "total": "2",
            "RaceTable": {
                "season": "2023",
                "round": "1",
                "Races": [
                    {
                        "QualifyingResults": [
                            {
                                "position": "1",
                                "Driver": {"driverId": "sparse_driver"},  # givenName & familyName missing
                                "Constructor": {"constructorId": "sparse_team"},  # name missing
                                "Q1": "1:30.000"
                            },
                            {
                                "position": "20",
                                "Driver": {"driverId": "no_time_driver", "givenName": "No", "familyName": "Time"},
                                "Constructor": {"constructorId": "some_team", "name": "Some Team"}
                                # Q1, Q2, Q3 all missing (DNS / DNQ)
                            }
                        ]
                    }
                ]
            }
        }
    }

    with patch.object(client, "get_json", return_value=sparse_qualifying_resp):
        res = service.get_round_qualifying(2023, 1)
        assert len(res) == 2

        # P1: Name falls back gracefully to driverId
        assert res[0].driver_name == "sparse_driver"
        assert res[0].constructor_name == "sparse_team"
        assert res[0].best_lap_time == "1:30.000"

        # P20: Driver with no time set
        assert res[1].position == 20
        assert res[1].best_lap_time is None
        assert res[1].q1 is None
        assert res[1].q2 is None
        assert res[1].q3 is None


def test_qualifying_service_empty_pre_1994_response():
    """Verify empty response for pre-1994 returns empty list without error."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = HistoricalQualifyingService(api_client=client)

    empty_resp = {"MRData": {"total": "0", "RaceTable": {"season": "1970", "Races": []}}}

    with patch.object(client, "get_json", return_value=empty_resp):
        q_results = service.get_round_qualifying(1970, round_num=1)
        assert q_results == []


# -----------------------------------------------------------------------------
# 2. Historical Starting Grid Service Tests
# -----------------------------------------------------------------------------

def test_grid_service_preserves_raw_grid_zero_and_nullable_positions():
    """Verify raw grid == '0' is classified as non-standard/unclassified and not pit-lane."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = HistoricalGridService(api_client=client)

    with patch.object(client, "get_json", return_value=MOCK_RACE_RESULTS_WITH_GRID_AND_SHARED_DRIVES):
        grid = service.get_starting_grid(1954, round_num=5)
        assert len(grid) == 5

        # P1 on grid (Ascari started P1)
        g1 = grid[0]
        assert g1.grid_position == 1
        assert g1.raw_grid_value == "1"
        assert g1.qualifying_position is None  # CRITICAL: Never inferred from results
        assert g1.driver_id == "ascari"
        assert "Grid P1" in g1.grid_status

        # Car with raw grid == '0'
        g_zero = [e for e in grid if e.raw_grid_value == "0"][0]
        assert g_zero.grid_position is None
        assert g_zero.raw_grid_value == "0"
        # CRITICAL: Must NOT claim pit lane!
        assert "pit lane" not in g_zero.grid_status.lower()
        assert "Unclassified / Non-standard" in g_zero.grid_status

        # Car with missing grid (grid == None)
        g_none = [e for e in grid if e.driver_id == "withdrawn_driver"][0]
        assert g_none.grid_position is None
        assert g_none.raw_grid_value == ""
        assert "unrecorded" in g_none.grid_status.lower()


# -----------------------------------------------------------------------------
# 3. Historical Race Classification Service Tests
# -----------------------------------------------------------------------------

def test_classification_service_fractional_points_and_shared_drives():
    """Verify exact float precision for fractional points (8.14) and shared drive preservation."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = HistoricalClassificationService(api_client=client)

    with patch.object(client, "get_json", return_value=MOCK_RACE_RESULTS_WITH_GRID_AND_SHARED_DRIVES):
        results = service.get_race_classification(1954, round_num=5)
        assert len(results) == 5

        # P1: Gonzalez with 8.14 fractional points
        r1 = results[0]
        assert r1.position == 1
        assert r1.position_text == "1"
        assert r1.points == 8.14  # CRITICAL: Never cast to int or rounded!
        assert isinstance(r1.points, float)
        assert r1.grid_position == 2
        assert r1.positions_gained == 1
        assert r1.status == "Finished"

        # P15: Retired driver with status "Engine"
        r_retired = [r for r in results if r.driver_id == "ascari"][0]
        assert r_retired.position == 15
        assert r_retired.position_text == "R"
        assert r_retired.status == "Engine"
        assert r_retired.points == 0.0
        # Position gained should be safely computed if both grid and pos are positive ints:
        # Ascari started P1, retired classified P15 -> positions_gained = -14
        assert r_retired.positions_gained == -14

        # Non-standard grid driver (grid == 0) -> positions_gained must be None
        r_grid_zero = [r for r in results if r.driver_id == "reserve_driver"][0]
        assert r_grid_zero.grid_position is None
        assert r_grid_zero.positions_gained is None  # CRITICAL: Never compute gains from grid 0!


def test_classification_service_excludes_sprint_sessions():
    """Verify Grand Prix classifications parse only official 'Results' and strictly exclude 'SprintResults'."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = HistoricalClassificationService(api_client=client)

    # Mock response containing both Results and SprintResults
    mock_dual_response = {
        "MRData": {
            "total": "1",
            "RaceTable": {
                "season": "2023",
                "round": "4",
                "Races": [
                    {
                        "season": "2023",
                        "round": "4",
                        "raceName": "Azerbaijan Grand Prix",
                        "Results": [
                            {
                                "number": "11",
                                "position": "1",
                                "positionText": "1",
                                "points": "25",
                                "Driver": {"driverId": "perez", "givenName": "Sergio", "familyName": "Perez"},
                                "Constructor": {"constructorId": "red_bull", "name": "Red Bull"},
                                "grid": "3",
                                "laps": "51",
                                "status": "Finished"
                            }
                        ],
                        "SprintResults": [
                            {
                                "number": "11",
                                "position": "1",
                                "positionText": "1",
                                "points": "8",  # Sprint points
                                "Driver": {"driverId": "perez", "givenName": "Sergio", "familyName": "Perez"},
                                "Constructor": {"constructorId": "red_bull", "name": "Red Bull"},
                                "grid": "2",
                                "laps": "17",
                                "status": "Finished"
                            }
                        ]
                    }
                ]
            }
        }
    }

    with patch.object(client, "get_json", return_value=mock_dual_response):
        gp_results = service.get_race_classification(2023, round_num=4)
        assert len(gp_results) == 1
        # Confirms only the Grand Prix race was parsed (51 laps, 25 points), NOT the sprint (17 laps, 8 points)
        assert gp_results[0].laps_completed == 51
        assert gp_results[0].points == 25.0


def test_is_race_confirmed_completed():
    """Verify is_race_confirmed_completed returns True when official results exist, False when empty."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = HistoricalClassificationService(api_client=client)

    with patch.object(client, "get_json", return_value=MOCK_RACE_RESULTS_WITH_GRID_AND_SHARED_DRIVES):
        assert service.is_race_confirmed_completed(1954, 5) is True

    with patch.object(client, "get_json", return_value={"MRData": {"RaceTable": {"Races": []}}}):
        assert service.is_race_confirmed_completed(2026, 24) is False


# -----------------------------------------------------------------------------
# 4. Zero Filesystem Persistence Assertion
# -----------------------------------------------------------------------------

def test_phase3_zero_disk_persistence():
    """Assert zero filesystem writes occur during Phase 3 qualifying, grid, and classification operations."""
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
        q_service = HistoricalQualifyingService(api_client=client)
        grid_service = HistoricalGridService(api_client=client)
        class_service = HistoricalClassificationService(api_client=client)

        with patch.object(client, "get_json", return_value=MOCK_2023_QUALIFYING_RESPONSE):
            _ = q_service.get_round_qualifying(2023, 1)

        with patch.object(client, "get_json", return_value=MOCK_RACE_RESULTS_WITH_GRID_AND_SHARED_DRIVES):
            _ = grid_service.get_starting_grid(1954, 5)
            _ = class_service.get_race_classification(1954, 5)

    assert len(write_calls) == 0, f"Detected illegal disk write operations: {write_calls}"


# -----------------------------------------------------------------------------
# 5. Edge Cases & Infrastructure Verification Tests
# -----------------------------------------------------------------------------

def test_grid_service_non_standard_and_missing_values():
    """Verify that non-numeric (PL, PIT), negative (-1), empty string, and None grid values are handled safely."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = HistoricalGridService(api_client=client)

    mock_edge_grid_resp = {
        "MRData": {
            "total": "4",
            "RaceTable": {
                "season": "2022",
                "round": "10",
                "Races": [
                    {
                        "Results": [
                            {"grid": "PL", "Driver": {"driverId": "driver_pl"}},
                            {"grid": "PIT", "Driver": {"driverId": "driver_pit"}},
                            {"grid": "-1", "Driver": {"driverId": "driver_neg"}},
                            {"grid": "", "Driver": {"driverId": "driver_empty"}},
                        ]
                    }
                ]
            }
        }
    }

    with patch.object(client, "get_json", return_value=mock_edge_grid_resp):
        entries = service.get_starting_grid(2022, 10)
        assert len(entries) == 4

        # "PL"
        assert entries[0].grid_position is None
        assert entries[0].raw_grid_value == "PL"
        assert "Non-numeric" in entries[0].grid_status

        # "PIT"
        assert entries[1].grid_position is None
        assert entries[1].raw_grid_value == "PIT"
        assert "Non-numeric" in entries[1].grid_status

        # "-1"
        assert entries[2].grid_position is None
        assert entries[2].raw_grid_value == "-1"
        assert "Non-standard" in entries[2].grid_status

        # ""
        assert entries[3].grid_position is None
        assert entries[3].raw_grid_value == ""
        assert "unrecorded" in entries[3].grid_status.lower()


def test_classification_missing_position_and_fastest_lap():
    """Verify classification entries with missing numerical position and presence/absence of fastest lap data."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = HistoricalClassificationService(api_client=client)

    mock_results = {
        "MRData": {
            "total": "2",
            "RaceTable": {
                "season": "2021",
                "round": "1",
                "Races": [
                    {
                        "Results": [
                            {
                                "position": "1",
                                "positionText": "1",
                                "points": "25",
                                "grid": "1",
                                "Driver": {"driverId": "hamilton", "givenName": "Lewis", "familyName": "Hamilton"},
                                "Constructor": {"constructorId": "mercedes", "name": "Mercedes"},
                                "FastestLap": {
                                    "rank": "1",
                                    "lap": "44",
                                    "Time": {"time": "1:32.090"}
                                }
                            },
                            {
                                "position": None,  # Missing position integer
                                "positionText": "D",  # Disqualified
                                "points": "0",
                                "grid": "5",
                                "Driver": {"driverId": "disqualified_driver", "givenName": "Disq", "familyName": "Driver"},
                                "Constructor": {"constructorId": "some_team", "name": "Some Team"},
                                "status": "Disqualified"
                                # FastestLap is omitted
                            }
                        ]
                    }
                ]
            }
        }
    }

    with patch.object(client, "get_json", return_value=mock_results):
        classified = service.get_race_classification(2021, 1)
        assert len(classified) == 2

        # P1: Hamilton with fastest lap data
        h = classified[0]
        assert h.position == 1
        assert h.fastest_lap_rank == 1
        assert h.fastest_lap_time == "1:32.090"
        assert h.positions_gained == 0

        # Disqualified driver without numerical position
        d = classified[1]
        assert d.position is None
        assert d.position_text == "D"
        assert d.status == "Disqualified"
        assert d.fastest_lap_rank is None
        assert d.fastest_lap_time is None
        # Positions gained must be None if finish position is None
        assert d.positions_gained is None


def test_infrastructure_pagination_and_cache_reuse():
    """Verify multi-page pagination for season queries and in-memory cache reuse."""
    cache = InMemoryTTLCache(default_ttl_seconds=3600)
    client = HistoricalAPIClient(cache=cache)
    q_service = HistoricalQualifyingService(api_client=client)
    grid_service = HistoricalGridService(api_client=client)
    class_service = HistoricalClassificationService(api_client=client)

    # 1. Test cache reuse
    call_count = 0
    def mock_fetch(url, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return MOCK_2023_QUALIFYING_RESPONSE

    with patch.object(client, "_fetch_with_retries", side_effect=mock_fetch):
        # Call twice for identical resource
        res1 = q_service.get_round_qualifying(2023, 1)
        res2 = q_service.get_round_qualifying(2023, 1)

        assert len(res1) == 3
        assert len(res2) == 3
        # Cache hit prevents second API request
        assert call_count == 1

    # 2. Test multi-page season queries with get_all_pages
    mock_races = [
        {"round": "1", "Results": [{"grid": "1", "position": "1", "Driver": {"driverId": "v1"}}]},
        {"round": "2", "Results": [{"grid": "2", "position": "1", "Driver": {"driverId": "v2"}}]},
    ]
    with patch.object(client, "get_all_pages", return_value=mock_races):
        season_grids = grid_service.get_season_starting_grids(2023)
        assert len(season_grids) == 2
        assert 1 in season_grids
        assert 2 in season_grids
        assert season_grids[1][0].driver_id == "v1"

        season_class = class_service.get_season_classifications(2023)
        assert len(season_class) == 2
        assert 1 in season_class
        assert 2 in season_class
        assert season_class[2][0].driver_id == "v2"

