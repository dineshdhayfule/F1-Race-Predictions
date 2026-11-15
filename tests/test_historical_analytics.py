"""
Unit tests for ApexGrid Historical Analytics Subsystem — Phase 4
================================================================
Validates all historical analytics calculations across 1950–2026:
- Driver statistics: starts, finishes, wins, podiums, poles, points (float precision), DNFs, and avg finish.
- Teammate head-to-head: same-team enforcement, qualifying comparison, missing data preservation, and shared drives.
- Historical qualifying analytics: session format preservation (Knockout, QB, QO, QA), pre-1994 handling, and grid deltas.
- Race performance: positions gained/lost strictly computed when grid > 0 and finish > 0, DNF rates, and distributions.
- Circuit history: track records, historical winners, and podiums without synthetic metadata.
- Season summary: provisional 2026 handling, completed vs scheduled rounds, and standings.
- Zero filesystem persistence.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.historical.models import (
    HistoricalCircuit,
    HistoricalEvent,
    SeasonCalendar,
    QualifyingResultEntry,
    StartingGridResultEntry,
    RaceResultEntry,
)
from src.historical.in_memory_cache import InMemoryTTLCache
from src.historical.api_client import HistoricalAPIClient
from src.historical.analytics import (
    DriverAnalyticsService,
    TeammateAnalyticsService,
    QualifyingAnalyticsService,
    RacePerformanceAnalyticsService,
    CircuitAnalyticsService,
    SeasonAnalyticsService,
)


# -----------------------------------------------------------------------------
# Fixtures & Mock Data
# -----------------------------------------------------------------------------

MOCK_DRIVER_RESULTS_ALONSO = [
    {
        "season": "2005",
        "round": "1",
        "Results": [
            {
                "position": "3",
                "positionText": "3",
                "points": "6",
                "grid": "13",
                "status": "Finished",
                "Driver": {"driverId": "alonso", "givenName": "Fernando", "familyName": "Alonso", "nationality": "Spanish"},
                "Constructor": {"constructorId": "renault", "name": "Renault"},
            }
        ],
    },
    {
        "season": "2005",
        "round": "2",
        "Results": [
            {
                "position": "1",
                "positionText": "1",
                "points": "10",
                "grid": "1",
                "status": "Finished",
                "Driver": {"driverId": "alonso", "givenName": "Fernando", "familyName": "Alonso"},
                "Constructor": {"constructorId": "renault", "name": "Renault"},
            }
        ],
    },
    {
        "season": "2005",
        "round": "7",
        "Results": [
            {
                "position": "18",
                "positionText": "R",
                "points": "0",
                "grid": "6",
                "status": "Suspension",
                "Driver": {"driverId": "alonso", "givenName": "Fernando", "familyName": "Alonso"},
                "Constructor": {"constructorId": "renault", "name": "Renault"},
            }
        ],
    },
]

MOCK_1954_SHARED_DRIVE_RESULTS = [
    {
        "season": "1954",
        "round": "5",
        "Results": [
            {
                "position": "1",
                "positionText": "1",
                "points": "8.14",  # Fractional point split!
                "grid": "2",
                "status": "Finished",
                "Driver": {"driverId": "gonzalez", "givenName": "Jose Froilan", "familyName": "Gonzalez", "nationality": "Argentine"},
                "Constructor": {"constructorId": "ferrari", "name": "Ferrari"},
            },
            {
                # Shared drive co-driver
                "position": "1",
                "positionText": "1",
                "points": "0.14",
                "grid": "0",  # Non-standard grid value
                "status": "Finished",
                "Driver": {"driverId": "trintignant", "givenName": "Maurice", "familyName": "Trintignant", "nationality": "French"},
                "Constructor": {"constructorId": "ferrari", "name": "Ferrari"},
            },
        ],
    }
]


# -----------------------------------------------------------------------------
# 1. Driver Analytics Service Tests
# -----------------------------------------------------------------------------

def test_driver_analytics_career_metrics_and_dnf():
    """Verify starts, classified finishes, wins, podiums, points, and DNF categorization."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = DriverAnalyticsService(api_client=client)

    with patch.object(client, "get_all_pages", return_value=MOCK_DRIVER_RESULTS_ALONSO):
        with patch.object(service.qualifying_service, "get_season_qualifying", return_value={}):
            stats = service.get_driver_stats("alonso")
            assert stats is not None
            assert stats.driver_name == "Fernando Alonso"
            assert stats.nationality == "Spanish"
            assert stats.total_entries == 3
            assert stats.total_starts == 3
            assert stats.classified_finishes == 2
            assert stats.wins == 1
            assert stats.podiums == 2
            assert stats.total_points == 16.0
            assert stats.dnf_count == 1
            assert "Suspension" in stats.retirement_reasons
            assert stats.retirement_reasons["Suspension"] == 1
            # Average finish strictly among classified finishes: positions [3, 1] -> (3 + 1) / 2 = 2.0
            assert stats.avg_finish_position == 2.0
            assert len(stats.seasons_breakdown) == 1
            assert stats.seasons_breakdown[0].season == 2005


def test_driver_analytics_fractional_points_and_shared_drives():
    """Verify exact preservation of historical fractional points (8.14) and shared drives."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = DriverAnalyticsService(api_client=client)

    with patch.object(client, "get_all_pages", return_value=MOCK_1954_SHARED_DRIVE_RESULTS):
        stats = service.get_driver_stats("gonzalez")
        assert stats is not None
        assert stats.total_points == 8.14
        assert stats.wins == 1
        assert stats.qualifying_session_p1s is None  # Pre-1994 pole positions unavailable
        assert "pre-1994" in stats.poles_coverage_note.lower()


def test_driver_analytics_pre_1994_coverage_note():
    """Verify that pre-1994 drivers have explicit coverage limitations."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = DriverAnalyticsService(api_client=client)

    mock_fangio = [
        {
            "season": "1951",
            "round": "1",
            "Results": [
                {
                    "position": "1",
                    "positionText": "1",
                    "points": "9",
                    "grid": "1",
                    "status": "Finished",
                    "Driver": {"driverId": "fangio", "givenName": "Juan Manuel", "familyName": "Fangio"},
                    "Constructor": {"constructorId": "alfa", "name": "Alfa Romeo"},
                }
            ],
        }
    ]
    with patch.object(client, "get_all_pages", return_value=mock_fangio):
        stats = service.get_driver_stats("fangio")
        assert stats is not None
        assert stats.qualifying_session_p1s is None
        assert any("pre-1994" in note.lower() for note in stats.coverage_notes)


# -----------------------------------------------------------------------------
# 2. Teammate Analytics Service Tests
# -----------------------------------------------------------------------------

def test_teammate_comparison_same_team_enforcement():
    """Verify teammate comparison only evaluates rounds where drivers shared the same constructor."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = TeammateAnalyticsService(api_client=client)

    mock_season_classifications = {
        1: [
            RaceResultEntry(position=1, position_text="1", driver_id="hamilton", driver_name="Lewis Hamilton", driver_number="44", constructor_id="mercedes", constructor_name="Mercedes", grid_position=1, raw_grid_value="1", positions_gained=0, race_time_or_gap=None, laps_completed=57, points=25.0, status="Finished"),
            RaceResultEntry(position=2, position_text="2", driver_id="rosberg", driver_name="Nico Rosberg", driver_number="6", constructor_id="mercedes", constructor_name="Mercedes", grid_position=2, raw_grid_value="2", positions_gained=0, race_time_or_gap=None, laps_completed=57, points=18.0, status="Finished"),
            RaceResultEntry(position=3, position_text="3", driver_id="vettel", driver_name="Sebastian Vettel", driver_number="5", constructor_id="ferrari", constructor_name="Ferrari", grid_position=3, raw_grid_value="3", positions_gained=0, race_time_or_gap=None, laps_completed=57, points=15.0, status="Finished"),
        ],
        2: [
            RaceResultEntry(position=1, position_text="1", driver_id="rosberg", driver_name="Nico Rosberg", driver_number="6", constructor_id="mercedes", constructor_name="Mercedes", grid_position=1, raw_grid_value="1", positions_gained=0, race_time_or_gap=None, laps_completed=56, points=25.0, status="Finished"),
            RaceResultEntry(position=15, position_text="R", driver_id="hamilton", driver_name="Lewis Hamilton", driver_number="44", constructor_id="mercedes", constructor_name="Mercedes", grid_position=2, raw_grid_value="2", positions_gained=None, race_time_or_gap=None, laps_completed=10, points=0.0, status="Engine"),
        ],
    }

    mock_season_quali = {
        1: [
            QualifyingResultEntry(position=1, driver_id="hamilton", driver_name="Lewis Hamilton", driver_code="HAM", constructor_id="mercedes", constructor_name="Mercedes", q3="1:30.000"),
            QualifyingResultEntry(position=2, driver_id="rosberg", driver_name="Nico Rosberg", driver_code="ROS", constructor_id="mercedes", constructor_name="Mercedes", q3="1:30.150"),
        ],
        2: [
            QualifyingResultEntry(position=1, driver_id="rosberg", driver_name="Nico Rosberg", driver_code="ROS", constructor_id="mercedes", constructor_name="Mercedes", q3="1:32.000"),
            QualifyingResultEntry(position=2, driver_id="hamilton", driver_name="Lewis Hamilton", driver_code="HAM", constructor_id="mercedes", constructor_name="Mercedes", q3="1:32.200"),
        ],
    }

    with patch.object(service.classification_service, "get_season_classifications", return_value=mock_season_classifications):
        with patch.object(service.qualifying_service, "get_season_qualifying", return_value=mock_season_quali):
            h2h = service.compare_teammates("hamilton", "rosberg", season=2016)
            assert h2h is not None
            assert h2h.races_cohabited == 2
            assert h2h.race_h2h == (1, 1)  # Hamilton won R1, Rosberg won R2
            assert h2h.qualifying_h2h == (1, 1)  # Hamilton pole R1, Rosberg pole R2
            assert h2h.points_comparison == (25.0, 43.0)
            assert h2h.dnfs_comparison == (1, 0)

            # Negative test: Non-teammates return None
            non_team = service.compare_teammates("hamilton", "vettel", season=2016)
            assert non_team is None


def test_teammate_comparison_pre_1994_qualifying_note():
    """Verify that comparing pre-1994 teammates does not fabricate qualifying results or count them as losses."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = TeammateAnalyticsService(api_client=client)

    mock_1988 = {
        1: [
            RaceResultEntry(position=1, position_text="1", driver_id="prost", driver_name="Alain Prost", driver_number="11", constructor_id="mclaren", constructor_name="McLaren", grid_position=3, raw_grid_value="3", positions_gained=2, race_time_or_gap=None, laps_completed=60, points=9.0, status="Finished"),
            RaceResultEntry(position=10, position_text="D", driver_id="senna", driver_name="Ayrton Senna", driver_number="12", constructor_id="mclaren", constructor_name="McLaren", grid_position=1, raw_grid_value="1", positions_gained=None, race_time_or_gap=None, laps_completed=30, points=0.0, status="Disqualified"),
        ]
    }

    with patch.object(service.classification_service, "get_season_classifications", return_value=mock_1988):
        h2h = service.compare_teammates("prost", "senna", season=1988)
        assert h2h is not None
        assert h2h.qualifying_sessions_compared == 0
        assert h2h.qualifying_h2h == (0, 0)  # Missing data is NEVER counted as a loss!
        assert "pre-1994" in h2h.qualifying_coverage_note.lower()


# -----------------------------------------------------------------------------
# 3. Qualifying Analytics Service Tests
# -----------------------------------------------------------------------------

def test_qualifying_analytics_format_handling_and_grid_deltas():
    """Verify qualifying analytics calculates average position and grid drops strictly when grid > 0."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = QualifyingAnalyticsService(api_client=client)

    mock_q = {
        1: [QualifyingResultEntry(position=1, driver_id="verstappen", driver_name="Max Verstappen", driver_code="VER", constructor_id="red_bull", constructor_name="Red Bull", session_format="Knockout")],
        2: [QualifyingResultEntry(position=2, driver_id="verstappen", driver_name="Max Verstappen", driver_code="VER", constructor_id="red_bull", constructor_name="Red Bull", session_format="Knockout")],
    }
    mock_grids = {
        1: [StartingGridResultEntry(grid_position=1, raw_grid_value="1", qualifying_position=None, driver_id="verstappen", driver_name="Max Verstappen", constructor_name="Red Bull", grid_status="Grid P1", source_notes="")],
        # Round 2: grid penalty dropped from P2 to P7 on starting grid
        2: [StartingGridResultEntry(grid_position=7, raw_grid_value="7", qualifying_position=None, driver_id="verstappen", driver_name="Max Verstappen", constructor_name="Red Bull", grid_status="Grid P7", source_notes="")],
    }

    with patch.object(service.qualifying_service, "get_season_qualifying", return_value=mock_q):
        with patch.object(service.grid_service, "get_season_starting_grids", return_value=mock_grids):
            q_sum = service.get_driver_qualifying_summary("verstappen", season=2023)
            assert q_sum is not None
            assert q_sum.total_sessions == 2
            assert q_sum.pole_positions == 1
            assert q_sum.front_row_starts == 2
            assert q_sum.avg_qualifying_position == 1.5
            assert "Knockout" in q_sum.formats_encountered
            # R1: 1 - 1 = 0 delta; R2: 7 - 2 = +5 penalty drop
            assert q_sum.qualifying_to_grid_deltas == [0, 5]
            assert q_sum.avg_grid_drop_or_gain == 2.5


def test_qualifying_analytics_pre_1994_unavailable():
    """Verify pre-1994 season returns explicit unavailable summary without error."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = QualifyingAnalyticsService(api_client=client)

    q_sum = service.get_driver_qualifying_summary("senna", season=1990)
    assert q_sum is not None
    assert q_sum.total_sessions == 0
    assert q_sum.avg_qualifying_position is None
    assert any("pre-1994" in lim.lower() for lim in q_sum.coverage_limitations)


# -----------------------------------------------------------------------------
# 4. Race Performance Analytics Service Tests
# -----------------------------------------------------------------------------

def test_race_performance_positions_gained_safe_exclusion():
    """Verify positions gained strictly requires grid > 0 and position > 0, excluding grid == '0'."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = RacePerformanceAnalyticsService(api_client=client)

    mock_race_results = [
        {
            "season": "2022",
            "round": "1",
            "Results": [
                {
                    "position": "4",
                    "positionText": "4",
                    "points": "12",
                    "grid": "10",  # Started P10, finished P4 -> gained +6
                    "status": "Finished",
                    "Driver": {"driverId": "russell", "givenName": "George", "familyName": "Russell"},
                }
            ],
        },
        {
            "season": "2022",
            "round": "2",
            "Results": [
                {
                    "position": "5",
                    "positionText": "5",
                    "points": "10",
                    "grid": "0",  # Non-standard grid slot 0 -> MUST BE EXCLUDED from positions gained
                    "status": "Finished",
                    "Driver": {"driverId": "russell", "givenName": "George", "familyName": "Russell"},
                }
            ],
        },
        {
            "season": "2022",
            "round": "3",
            "Results": [
                {
                    "position": "18",
                    "positionText": "R",
                    "points": "0",
                    "grid": "2",  # DNF -> MUST BE EXCLUDED from positions gained
                    "status": "Collision",
                    "Driver": {"driverId": "russell", "givenName": "George", "familyName": "Russell"},
                }
            ],
        },
    ]

    with patch.object(client, "get_all_pages", return_value=mock_race_results):
        perf = service.get_driver_race_performance("russell")
        assert perf is not None
        assert perf.total_races == 3
        assert perf.starts == 3
        assert perf.classified_finishes == 2
        assert perf.dnfs == 1
        assert perf.dnf_rate_pct == 33.3
        # Positions gained must ONLY include round 1 (+6)
        assert perf.valid_grid_starts_count == 1
        assert perf.positions_gained_total == 6
        assert perf.avg_positions_gained == 6.0


# -----------------------------------------------------------------------------
# 5. Circuit Analytics Service Tests
# -----------------------------------------------------------------------------

def test_circuit_analytics_aggregation():
    """Verify circuit winners, podiums, and metadata aggregation from race results."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = CircuitAnalyticsService(api_client=client)

    mock_circuit_races = [
        {
            "season": "2021",
            "round": "14",
            "Results": [
                {"position": "1", "Driver": {"givenName": "Daniel", "familyName": "Ricciardo"}, "Constructor": {"name": "McLaren"}},
                {"position": "2", "Driver": {"givenName": "Lando", "familyName": "Norris"}, "Constructor": {"name": "McLaren"}},
                {"position": "3", "Driver": {"givenName": "Valtteri", "familyName": "Bottas"}, "Constructor": {"name": "Mercedes"}},
            ],
        },
        {
            "season": "2022",
            "round": "16",
            "Results": [
                {"position": "1", "Driver": {"givenName": "Max", "familyName": "Verstappen"}, "Constructor": {"name": "Red Bull"}},
                {"position": "2", "Driver": {"givenName": "Charles", "familyName": "Leclerc"}, "Constructor": {"name": "Ferrari"}},
                {"position": "3", "Driver": {"givenName": "George", "familyName": "Russell"}, "Constructor": {"name": "Mercedes"}},
            ],
        },
    ]

    mock_circuit = HistoricalCircuit(circuit_id="monza", name="Autodromo Nazionale Monza", locality="Monza", country="Italy")

    with patch.object(service.circuits_service, "get_circuit", return_value=mock_circuit):
        with patch.object(client, "get_all_pages", return_value=mock_circuit_races):
            c_hist = service.get_circuit_history("monza")
            assert c_hist is not None
            assert c_hist.circuit_name == "Autodromo Nazionale Monza"
            assert c_hist.total_races_hosted == 2
            assert c_hist.first_gp_year == 2021
            assert c_hist.last_gp_year == 2022
            winning_drivers = dict(c_hist.top_winning_drivers)
            assert winning_drivers["Daniel Ricciardo"] == 1
            assert winning_drivers["Max Verstappen"] == 1
            winning_teams = dict(c_hist.top_winning_constructors)
            assert winning_teams["McLaren"] == 1
            assert winning_teams["Red Bull"] == 1


# -----------------------------------------------------------------------------
# 6. Season Analytics Service Tests
# -----------------------------------------------------------------------------

def test_season_analytics_provisional_and_in_progress():
    """Verify in-progress seasons (e.g. 2026) are labeled provisional and reflect actual completed rounds."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = SeasonAnalyticsService(api_client=client)

    mock_cal = SeasonCalendar(year=2026, total_rounds=24, completed_rounds=1, upcoming_rounds=23, past_scheduled_rounds=0, is_in_progress=True)
    mock_classifications = {
        1: [
            RaceResultEntry(position=1, position_text="1", driver_id="leclerc", driver_name="Charles Leclerc", driver_number="16", constructor_id="ferrari", constructor_name="Ferrari", grid_position=1, raw_grid_value="1", positions_gained=0, race_time_or_gap=None, laps_completed=58, points=25.0, status="Finished"),
        ]
    }

    with patch.object(service.events_service, "get_season_calendar", return_value=mock_cal):
        with patch.object(service.classification_service, "get_season_classifications", return_value=mock_classifications):
            with patch.object(client, "get_json", return_value={"MRData": {"StandingsTable": {"StandingsLists": []}}}):
                summary = service.get_season_summary(2026)
                assert summary is not None
                assert summary.total_scheduled_rounds == 24
                assert summary.completed_rounds == 1
                assert summary.is_in_progress is True
                assert "In-Progress / Provisional" in summary.status_display
                assert summary.participating_drivers_count == 1
                assert summary.top_drivers_standings[0][1] == "Charles Leclerc"
                assert summary.top_drivers_standings[0][3] == 25.0
                assert any("provisional" in note.lower() for note in summary.coverage_notes)


# -----------------------------------------------------------------------------
# 7. Zero Filesystem Persistence Assertion
# -----------------------------------------------------------------------------

def test_phase4_zero_disk_persistence():
    """Assert zero filesystem writes occur during Phase 4 historical analytics operations."""
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
        d_service = DriverAnalyticsService(api_client=client)
        t_service = TeammateAnalyticsService(api_client=client)

        with patch.object(client, "get_all_pages", return_value=MOCK_DRIVER_RESULTS_ALONSO):
            with patch.object(d_service.qualifying_service, "get_season_qualifying", return_value={}):
                _ = d_service.get_driver_stats("alonso")

    assert len(write_calls) == 0, f"Detected illegal disk write operations: {write_calls}"


# -----------------------------------------------------------------------------
# 8. Method Signatures & Driver Stats In-Depth Verification Tests
# -----------------------------------------------------------------------------

def test_get_all_pages_bypass_cache_signature():
    """Verify get_all_pages explicitly supports bypass_cache parameter and passes it to get_json."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())

    with patch.object(client, "get_json", return_value={"MRData": {"total": "1", "RaceTable": {"Races": [{"round": "1"}]}}}) as mock_get_json:
        # Call with explicit bypass_cache=True
        res = client.get_all_pages(
            endpoint="2023/results",
            table_key="RaceTable",
            items_key="Races",
            bypass_cache=True,
        )
        assert len(res) == 1
        mock_get_json.assert_called_once_with("2023/results", params={"limit": 100, "offset": 0}, ttl_seconds=None, bypass_cache=True)


def test_senna_1994_preserves_records_and_poles_without_fabrication():
    """
    Verify Ayrton Senna's 1994 season:
    - 3 race entries, 3 starts, 0 classified finishes, 3 DNFs.
    - 3 pole positions (P1 in qualifying for rounds 1, 2, 3).
    - Career poles must NOT be zero.
    - Average finish must be None (no classified finishes; unclassified DNFs are not averaged).
    """
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = DriverAnalyticsService(api_client=client)

    mock_senna_1994_results = [
        {"season": "1994", "round": "1", "Results": [{"position": "18", "positionText": "R", "grid": "1", "status": "Spun off", "points": "0", "Driver": {"driverId": "senna", "givenName": "Ayrton", "familyName": "Senna"}}]},
        {"season": "1994", "round": "2", "Results": [{"position": "24", "positionText": "R", "grid": "1", "status": "Collision", "points": "0", "Driver": {"driverId": "senna", "givenName": "Ayrton", "familyName": "Senna"}}]},
        {"season": "1994", "round": "3", "Results": [{"position": "22", "positionText": "R", "grid": "1", "status": "Fatal accident", "points": "0", "Driver": {"driverId": "senna", "givenName": "Ayrton", "familyName": "Senna"}}]},
    ]

    mock_senna_1994_quali = {
        1: [QualifyingResultEntry(position=1, driver_id="senna", driver_name="Ayrton Senna", driver_code=None, constructor_id="williams", constructor_name="Williams", q1="1:15.962")],
        2: [QualifyingResultEntry(position=1, driver_id="senna", driver_name="Ayrton Senna", driver_code=None, constructor_id="williams", constructor_name="Williams", q1="1:10.218")],
        3: [QualifyingResultEntry(position=1, driver_id="senna", driver_name="Ayrton Senna", driver_code=None, constructor_id="williams", constructor_name="Williams", q1="1:21.548")],
    }

    with patch.object(client, "get_all_pages", return_value=mock_senna_1994_results):
        with patch.object(service.qualifying_service, "get_season_qualifying", return_value=mock_senna_1994_quali):
            stats = service.get_driver_stats("senna")
            assert stats is not None
            assert stats.driver_name == "Ayrton Senna"
            assert stats.total_entries == 3
            assert stats.total_starts == 3
            assert stats.classified_finishes == 0
            assert stats.dnf_count == 3
            # Crucial: Senna must NOT show zero career qualifying P1s!
            assert stats.qualifying_session_p1s == 3
            # Crucial: Average finish position must be None because he had 0 classified finishes
            assert stats.avg_finish_position is None
            # Season breakdown
            sb = stats.seasons_breakdown[0]
            assert sb.season == 1994
            assert sb.starts == 3
            assert sb.classified_finishes == 0
            assert sb.dnfs == 3
            assert sb.poles == 3
            assert sb.best_finish is None
            assert sb.avg_finish is None


def test_qualifying_api_error_does_not_become_zero_poles():
    """Verify an API error during qualifying retrieval sets poles to None, never converting it to zero."""
    from src.historical.api_client import HistoricalAPIError
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = DriverAnalyticsService(api_client=client)

    with patch.object(client, "get_all_pages", return_value=MOCK_DRIVER_RESULTS_ALONSO):
        # Simulate network failure during qualifying fetch
        with patch.object(service.qualifying_service, "get_season_qualifying", side_effect=HistoricalAPIError("Network timeout")):
            stats = service.get_driver_stats("alonso")
            assert stats is not None
            # Crucial: API failure must produce None, NOT 0!
            assert stats.qualifying_session_p1s is None
            assert "api/network error" in stats.poles_coverage_note.lower()
            assert stats.seasons_breakdown[0].poles is None


def test_seasons_without_recorded_entries_distinguished():
    """Verify seasons without participation (e.g. Alonso in 2002) are identified without claiming affirmative non-participation."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = DriverAnalyticsService(api_client=client)

    # Alonso raced in 2001 and 2003, with zero entries in 2002
    mock_alonso_gap = [
        {"season": "2001", "round": "1", "Results": [{"position": "12", "positionText": "12", "status": "Finished", "points": "0", "Driver": {"driverId": "alonso"}}]},
        {"season": "2003", "round": "1", "Results": [{"position": "7", "positionText": "7", "status": "Finished", "points": "2", "Driver": {"driverId": "alonso"}}]},
    ]

    with patch.object(client, "get_all_pages", return_value=mock_alonso_gap):
        with patch.object(service.qualifying_service, "get_season_qualifying", return_value={}):
            stats = service.get_driver_stats("alonso")
            assert stats is not None
            assert stats.seasons_active == [2001, 2003]
            # 2002 correctly detected as season without recorded entries in provider archive
            assert stats.seasons_without_recorded_entries == [2002]
            assert any("seasons without recorded championship entries" in note.lower() and "2002" in note for note in stats.coverage_notes)


def test_classified_despite_retiring_distinction():
    """Verify driver classified despite retiring (>=90% race distance) is classified and recorded in DNF count."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = DriverAnalyticsService(api_client=client)

    mock_late_retirement = [
        {
            "season": "2020",
            "round": "1",
            "Results": [
                {
                    "position": "10",
                    "positionText": "10",  # Classified P10
                    "status": "Engine",  # Retired before chequered flag
                    "points": "1",
                    "Driver": {"driverId": "norris", "givenName": "Lando", "familyName": "Norris"},
                }
            ],
        }
    ]

    with patch.object(client, "get_all_pages", return_value=mock_late_retirement):
        with patch.object(service.qualifying_service, "get_season_qualifying", return_value={}):
            stats = service.get_driver_stats("norris")
            assert stats is not None
            assert stats.classified_finishes == 1
            assert stats.classified_retirements_count == 1
            assert stats.dnf_count == 1
            assert stats.avg_finish_position == 10.0
            assert "Engine" in stats.retirement_reasons


def test_senna_poles_unavailable_with_covered_count():
    """Verify that a career spanning pre-1994 sets career poles to None, with known_covered_poles exposed."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = DriverAnalyticsService(api_client=client)

    mock_senna_multi_season = [
        # Pre-1994 season (1993)
        {"season": "1993", "round": "16", "Results": [{"position": "1", "positionText": "1", "grid": "1", "status": "Finished", "points": "10", "Driver": {"driverId": "senna", "givenName": "Ayrton", "familyName": "Senna"}}]},
        # Covered season (1994)
        {"season": "1994", "round": "1", "Results": [{"position": "18", "positionText": "R", "grid": "1", "status": "Spun off", "points": "0", "Driver": {"driverId": "senna", "givenName": "Ayrton", "familyName": "Senna"}}]},
        {"season": "1994", "round": "2", "Results": [{"position": "24", "positionText": "R", "grid": "1", "status": "Collision", "points": "0", "Driver": {"driverId": "senna", "givenName": "Ayrton", "familyName": "Senna"}}]},
        {"season": "1994", "round": "3", "Results": [{"position": "22", "positionText": "R", "grid": "1", "status": "Fatal accident", "points": "0", "Driver": {"driverId": "senna", "givenName": "Ayrton", "familyName": "Senna"}}]},
    ]

    mock_senna_1994_quali = {
        1: [QualifyingResultEntry(position=1, driver_id="senna", driver_name="Ayrton Senna", driver_code=None, constructor_id="williams", constructor_name="Williams", q1="1:15.962")],
        2: [QualifyingResultEntry(position=1, driver_id="senna", driver_name="Ayrton Senna", driver_code=None, constructor_id="williams", constructor_name="Williams", q1="1:10.218")],
        3: [QualifyingResultEntry(position=1, driver_id="senna", driver_name="Ayrton Senna", driver_code=None, constructor_id="williams", constructor_name="Williams", q1="1:21.548")],
    }

    with patch.object(client, "get_all_pages", return_value=mock_senna_multi_season):
        with patch.object(service.qualifying_service, "get_season_qualifying", return_value=mock_senna_1994_quali):
            stats = service.get_driver_stats("senna")
            assert stats is not None
            # Complete career poles must be None because pre-1994 seasons lack qualifying coverage
            assert stats.qualifying_session_p1s is None
            # Covered-era subtotal must be 3
            assert stats.known_covered_poles == 3
            assert "unavailable" in stats.poles_coverage_note.lower()
            assert "3" in stats.poles_coverage_note


def test_disqualified_starter_increments_starts():
    """Verify that a driver who started the race and was disqualified/excluded is counted in total_starts."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = DriverAnalyticsService(api_client=client)

    mock_dsq = [
        {
            "season": "1988",
            "round": "1",
            "Results": [
                {
                    "position": "10",
                    "positionText": "D",  # Disqualified after starting
                    "status": "Disqualified",
                    "points": "0",
                    "grid": "1",
                    "Driver": {"driverId": "senna", "givenName": "Ayrton", "familyName": "Senna"},
                }
            ],
        }
    ]

    with patch.object(client, "get_all_pages", return_value=mock_dsq):
        with patch.object(service.qualifying_service, "get_season_qualifying", return_value={}):
            stats = service.get_driver_stats("senna")
            assert stats is not None
            assert stats.total_entries == 1
            # Starter who got disqualified counts as a start
            assert stats.total_starts == 1
            # Disqualified is not a classified finish
            assert stats.classified_finishes == 0
            # Disqualified is not counted as DNF (mechanical/accident retirement)
            assert stats.dnf_count == 0


def test_pre_race_withdrawal_does_not_increment_starts():
    """Verify that a driver who withdrew or did not start (DNS/DNQ/W) does not increment total_starts."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = DriverAnalyticsService(api_client=client)

    mock_dns = [
        {
            "season": "2005",
            "round": "9",  # US GP 2005 Michelin withdrawal
            "Results": [
                {
                    "position": "20",
                    "positionText": "W",  # Withdrew
                    "status": "Withdrew",
                    "points": "0",
                    "grid": "6",
                    "Driver": {"driverId": "alonso", "givenName": "Fernando", "familyName": "Alonso"},
                }
            ],
        }
    ]

    with patch.object(client, "get_all_pages", return_value=mock_dns):
        with patch.object(service.qualifying_service, "get_season_qualifying", return_value={}):
            stats = service.get_driver_stats("alonso")
            assert stats is not None
            assert stats.total_entries == 1
            # Did not start
            assert stats.total_starts == 0
            assert stats.classified_finishes == 0
            assert stats.dnf_count == 0


def test_lapped_finishers_not_counted_as_dnfs():
    """Verify that drivers who finished the race laps down (+1 Lap, etc.) are classified finishes, NOT DNFs."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = DriverAnalyticsService(api_client=client)

    mock_lapped = [
        {
            "season": "2023",
            "round": "1",
            "Results": [
                {
                    "position": "11",
                    "positionText": "11",
                    "status": "+1 Lap",
                    "points": "0",
                    "grid": "12",
                    "Driver": {"driverId": "albon", "givenName": "Alexander", "familyName": "Albon"},
                }
            ],
        }
    ]

    with patch.object(client, "get_all_pages", return_value=mock_lapped):
        with patch.object(service.qualifying_service, "get_season_qualifying", return_value={}):
            stats = service.get_driver_stats("albon")
            assert stats is not None
            assert stats.total_entries == 1
            assert stats.total_starts == 1
            assert stats.classified_finishes == 1
            assert stats.dnf_count == 0
            assert stats.avg_finish_position == 11.0


def test_empty_results_do_not_produce_zero_start_season_breakdown():
    """Verify that a race without results for the driver does not create an empty 0-start season breakdown."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = DriverAnalyticsService(api_client=client)

    mock_with_empty_2026 = [
        # Driver raced in 2024
        {
            "season": "2024",
            "round": "1",
            "Results": [
                {
                    "position": "1",
                    "positionText": "1",
                    "status": "Finished",
                    "points": "25",
                    "grid": "1",
                    "Driver": {"driverId": "alonso", "givenName": "Fernando", "familyName": "Alonso"},
                }
            ],
        },
        # 2026 race exists in schedule/calendar but Results is empty or driver did not participate
        {
            "season": "2026",
            "round": "1",
            "Results": [],
        },
    ]

    with patch.object(client, "get_all_pages", return_value=mock_with_empty_2026):
        with patch.object(service.qualifying_service, "get_season_qualifying", return_value={}):
            stats = service.get_driver_stats("alonso")
            assert stats is not None
            # 2026 must NOT be in seasons_active since driver had no entries in 2026
            assert stats.seasons_active == [2024]
            assert len(stats.seasons_breakdown) == 1
            assert stats.seasons_breakdown[0].season == 2024


def test_modern_lapped_status_handling_career_and_season():
    """Verify modern 'Lapped' status is recognized as a running finish, NOT counted as retirement or DNF."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = DriverAnalyticsService(api_client=client)

    mock_results = [
        # Round 1: Finished on lead lap
        {"season": "2024", "round": "1", "Results": [{"position": "1", "positionText": "1", "status": "Finished", "points": "25", "grid": "1", "Driver": {"driverId": "alonso", "givenName": "Fernando", "familyName": "Alonso"}}]},
        # Round 2: Lapped (-1 lap or more in modern Jolpica format)
        {"season": "2024", "round": "2", "Results": [{"position": "12", "positionText": "12", "status": "Lapped", "points": "0", "grid": "14", "Driver": {"driverId": "alonso", "givenName": "Fernando", "familyName": "Alonso"}}]},
        # Round 3: +1 Lap (traditional Ergast format)
        {"season": "2024", "round": "3", "Results": [{"position": "11", "positionText": "11", "status": "+1 Lap", "points": "0", "grid": "10", "Driver": {"driverId": "alonso", "givenName": "Fernando", "familyName": "Alonso"}}]},
    ]

    with patch.object(client, "get_all_pages", return_value=mock_results):
        with patch.object(service.qualifying_service, "get_season_qualifying", return_value={}):
            stats = service.get_driver_stats("alonso")
            assert stats is not None
            assert stats.total_entries == 3
            assert stats.total_starts == 3
            assert stats.running_finishes == 3  # Finished + Lapped + +1 Lap
            assert stats.classified_finishes == 3
            assert stats.classified_retirements_count == 0
            assert stats.unclassified_retirements_count == 0
            assert stats.dnf_count == 0  # CRUCIAL: Lapped cars MUST NOT be counted as DNFs!
            assert stats.non_starts_count == 0
            # Season breakdown
            sb = stats.seasons_breakdown[0]
            assert sb.running_finishes == 3
            assert sb.classified_finishes == 3
            assert sb.dnfs == 0


def test_retired_driver_with_numeric_classification_position():
    """Verify that a driver who retired (e.g. Engine failure) but completed >=90% distance is classified AND a DNF."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = DriverAnalyticsService(api_client=client)

    mock_classified_retirement = [
        {
            "season": "2022",
            "round": "20",
            "Results": [
                {
                    "position": "19",
                    "positionText": "19",  # Numerically classified P19
                    "status": "Engine",  # Terminal engine blowout
                    "points": "0",
                    "grid": "9",
                    "Driver": {"driverId": "alonso", "givenName": "Fernando", "familyName": "Alonso"},
                }
            ],
        }
    ]

    with patch.object(client, "get_all_pages", return_value=mock_classified_retirement):
        with patch.object(service.qualifying_service, "get_season_qualifying", return_value={}):
            stats = service.get_driver_stats("alonso")
            assert stats is not None
            assert stats.total_starts == 1
            assert stats.running_finishes == 0
            assert stats.classified_finishes == 1
            assert stats.classified_retirements_count == 1
            assert stats.unclassified_retirements_count == 0
            assert stats.dnf_count == 1  # Retiring makes it a DNF
            assert stats.avg_finish_position == 19.0


def test_driver_qualifying_direct_endpoint_and_error_handling():
    """Verify driver-specific qualifying retrieval via get_driver_qualifying_results and API error handling."""
    from src.historical.api_client import HistoricalAPIError
    client = HistoricalAPIClient(cache=InMemoryTTLCache())
    service = DriverAnalyticsService(api_client=client)

    mock_race_results = [
        {"season": "2005", "round": "1", "Results": [{"position": "3", "positionText": "3", "status": "Finished", "points": "6", "grid": "13", "Driver": {"driverId": "alonso"}}]},
        {"season": "2005", "round": "2", "Results": [{"position": "1", "positionText": "1", "status": "Finished", "points": "10", "grid": "1", "Driver": {"driverId": "alonso"}}]},
    ]

    mock_driver_quali = [
        {"season": "2005", "round": "1", "QualifyingResults": [{"position": "13", "Driver": {"driverId": "alonso"}}]},
        {"season": "2005", "round": "2", "QualifyingResults": [{"position": "1", "Driver": {"driverId": "alonso"}}]},
    ]

    # Test 1: Successful retrieval via get_driver_qualifying_results
    with patch.object(client, "get_all_pages", return_value=mock_race_results):
        with patch.object(service.qualifying_service, "get_driver_qualifying_results", return_value=mock_driver_quali):
            stats = service.get_driver_stats("alonso")
            assert stats is not None
            assert stats.qualifying_session_p1s == 1
            assert stats.seasons_breakdown[0].poles == 1

    # Test 2: API error during qualifying retrieval sets poles to None, NEVER partial 0
    with patch.object(client, "get_all_pages", return_value=mock_race_results):
        with patch.object(service.qualifying_service, "get_driver_qualifying_results", side_effect=HistoricalAPIError("Network timeout")):
            stats = service.get_driver_stats("alonso")
            assert stats is not None
            assert stats.qualifying_session_p1s is None
            assert "api/network error" in stats.poles_coverage_note.lower()
            assert stats.seasons_breakdown[0].poles is None



