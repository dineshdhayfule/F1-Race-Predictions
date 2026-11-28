"""
Isolated Unit Tests for Historical Motorsport Intelligence Expansion (Options 1–11)
===================================================================================
Tests:
- SessionResultsService (Race, Qualifying, Sprint, Practice Telemetry)
- DominanceAnalyticsService (Constructors, Highest Win Rate Seasons, 1-2 Finishes)
- AllTimeRecordsService (Wins, Poles, Podiums, Streaks, Age Milestones)
- ReliabilityAnalyticsService (Era Attrition, Most Attritional Races, Reliable Teams)
- EraAnalyticsService (Canonical 8 Eras Overview, Era Comparisons)
- HistoricalStatsMenu instantiation & options mapping (Options 1–11)
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from apexgrid.historical.models import RaceResultEntry, SprintResultEntry, QualifyingResultEntry
from apexgrid.historical.api_client import HistoricalAPIClient
from apexgrid.historical.analytics import (
    SessionResultsService,
    DominanceAnalyticsService,
    AllTimeRecordsService,
    ReliabilityAnalyticsService,
    EraAnalyticsService,
)
from apexgrid.cli.historical_menu import HistoricalStatsMenu


def test_session_results_service_practice_classification():
    """Practice classification calculates relative lap deltas correctly."""
    mock_df = pd.DataFrame({
        "Driver": ["VER", "NOR", "LEC"],
        "LapTimeSeconds": [80.500, 80.650, 81.100],
    })
    mock_loader = MagicMock()
    mock_loader.load_all_practice_sessions.return_value = {
        1: {"fp1": mock_df}
    }
    service = SessionResultsService(data_loader=mock_loader)
    
    # Pre-2018 must gracefully return empty list
    assert service.get_practice_classification(2017, 1, "fp1") == []

    # 2024 returns ordered classification
    res = service.get_practice_classification(2024, 1, "fp1")
    assert len(res) == 3
    assert res[0]["driver"] == "VER"
    assert res[0]["gap"] == "P1"
    assert res[1]["driver"] == "NOR"
    assert "+0.150s" in res[1]["gap"]


def test_session_results_service_sprint_parsing():
    """Sprint results parser maps Jolpica JSON into SprintResultEntry."""
    mock_api = MagicMock()
    mock_api.get_json.return_value = {
        "MRData": {
            "RaceTable": {
                "Races": [{
                    "SprintResults": [
                        {
                            "position": "1",
                            "positionText": "1",
                            "points": "8",
                            "grid": "1",
                            "laps": "17",
                            "status": "Finished",
                            "Driver": {"driverId": "perez", "givenName": "Sergio", "familyName": "Perez"},
                            "Constructor": {"name": "Red Bull"},
                            "Time": {"time": "33:17.667"}
                        }
                    ]
                }]
            }
        }
    }
    service = SessionResultsService(api_client=mock_api)
    results = service.get_sprint_results(2023, 4)
    assert len(results) == 1
    assert results[0].driver_id == "perez"
    assert results[0].driver_name == "Sergio Perez"
    assert results[0].points == 8.0
    assert results[0].position == 1


def test_dominance_analytics_service():
    """Dominance service computes exact win_pct and sorts constructors by wins/championships."""
    service = DominanceAnalyticsService()
    constructors = service.get_constructor_dominance()
    assert len(constructors) >= 15
    top_teams = [c["constructor"] for c in constructors[:4]]
    assert "Ferrari" in top_teams
    assert "McLaren" in top_teams
    for entry in constructors:
        expected_win_pct = round((entry["wins"] / entry["starts"]) * 100, 2)
        assert entry["win_percentage"] == expected_win_pct

    # Highest win rate seasons sorted descending
    seasons = service.get_highest_win_rate_seasons(top_n=5)
    assert len(seasons) == 5
    assert seasons[0]["win_pct"] > 90.0
    assert all(seasons[i]["win_pct"] >= seasons[i + 1]["win_pct"] for i in range(len(seasons) - 1))

    # 1-2 finishes sorted descending
    one_twos = service.get_one_two_finish_rankings(top_n=5)
    assert len(one_twos) == 5
    assert one_twos[0]["one_twos"] > 50
    assert all(one_twos[i]["one_twos"] >= one_twos[i + 1]["one_twos"] for i in range(len(one_twos) - 1))


def test_session_results_service_race_results_and_practice_nan_handling():
    """SessionResultsService parses Jolpica race JSON (points/status) and drops NaN practice laps."""
    mock_api = MagicMock()
    mock_api.get_json.return_value = {
        "MRData": {
            "RaceTable": {
                "Races": [{
                    "Results": [
                        {
                            "position": "1",
                            "positionText": "1",
                            "points": "25",
                            "grid": "2",
                            "laps": "57",
                            "status": "Finished",
                            "Driver": {"driverId": "norris", "givenName": "Lando", "familyName": "Norris"},
                            "Constructor": {"name": "McLaren"},
                            "Time": {"time": "1:30:00.000"},
                        },
                        {
                            "position": "20",
                            "positionText": "R",
                            "points": "0",
                            "grid": "5",
                            "laps": "12",
                            "status": "Engine",
                            "Driver": {"driverId": "leclerc", "givenName": "Charles", "familyName": "Leclerc"},
                            "Constructor": {"name": "Ferrari"},
                        },
                    ]
                }]
            }
        }
    }
    service = SessionResultsService(api_client=mock_api)
    race_res = service.get_race_results(2025, 1)
    assert len(race_res) == 2
    assert race_res[0].driver_name == "Lando Norris"
    assert race_res[0].points == 25.0
    assert race_res[1].status == "Engine"
    assert race_res[1].points == 0.0

    mock_loader = MagicMock()
    mock_loader.load_all_practice_sessions.return_value = {
        1: {"fp1": pd.DataFrame({"Driver": ["VER", "STR"], "LapTimeSeconds": [81.0, float("nan")]})}
    }
    practice_svc = SessionResultsService(data_loader=mock_loader)
    practice_res = practice_svc.get_practice_classification(2025, 1, "fp1")
    assert [row["driver"] for row in practice_res] == ["VER"]


def test_all_time_records_service():
    """All-time records service provides accurate historical metrics."""
    service = AllTimeRecordsService()
    
    # Wins
    wins = service.get_all_time_wins(top_n=5)
    assert len(wins) == 5
    assert wins[0]["driver"] == "Lewis Hamilton"
    assert wins[0]["wins"] >= 100

    # Poles
    poles = service.get_all_time_poles(top_n=5)
    assert len(poles) == 5
    assert poles[0]["driver"] == "Lewis Hamilton"

    # Streaks
    streaks = service.get_consecutive_records("consecutive_wins")
    assert len(streaks) >= 5
    assert streaks[0]["driver"] == "Max Verstappen"
    assert streaks[0]["count"] == 10

    # Age milestones
    milestones = service.get_age_milestones()
    assert "youngest_winners" in milestones
    assert "oldest_winners" in milestones
    assert "most_starts_no_win" in milestones
    assert milestones["youngest_winners"][0]["driver"] == "Max Verstappen"
    assert milestones["oldest_winners"][0]["driver"] == "Luigi Fagioli"


def test_reliability_analytics_service():
    """Reliability service returns historical attrition and race records."""
    service = ReliabilityAnalyticsService()
    
    # Era attrition
    attrition = service.get_season_attrition_rates()
    assert len(attrition) >= 8
    # Modern finish rate should exceed 1950s finish rate
    assert attrition[-1]["finish_rate_pct"] > attrition[0]["finish_rate_pct"]

    # Most attritional races
    races = service.get_most_attritional_races(top_n=5)
    assert len(races) == 5
    assert any("Monaco" in r["race"] for r in races)


def test_era_analytics_service():
    """Era service returns all 8 canonical F1 regulatory eras."""
    service = EraAnalyticsService()
    eras = service.get_era_overview()
    assert len(eras) == 8
    era_ids = [e["id"] for e in eras]
    assert "front_engine" in era_ids
    assert "first_turbo" in era_ids
    assert "v10_high_rev" in era_ids
    assert "v6_turbo_hybrid" in era_ids
    assert "active_aero_2026" in era_ids

    # Comparison
    cmp_res = service.compare_two_eras("v10_high_rev", "v6_turbo_hybrid")
    assert cmp_res is not None
    assert cmp_res["era_1"]["name"] == "Atmospheric V10 / High-Rev Era"


def test_historical_menu_instantiation():
    """HistoricalStatsMenu instantiates all 11 service attributes cleanly."""
    menu = HistoricalStatsMenu()
    assert hasattr(menu, "session_service")
    assert hasattr(menu, "driver_service")
    assert hasattr(menu, "teammate_service")
    assert hasattr(menu, "qualifying_service")
    assert hasattr(menu, "race_service")
    assert hasattr(menu, "circuit_service")
    assert hasattr(menu, "season_service")
    assert hasattr(menu, "dominance_service")
    assert hasattr(menu, "records_service")
    assert hasattr(menu, "reliability_service")
    assert hasattr(menu, "era_service")
    assert hasattr(menu, "view_session_results")
    assert hasattr(menu, "view_constructor_dominance")
    assert hasattr(menu, "view_all_time_records")
    assert hasattr(menu, "view_reliability_stats")
    assert hasattr(menu, "view_era_comparisons")


def test_race_result_entry_backward_compatible_properties():
    """RaceResultEntry and SprintResultEntry support .grid, .laps, .time_str, .given_name, .family_name."""
    entry = RaceResultEntry(
        position=1,
        position_text="1",
        driver_id="max_verstappen",
        driver_name="Max Verstappen",
        driver_number="3",
        constructor_id="red_bull",
        constructor_name="Red Bull",
        grid_position=2,
        raw_grid_value="2",
        positions_gained=1,
        race_time_or_gap="1:25:00.000",
        laps_completed=57,
        points=25.0,
        status="Finished",
    )
    assert entry.grid == 2
    assert entry.laps == 57
    assert entry.time_str == "1:25:00.000"
    assert entry.given_name == "Max"
    assert entry.family_name == "Verstappen"

    sprint_entry = SprintResultEntry(
        position=1,
        position_text="1",
        driver_id="norris",
        driver_name="Lando Norris",
        constructor_name="McLaren",
        grid_position=1,
        laps_completed=19,
        points=8.0,
        status="Finished",
        race_time_or_gap="28:15.123",
    )
    assert sprint_entry.grid == 1
    assert sprint_entry.laps == 19
    assert sprint_entry.time_str == "28:15.123"
    assert sprint_entry.given_name == "Lando"
    assert sprint_entry.family_name == "Norris"


def test_session_results_service_is_sprint_round():
    """is_sprint_round accurately identifies sprint vs conventional rounds."""
    service = SessionResultsService()
    # Pre-2021 is never sprint
    assert service.is_sprint_round(2017, 1) is False
    assert service.is_sprint_round(2020, 1) is False
    # 2026 Round 4 Miami is a sprint round in local data
    assert service.is_sprint_round(2026, 4) is True
    # 2026 Round 21 Las Vegas is conventional
    assert service.is_sprint_round(2026, 21) is False


def test_session_results_service_sprint_qualifying():
    """get_sprint_qualifying_results parses SQ telemetry and computes delta to pole."""
    service = SessionResultsService()
    sqs = service.get_sprint_qualifying_results(2026, 4)
    assert len(sqs) > 0
    p1 = sqs[0]
    assert p1.position == 1
    assert p1.driver_name == "Lando Norris"
    assert p1.session_format == "Sprint Qualifying (SQ)"
    assert p1.session_code == "SQ"
    assert p1.best_lap_time is not None
    assert p1.gap_to_pole is None  # P1 has no delta to pole
    if len(sqs) > 1:
        assert sqs[1].gap_to_pole is not None
        assert sqs[1].gap_to_pole.startswith("+")


def test_season_analytics_full_grid_standings_and_gaps():
    """SeasonAnalyticsService populates full-grid standings, gap_to_leader, podiums, poles, and dnfs."""
    from types import SimpleNamespace
    from apexgrid.historical.analytics import SeasonAnalyticsService, DriverSeasonStanding, ConstructorSeasonStanding

    mock_events = MagicMock()
    mock_events.get_season_calendar.return_value = SimpleNamespace(total_rounds=23)

    # 13 drivers across 11 constructors in 1 completed round (including a Jolpica-style DNF with numeric position and a DNS)
    entries = []
    for i in range(1, 14):
        is_dnf = i == 12
        is_dns = i == 13
        entries.append(
            RaceResultEntry(
                position=i,
                position_text="W" if is_dns else ("R" if is_dnf else str(i)),
                driver_id=f"drv_{i}",
                driver_name=f"Driver {i}",
                driver_number=str(i),
                constructor_id=f"team_{(i - 1) % 11 + 1}",
                constructor_name=f"Team {(i - 1) % 11 + 1}",
                grid_position=i,
                raw_grid_value=str(i),
                positions_gained=0,
                race_time_or_gap="1:30:00",
                laps_completed=57 if not (is_dnf or is_dns) else (10 if is_dnf else 0),
                points=float(max(0, 26 - i * 2)),
                status="Did not start" if is_dns else ("Engine" if is_dnf else "Finished"),
            )
        )
    mock_class = MagicMock()
    mock_class.get_season_classifications.return_value = {1: entries}

    mock_quali = MagicMock()
    mock_quali.get_season_qualifying.return_value = {
        1: [
            QualifyingResultEntry(
                position=1,
                driver_id="drv_1",
                driver_name="Driver 1",
                driver_code="DR1",
                constructor_id="team_1",
                constructor_name="Team 1",
                best_lap_time="1:20.000",
            )
        ]
    }

    mock_api = MagicMock()
    mock_api.get_json.return_value = {}  # Exercise classification fallback path

    svc = SeasonAnalyticsService(
        api_client=mock_api,
        events_service=mock_events,
        classification_service=mock_class,
        qualifying_service=mock_quali,
    )
    summary = svc.get_season_summary(2026)
    assert summary is not None
    assert len(summary.top_drivers_standings) == 10
    assert len(summary.top_constructors_standings) == 10
    assert len(summary.full_drivers_standings) == 13
    assert len(summary.full_constructors_standings) == 11
    assert isinstance(summary.full_drivers_standings[0], DriverSeasonStanding)
    assert isinstance(summary.full_constructors_standings[0], ConstructorSeasonStanding)

    leader = summary.full_drivers_standings[0]
    assert leader.rank == 1
    assert leader.gap_to_leader == 0.0
    assert leader.podiums == 1
    assert leader.poles == 1

    p2 = summary.full_drivers_standings[1]
    assert p2.gap_to_leader == round(leader.points - p2.points, 2)
    assert p2.podiums == 1

    dnf_drv = next(s for s in summary.full_drivers_standings if s.driver_name == "Driver 12")
    assert dnf_drv.dnfs == 1
    assert dnf_drv.dns == 0

    dns_drv = next(s for s in summary.full_drivers_standings if s.driver_name == "Driver 13")
    assert dns_drv.dns == 1
    assert dns_drv.dnfs == 0

