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
    """Dominance service calculates constructor win rates and season rankings."""
    service = DominanceAnalyticsService()
    constructors = service.get_constructor_dominance()
    assert len(constructors) >= 15
    # Ferrari, McLaren, Mercedes, Red Bull among top ranks
    top_teams = [c["constructor"] for c in constructors[:4]]
    assert "Ferrari" in top_teams
    assert "McLaren" in top_teams

    # Highest win rate seasons
    seasons = service.get_highest_win_rate_seasons(top_n=5)
    assert len(seasons) == 5
    assert seasons[0]["win_pct"] > 90.0  # Red Bull 2023 or McLaren 1988

    # 1-2 finishes
    one_twos = service.get_one_two_finish_rankings(top_n=5)
    assert len(one_twos) == 5
    assert one_twos[0]["one_twos"] > 50


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

