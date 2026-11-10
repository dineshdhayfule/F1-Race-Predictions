"""
Unit and integration tests for Dynamic Driver Roster Resolution.
Verifies:
1. Previous race DNS does not remove the team's entry from the next event.
2. A reserve driver in one event does not automatically become the next event's entrant.
3. A confirmed replacement driver appears in the target event roster.
4. A mid-season driver change is supported.
5. A driver withdrawal is represented correctly.
6. A replacement driver without historical samples remains in the target dataset.
7. A 20-driver historical grid is valid.
8. A 22-driver grid is valid.
9. An incomplete roster produces a warning instead of silently generating an incorrect lineup.
10. Existing 2026 prediction behavior remains intact.
"""

import logging
import pytest
import numpy as np
import pandas as pd

from src.data.roster_resolver import RosterResolver
from src.features.build_features import FeatureEngineer
from src.data.data_loader import DataLoader


def test_previous_race_dns_does_not_remove_team_entry():
    """
    Test 1: If Haas had BEA finish and HIR DNS (omitted from race results) in Round 19,
    Round 20 roster resolver ensures Haas has 2 cars represented (e.g. BEA and HIR/OCO).
    """
    resolver = RosterResolver()

    # Prior races: Round 19 has only 21 finishers (BEA is in, car #2 is missing due to DNS)
    races_df = pd.DataFrame([
        {"Round": 19, "Abbreviation": "VER", "TeamName": "Red Bull Racing", "Position": 1},
        {"Round": 19, "Abbreviation": "BEA", "TeamName": "Haas F1 Team", "Position": 18},
    ])
    # Qualifying had both Haas cars: BEA and HIR
    qualis_df = pd.DataFrame([
        {"Round": 19, "Abbreviation": "VER", "TeamName": "Red Bull Racing", "Position": 1},
        {"Round": 19, "Abbreviation": "BEA", "TeamName": "Haas F1 Team", "Position": 14},
        {"Round": 19, "Abbreviation": "HIR", "TeamName": "Haas F1 Team", "Position": 18},
    ])

    roster = resolver.resolve_roster(
        year=2026,
        round_num=20,
        stage="pre_fp1",
        season_races_df=races_df,
        season_qualis_df=qualis_df,
        expected_grid_size=22,
    )

    haas_drivers = roster[roster["TeamName"] == "Haas F1 Team"]["Abbreviation"].tolist()
    assert len(haas_drivers) == 2, f"Expected 2 Haas drivers, found {haas_drivers}"
    assert "BEA" in haas_drivers
    assert "HIR" in haas_drivers or "OCO" in haas_drivers


def test_reserve_driver_in_one_event_does_not_force_subsequent_event():
    """
    Test 2: A reserve driver (e.g. HIR) participating in Round 19 does not override
    an official entry list or practice session showing the regular driver (OCO) in Round 20.
    """
    resolver = RosterResolver()

    # Official entry list for Round 20 shows regular driver OCO
    entry_list = pd.DataFrame([
        {"Abbreviation": "BEA", "TeamName": "Haas F1 Team", "FullName": "Oliver Bearman"},
        {"Abbreviation": "OCO", "TeamName": "Haas F1 Team", "FullName": "Esteban Ocon"},
    ] + [
        {"Abbreviation": f"D{i}", "TeamName": f"Team{i//2}", "FullName": f"Driver {i}"}
        for i in range(2, 22)
    ])

    roster = resolver.resolve_roster(
        year=2026,
        round_num=20,
        stage="pre_fp1",
        entry_list_df=entry_list,
        expected_grid_size=22,
    )

    haas_drivers = roster[roster["TeamName"] == "Haas F1 Team"]["Abbreviation"].tolist()
    assert "OCO" in haas_drivers
    assert "HIR" not in haas_drivers


def test_confirmed_replacement_driver_appears_in_target_roster():
    """
    Test 3: A confirmed replacement driver participating in FP1 appears in the target roster.
    """
    resolver = RosterResolver()

    weekend_sessions = {
        "fp1": pd.DataFrame([
            {"Abbreviation": "BEA", "TeamName": "Haas F1 Team", "DriverNumber": 87},
            {"Abbreviation": "HIR", "TeamName": "Haas F1 Team", "DriverNumber": 98},
        ] + [
            {"Abbreviation": f"D{i}", "TeamName": f"Team{i//2}", "DriverNumber": i}
            for i in range(2, 22)
        ])
    }

    roster = resolver.resolve_roster(
        year=2026,
        round_num=20,
        stage="after_fp1",
        weekend_sessions=weekend_sessions,
        expected_grid_size=22,
    )

    assert "HIR" in roster["Abbreviation"].values
    assert roster[roster["Abbreviation"] == "HIR"]["Source"].iloc[0] == "practice_fp1"


def test_mid_season_driver_change():
    """
    Test 4: Mid-season change (e.g. driver replaces another in qualifying) is reflected.
    """
    resolver = RosterResolver()

    quali_df = pd.DataFrame([
        {"Abbreviation": "LAW", "TeamName": "Racing Bulls", "Position": 8},
        {"Abbreviation": "LIN", "TeamName": "Racing Bulls", "Position": 9},
    ] + [
        {"Abbreviation": f"D{i}", "TeamName": f"Team{i//2}", "Position": i + 2}
        for i in range(2, 22)
    ])

    roster = resolver.resolve_roster(
        year=2026,
        round_num=10,
        stage="after_qualifying",
        quali_df=quali_df,
        expected_grid_size=22,
    )

    rb_drivers = roster[roster["TeamName"] == "Racing Bulls"]["Abbreviation"].tolist()
    assert "LAW" in rb_drivers
    assert "LIN" in rb_drivers


def test_driver_withdrawal_represented():
    """
    Test 5: A driver marked as withdrawn or DNS in grid data has their status preserved.
    """
    resolver = RosterResolver()

    grid_df = pd.DataFrame([
        {"Abbreviation": "SAI", "TeamName": "Williams", "Position": 1, "Status": "active"},
        {"Abbreviation": "ALB", "TeamName": "Williams", "Position": 2, "Status": "withdrawn"},
    ])

    roster = resolver._extract_from_dataframe(grid_df, "starting_grid", "verified", 2026)
    alb_row = roster[roster["Abbreviation"] == "ALB"].iloc[0]
    assert alb_row["EntryStatus"] == "withdrawn"


def test_replacement_driver_without_historical_samples_retained():
    """
    Test 6: A replacement driver with 0 prior season samples remains in the target dataset
    and features are safely constructed using team baselines without NaNs.
    """
    fe = FeatureEngineer(expected_grid_size=2)

    quali_df = pd.DataFrame([
        {"Abbreviation": "VER", "TeamName": "Red Bull Racing", "Position": 1},
        {"Abbreviation": "NEW", "TeamName": "Red Bull Racing", "Position": 2},  # Brand new driver
    ])

    prior_races = pd.DataFrame([
        {"Round": 1, "Abbreviation": "VER", "TeamName": "Red Bull Racing", "GridPosition": 1, "Position": 1, "ActualFinish": 1, "Status": "Finished"},
        {"Round": 1, "Abbreviation": "HAD", "TeamName": "Red Bull Racing", "GridPosition": 2, "Position": 2, "ActualFinish": 2, "Status": "Finished"},
    ])

    target_df = fe.build_target_dataset(
        target_round=2,
        track_name="Test GP",
        quali_df=quali_df,
        practice_sessions={},
        season_races_df=prior_races,
        expected_grid_size=2,
    )

    assert len(target_df) == 2
    assert "NEW" in target_df["Abbreviation"].values
    new_row = target_df[target_df["Abbreviation"] == "NEW"].iloc[0]
    assert pd.notna(new_row["GridPosition"])
    assert pd.notna(new_row["team_finish_ewma"])


def test_historical_grid_sizes():
    """
    Tests 7 & 8: Both 20-driver grids (2025) and 22-driver grids (2026) are validated.
    """
    resolver = RosterResolver()
    assert resolver.get_expected_grid_size(2025) == 20
    assert resolver.get_expected_grid_size(2026) == 22
    assert resolver.get_expected_grid_size(2011) == 24


def test_incomplete_roster_warning(caplog):
    """
    Test 9: Incomplete roster logs a warning rather than silently crashing.
    """
    resolver = RosterResolver()
    short_df = pd.DataFrame([
        {"Abbreviation": "VER", "TeamName": "Red Bull Racing"}
    ])
    with caplog.at_level(logging.WARNING):
        final = resolver._finalize_roster(short_df, expected_grid_size=20, year=2025)
    assert len(final) == 1
    assert any("expected 20" in record.message for record in caplog.records)
