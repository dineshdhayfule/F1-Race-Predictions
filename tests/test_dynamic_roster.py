import logging

import pandas as pd

from apexgrid.data.roster_resolver import RosterResolver
from apexgrid.features.build_features import FeatureEngineer


def test_previous_race_dns_repairs_team_slot():
    resolver = RosterResolver()
    races = pd.DataFrame([
        {"Round": 19, "Abbreviation": "VER", "TeamName": "Red Bull Racing", "Position": 1},
        {"Round": 19, "Abbreviation": "BEA", "TeamName": "Haas F1 Team", "Position": 18},
    ])
    qualis = pd.DataFrame([
        {"Round": 19, "Abbreviation": "VER", "TeamName": "Red Bull Racing", "Position": 1},
        {"Round": 19, "Abbreviation": "BEA", "TeamName": "Haas F1 Team", "Position": 14},
        {"Round": 19, "Abbreviation": "HIR", "TeamName": "Haas F1 Team", "Position": 18},
    ])
    roster = resolver.resolve_roster(
        year=2026, round_num=20, stage="pre_fp1",
        season_races_df=races, season_qualis_df=qualis, expected_grid_size=22,
    )
    haas = roster[roster["TeamName"] == "Haas F1 Team"]["Abbreviation"].tolist()
    assert len(haas) == 2
    assert "BEA" in haas


def test_official_entry_list_excludes_previous_reserve():
    resolver = RosterResolver()
    entry_list = pd.DataFrame([
        {"Abbreviation": "BEA", "TeamName": "Haas F1 Team", "FullName": "Oliver Bearman"},
        {"Abbreviation": "OCO", "TeamName": "Haas F1 Team", "FullName": "Esteban Ocon"},
    ] + [
        {"Abbreviation": f"D{i}", "TeamName": f"Team{i//2}", "FullName": f"Driver {i}"}
        for i in range(2, 22)
    ])
    roster = resolver.resolve_roster(
        year=2026, round_num=20, stage="pre_fp1",
        entry_list_df=entry_list, expected_grid_size=22,
    )
    assert "OCO" in roster["Abbreviation"].values
    assert "HIR" not in roster["Abbreviation"].values


def test_latest_practice_session_excludes_fp1_only_reserve():
    resolver = RosterResolver()
    fp1 = ["BEG", "ALB", "ANT", "BOR", "BOT", "CAM", "COL", "CRA", "GAS",
           "HAD", "HIR", "HUL", "LEC", "PER", "RUS", "SAI", "VER"]
    fp2 = ["ALB", "ALO", "ANT", "BEA", "BOR", "BOT", "CAM", "COL", "GAS",
           "HAD", "HAM", "HUL", "LAW", "LEC", "LIN", "NOR", "PER", "PIA",
           "RUS", "SAI", "STR", "VER"]
    sessions = {
        "fp1": pd.DataFrame([{"Abbreviation": d, "TeamName": "Test Team"} for d in fp1]),
        "fp2": pd.DataFrame([{"Abbreviation": d, "TeamName": "Test Team"} for d in fp2]),
    }
    roster = resolver.resolve_roster(
        year=2026, round_num=21, stage="after_fp2",
        weekend_sessions=sessions, expected_grid_size=22,
    )
    assert len(roster) == 22
    assert "BEG" not in roster["Abbreviation"].values
    assert "ALO" in roster["Abbreviation"].values


def test_confirmed_replacement_is_retained_from_practice():
    resolver = RosterResolver()
    sessions = {"fp1": pd.DataFrame([
        {"Abbreviation": "BEA", "TeamName": "Haas F1 Team", "DriverNumber": 87},
        {"Abbreviation": "HIR", "TeamName": "Haas F1 Team", "DriverNumber": 98},
    ] + [
        {"Abbreviation": f"D{i}", "TeamName": f"Team{i//2}", "DriverNumber": i}
        for i in range(2, 22)
    ])}
    roster = resolver.resolve_roster(
        year=2026, round_num=20, stage="after_fp1",
        weekend_sessions=sessions, expected_grid_size=22,
    )
    assert "HIR" in roster["Abbreviation"].values
    assert roster.loc[roster["Abbreviation"] == "HIR", "Source"].iloc[0] == "practice_fp1"


def test_mid_season_replacement_is_present():
    resolver = RosterResolver()
    qualis = pd.DataFrame([
        {"Abbreviation": "LAW", "TeamName": "Racing Bulls", "Position": 8},
        {"Abbreviation": "LIN", "TeamName": "Racing Bulls", "Position": 9},
    ] + [
        {"Abbreviation": f"D{i}", "TeamName": f"Team{i//2}", "Position": i + 2}
        for i in range(2, 22)
    ])
    roster = resolver.resolve_roster(
        year=2026, round_num=10, stage="after_qualifying",
        quali_df=qualis, expected_grid_size=22,
    )
    assert "LAW" in roster["Abbreviation"].values
    assert "LIN" in roster["Abbreviation"].values


def test_withdrawal_status_is_preserved():
    resolver = RosterResolver()
    grid = pd.DataFrame([
        {"Abbreviation": "SAI", "TeamName": "Williams", "Position": 1, "Status": "active"},
        {"Abbreviation": "ALB", "TeamName": "Williams", "Position": 2, "Status": "withdrawn"},
    ])
    roster = resolver._extract_from_dataframe(grid, "starting_grid", "verified", 2026)
    assert roster.loc[roster["Abbreviation"] == "ALB", "EntryStatus"].iloc[0] == "withdrawn"


def test_replacement_without_history_remains_in_target_data():
    engineer = FeatureEngineer(expected_grid_size=2)
    qualis = pd.DataFrame([
        {"Abbreviation": "VER", "TeamName": "Red Bull Racing", "Position": 1},
        {"Abbreviation": "NEW", "TeamName": "Red Bull Racing", "Position": 2},
    ])
    prior = pd.DataFrame([
        {"Round": 1, "Abbreviation": "VER", "TeamName": "Red Bull Racing",
         "GridPosition": 1, "Position": 1, "ActualFinish": 1, "Status": "Finished"},
        {"Round": 1, "Abbreviation": "HAD", "TeamName": "Red Bull Racing",
         "GridPosition": 2, "Position": 2, "ActualFinish": 2, "Status": "Finished"},
    ])
    target = engineer.build_target_dataset(
        target_round=2, track_name="Test GP", quali_df=qualis,
        practice_sessions={}, season_races_df=prior, expected_grid_size=2,
    )
    assert "NEW" in target["Abbreviation"].values
    row = target[target["Abbreviation"] == "NEW"].iloc[0]
    assert pd.notna(row["GridPosition"])
    assert pd.notna(row["team_finish_ewma"])


def test_historical_grid_sizes():
    resolver = RosterResolver()
    assert resolver.get_expected_grid_size(2025) == 20
    assert resolver.get_expected_grid_size(2026) == 22
    assert resolver.get_expected_grid_size(2011) == 24


def test_incomplete_roster_logs_warning(caplog):
    resolver = RosterResolver()
    short = pd.DataFrame([{"Abbreviation": "VER", "TeamName": "Red Bull Racing"}])
    with caplog.at_level(logging.WARNING):
        final = resolver._finalize_roster(short, expected_grid_size=20, year=2025)
    assert len(final) == 1
    assert any("expected 20" in record.message for record in caplog.records)


def test_partial_practice_session_retains_confirmed_drivers():
    resolver = RosterResolver()
    # 17 confirmed drivers participated in FP1 (e.g. red-flagged practice)
    confirmed_17 = [
        "VER", "NOR", "LEC", "HAM", "RUS", "PIA", "SAI", "ALO",
        "GAS", "OCO", "ALB", "TSU", "STR", "HUL", "BOT", "ZHO", "MAG"
    ]
    # Prior round season data contains full 22-driver grid
    prior_22 = confirmed_17 + ["RIC", "LAW", "SAR", "BEA", "DOO"]
    prior_races = pd.DataFrame([
        {"Round": 20, "Abbreviation": d, "TeamName": f"Team_{i//2}", "Position": i + 1}
        for i, d in enumerate(prior_22)
    ])
    prior_qualis = pd.DataFrame([
        {"Round": 20, "Abbreviation": d, "TeamName": f"Team_{i//2}", "Position": i + 1}
        for i, d in enumerate(prior_22)
    ])
    sessions = {
        "fp1": pd.DataFrame([{"Abbreviation": d, "TeamName": f"Team_{i//2}"} for i, d in enumerate(confirmed_17)]),
    }
    roster = resolver.resolve_roster(
        year=2026,
        round_num=21,
        stage="after_fp1",
        weekend_sessions=sessions,
        season_races_df=prior_races,
        season_qualis_df=prior_qualis,
        expected_grid_size=22,
    )
    # Roster retains confirmed practice drivers and supplements up to expected_grid_size
    assert len(roster) == 22
    for d in confirmed_17:
        assert d in roster["Abbreviation"].values
        assert roster.loc[roster["Abbreviation"] == d, "Source"].values[0] == "practice_fp1"
    # Unconfirmed prior-round drivers are added from season pool to reach expected grid size
    for unconfirmed in ["RIC", "LAW", "SAR", "BEA", "DOO"]:
        assert unconfirmed in roster["Abbreviation"].values
        assert roster.loc[roster["Abbreviation"] == unconfirmed, "Source"].values[0] == "season_pool"
    assert "catalog_baseline" not in roster["Source"].values


def test_unresolved_roster_returns_empty_without_catalog_invention():
    resolver = RosterResolver()
    roster = resolver.resolve_roster(
        year=2026,
        round_num=1,
        stage="pre_fp1",
        expected_grid_size=22,
    )
    assert roster.empty
    assert "catalog_baseline" not in roster.get("Source", pd.Series(dtype=str)).values
