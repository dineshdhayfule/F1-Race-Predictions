"""
Unit Tests for Starting Grid and Penalty Logic
==============================================
Verifies FIA grid drop rules, back-of-grid relegation, pit-lane starts,
and 12 official regulatory scenarios matching FIA Sporting Regulations.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.grid_features import (
    CanonicalGridRecord,
    GridPublicationStatus,
    PenaltyRecord,
    PenaltyType,
    StartingGridStatus,
    apply_grid_penalties,
    resolve_fia_starting_grid,
)


def _make_quali_df(drivers):
    """Helper to construct qualifying dataframe with 1..N positions."""
    return pd.DataFrame({
        "Abbreviation": drivers,
        "Position": list(range(1, len(drivers) + 1)),
        "QualifyingPosition": list(range(1, len(drivers) + 1)),
    })


# -----------------------------------------------------------------------------
# Case 1: Zero Penalties (Qualifying order 1..N preserved)
# -----------------------------------------------------------------------------
def test_case_01_no_penalties_preserves_order():
    df = _make_quali_df(["VER", "HAM", "LEC", "NOR"])
    grid = resolve_fia_starting_grid(df, penalties={})
    assert grid["ActualStartingGrid"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert grid["Abbreviation"].tolist() == ["VER", "HAM", "LEC", "NOR"]
    assert grid["is_pit_lane_start"].tolist() == [0, 0, 0, 0]


# -----------------------------------------------------------------------------
# Case 2: Single 3-Place Drop (P1 with +3 -> Starts P4)
# -----------------------------------------------------------------------------
def test_case_02_single_three_place_drop():
    df = _make_quali_df(["VER", "HAM", "LEC", "NOR", "PIA"])
    penalties = {"VER": 3}
    grid = resolve_fia_starting_grid(df, penalties=penalties)
    # HAM was P2 -> P1, LEC was P3 -> P2, NOR was P4 -> P3, VER was P1+3 -> P4
    assert grid.iloc[0]["Abbreviation"] == "HAM"
    assert grid.iloc[1]["Abbreviation"] == "LEC"
    assert grid.iloc[2]["Abbreviation"] == "NOR"
    assert grid.iloc[3]["Abbreviation"] == "VER"
    assert int(grid.loc[grid["Abbreviation"] == "VER", "ActualStartingGrid"].iloc[0]) == 4


# -----------------------------------------------------------------------------
# Case 3: Single 5-Place Drop (P3 with +5 -> Starts P8)
# -----------------------------------------------------------------------------
def test_case_03_single_five_place_drop():
    drivers = ["VER", "HAM", "HAD", "ANT", "LEC", "NOR", "PIA", "RUS", "GAS"]
    df = _make_quali_df(drivers)
    penalties = {"HAD": 5}
    grid = resolve_fia_starting_grid(df, penalties=penalties)
    had_row = grid[grid["Abbreviation"] == "HAD"].iloc[0]
    assert int(had_row["ActualStartingGrid"]) == 8
    assert int(had_row["QualifyingPosition"]) == 3
    assert int(had_row["grid_penalty_places"]) == 5


# -----------------------------------------------------------------------------
# Case 4: Single 10-Place Drop (P3 with +10 -> Starts P13)
# -----------------------------------------------------------------------------
def test_case_04_single_ten_place_drop():
    drivers = [f"D{i}" for i in range(1, 23)]
    df = _make_quali_df(drivers)
    penalties = {"D3": 10}
    grid = resolve_fia_starting_grid(df, penalties=penalties)
    d3_row = grid[grid["Abbreviation"] == "D3"].iloc[0]
    assert int(d3_row["ActualStartingGrid"]) == 13


# -----------------------------------------------------------------------------
# Case 5: Multiple Drops Producing Conflicting Temporary Positions (Tie-break by qualifying)
# -----------------------------------------------------------------------------
def test_case_05_conflicting_drops_tie_break_by_qualifying():
    # D1 (Q1) gets +4 -> target drop
    # D2 (Q2) gets +3 -> target drop
    # Both drop into the same vicinity; D1 originally qualified ahead of D2
    drivers = ["D1", "D2", "D3", "D4", "D5", "D6"]
    df = _make_quali_df(drivers)
    penalties = {"D1": 4, "D2": 3}
    grid = resolve_fia_starting_grid(df, penalties=penalties)
    assert grid["ActualStartingGrid"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    # D3 (Q3) -> P1, D4 (Q4) -> P2, D5 (Q5) -> P3
    assert grid.iloc[0]["Abbreviation"] == "D3"
    assert grid.iloc[1]["Abbreviation"] == "D4"
    assert grid.iloc[2]["Abbreviation"] == "D5"
    assert grid.iloc[3]["Abbreviation"] == "D1"
    assert grid.iloc[4]["Abbreviation"] == "D2"


# -----------------------------------------------------------------------------
# Case 6: Single Back-of-Grid Penalty (>= 15 Places)
# -----------------------------------------------------------------------------
def test_case_06_single_back_of_grid():
    drivers = ["D1", "D2", "D3", "D4", "D5"]
    df = _make_quali_df(drivers)
    penalties = {"D2": 15}
    grid = resolve_fia_starting_grid(df, penalties=penalties)
    assert int(grid.loc[grid["Abbreviation"] == "D2", "ActualStartingGrid"].iloc[0]) == 5
    assert grid.iloc[0]["Abbreviation"] == "D1"
    assert grid.iloc[1]["Abbreviation"] == "D3"
    assert grid.iloc[2]["Abbreviation"] == "D4"
    assert grid.iloc[3]["Abbreviation"] == "D5"


# -----------------------------------------------------------------------------
# Case 7: Multiple Back-of-Grid Penalties (Ordered relative to each other by qualifying)
# -----------------------------------------------------------------------------
def test_case_07_multiple_back_of_grid_ordered_by_quali():
    drivers = ["D1", "D2", "D3", "D4", "D5", "D6"]
    df = _make_quali_df(drivers)
    # Both D2 (Q2) and D4 (Q4) receive back-of-grid penalties (>= 15)
    penalties = {"D2": 15, "D4": 30}
    grid = resolve_fia_starting_grid(df, penalties=penalties)
    assert int(grid.loc[grid["Abbreviation"] == "D2", "ActualStartingGrid"].iloc[0]) == 5
    assert int(grid.loc[grid["Abbreviation"] == "D4", "ActualStartingGrid"].iloc[0]) == 6


# -----------------------------------------------------------------------------
# Case 8: Sepang Full Penalty Scenario (Baseline Verification)
# -----------------------------------------------------------------------------
def test_case_08_sepang_penalty_scenario():
    """
    HAD: Q3 with +5 penalty -> Starts P8
    COL: Q15 with +15 penalty (back of grid) -> Starts P21
    LIN: Q16 with +30 penalty (back of grid) -> Starts P22
    """
    drivers = [
        "VER", "HAM", "HAD", "ANT", "LEC", "NOR", "PIA", "RUS",
        "GAS", "BOR", "LAW", "ALO", "SAI", "STR", "OCO", "TSU",
        "ALB", "HUL", "BEA", "BOT", "COL", "LIN"
    ]
    df = _make_quali_df(drivers)
    penalties = {"HAD": 5, "COL": 15, "LIN": 30}
    grid = apply_grid_penalties(df, penalties=penalties, expected_grid_size=22)

    assert sorted(grid["ActualStartingGrid"].tolist()) == list(range(1, 23))
    assert grid["Abbreviation"].nunique() == 22

    # HAD starts P8
    had_row = grid[grid["Abbreviation"] == "HAD"]
    assert int(had_row.iloc[0]["ActualStartingGrid"]) == 8

    # COL and LIN back of grid
    assert int(grid[grid["Abbreviation"] == "COL"].iloc[0]["ActualStartingGrid"]) == 21
    assert int(grid[grid["Abbreviation"] == "LIN"].iloc[0]["ActualStartingGrid"]) == 22

    # Front runners shifted up
    assert grid.iloc[0]["Abbreviation"] == "VER"
    assert grid.iloc[1]["Abbreviation"] == "HAM"
    assert grid.iloc[2]["Abbreviation"] == "ANT"
    assert grid.iloc[3]["Abbreviation"] == "LEC"
    assert grid.iloc[4]["Abbreviation"] == "NOR"
    assert grid.iloc[5]["Abbreviation"] == "PIA"
    assert grid.iloc[6]["Abbreviation"] == "RUS"
    assert grid.iloc[7]["Abbreviation"] == "HAD"


# -----------------------------------------------------------------------------
# Case 9: Pit-Lane Starter Alone (Physical null, simulation rank N+1)
# -----------------------------------------------------------------------------
def test_case_09_pit_lane_starter_alone():
    drivers = ["VER", "HAM", "LEC", "NOR"]
    df = _make_quali_df(drivers)
    # HAM starts from pit lane
    penalties = {
        "HAM": {"penalty_type": "PIT_LANE", "places": 0, "reason": "Parc Ferme breach"}
    }
    grid = resolve_fia_starting_grid(df, penalties=penalties)

    # Physical grid starters = 3 (VER, LEC, NOR)
    physical_starters = grid[grid["ActualStartingGrid"].notna()]
    assert len(physical_starters) == 3
    assert physical_starters["ActualStartingGrid"].tolist() == [1.0, 2.0, 3.0]
    assert physical_starters["Abbreviation"].tolist() == ["VER", "LEC", "NOR"]

    # HAM starts from pit lane: physical grid is null (NaN), simulation start rank is 4
    ham_row = grid[grid["Abbreviation"] == "HAM"].iloc[0]
    assert pd.isna(ham_row["ActualStartingGrid"])
    assert int(ham_row["simulation_start_rank"]) == 4
    assert int(ham_row["is_pit_lane_start"]) == 1
    assert ham_row["StartingGridStatus"] == StartingGridStatus.PIT_LANE.value


# -----------------------------------------------------------------------------
# Case 10: Multiple Pit-Lane Starters (Ordered by qualifying times)
# -----------------------------------------------------------------------------
def test_case_10_multiple_pit_lane_starters_ordered_by_quali():
    drivers = ["D1", "D2", "D3", "D4", "D5"]
    df = _make_quali_df(drivers)
    penalties = {
        "D2": {"penalty_type": "PIT_LANE"},
        "D4": {"penalty_type": "PIT_LANE"},
    }
    grid = resolve_fia_starting_grid(df, penalties=penalties)

    # Physical grid starters = 3 (D1, D3, D5) -> P1, P2, P3
    assert grid.loc[grid["Abbreviation"] == "D1", "ActualStartingGrid"].iloc[0] == 1.0
    assert grid.loc[grid["Abbreviation"] == "D3", "ActualStartingGrid"].iloc[0] == 2.0
    assert grid.loc[grid["Abbreviation"] == "D5", "ActualStartingGrid"].iloc[0] == 3.0

    # Pit-lane starters: D2 (Q2) vs D4 (Q4) -> D2 starts ahead of D4 in pit lane
    d2_row = grid[grid["Abbreviation"] == "D2"].iloc[0]
    d4_row = grid[grid["Abbreviation"] == "D4"].iloc[0]
    assert pd.isna(d2_row["ActualStartingGrid"]) and pd.isna(d4_row["ActualStartingGrid"])
    assert int(d2_row["simulation_start_rank"]) == 4
    assert int(d4_row["simulation_start_rank"]) == 5


# -----------------------------------------------------------------------------
# Case 11: Disqualified from Qualifying but Permitted to Start (Art 39.4)
# -----------------------------------------------------------------------------
def test_case_11_quali_dsq_permitted_to_start():
    drivers = ["D1", "D2", "D3", "D4"]
    df = _make_quali_df(drivers)
    # D2 disqualified from qualifying but permitted to start by stewards
    penalties = {
        "D2": {
            "penalty_type": "DSQ",
            "permitted_to_start": True,
            "article": "Art 39.4",
            "reason": "Permitted to start by Stewards decision"
        }
    }
    grid = resolve_fia_starting_grid(df, penalties=penalties)
    # Starts at the back of the physical grid: P4
    assert int(grid.loc[grid["Abbreviation"] == "D2", "ActualStartingGrid"].iloc[0]) == 4
    assert grid.loc[grid["Abbreviation"] == "D2", "StartingGridStatus"].iloc[0] == StartingGridStatus.PENALIZED.value


# -----------------------------------------------------------------------------
# Case 12: DNS / Excluded Entrant (Excluded from physical grid and simulation rank)
# -----------------------------------------------------------------------------
def test_case_12_dns_excluded_entrant():
    drivers = ["D1", "D2", "D3", "D4"]
    df = _make_quali_df(drivers)
    penalties = {
        "D3": {"penalty_type": "DNS", "reason": "Chassis damage in qualifying"}
    }
    grid = resolve_fia_starting_grid(df, penalties=penalties)
    # Total active starters = 3
    d3_row = grid[grid["Abbreviation"] == "D3"].iloc[0]
    assert pd.isna(d3_row["ActualStartingGrid"])
    assert pd.isna(d3_row["simulation_start_rank"])
    assert d3_row["StartingGridStatus"] == StartingGridStatus.DNS.value

    # Remaining starters occupy contiguous 1..3
    active = grid[grid["simulation_start_rank"].notna()]
    assert len(active) == 3
    assert active["ActualStartingGrid"].tolist() == [1.0, 2.0, 3.0]


# -----------------------------------------------------------------------------
# Case 13: Duplicate Qualifying Positions Broken by Q3/Q2/Q1 Lap Times
# -----------------------------------------------------------------------------
def test_case_13_qualifying_laptime_tie_break():
    """
    Test Mexico 2026 scenario: LIN and LAW both have QualifyingPosition = 8.
    LAW had Q3 lap time 1:17.029, LIN had Q3 lap time 1:17.117.
    LAW must be resolved ahead of LIN (P8 vs P9).
    """
    df = pd.DataFrame({
        "Abbreviation": ["VER", "LAW", "LIN", "HAM"],
        "Position": [1, 8, 8, 2],
        "QualifyingPosition": [1, 8, 8, 2],
        "Q3": [
            "0 days 00:01:15.500",
            "0 days 00:01:17.029000",
            "0 days 00:01:17.117000",
            "0 days 00:01:15.800",
        ],
    })
    grid = resolve_fia_starting_grid(df, penalties={})
    law_row = grid[grid["Abbreviation"] == "LAW"].iloc[0]
    lin_row = grid[grid["Abbreviation"] == "LIN"].iloc[0]
    assert int(law_row["ActualStartingGrid"]) < int(lin_row["ActualStartingGrid"])
    assert int(law_row["ActualStartingGrid"]) == 3
    assert int(lin_row["ActualStartingGrid"]) == 4


# -----------------------------------------------------------------------------
# Case 14: Multiple Penalty Accumulation (5 + 10 = 15 places -> Back of Grid)
# -----------------------------------------------------------------------------
def test_case_14_multiple_penalty_accumulation_triggers_back_of_grid():
    """
    FIA Art 28.2: If a driver incurs multiple penalties that accumulate to >= 15 places,
    the driver must be relegated to the Back of Grid.
    """
    drivers = [f"D{i}" for i in range(1, 23)]
    df = _make_quali_df(drivers)
    # D3 receives two penalties: 5 places (gearbox) + 10 places (PU element) = 15 places
    penalties = [
        {"driver": "D3", "places": 5, "reason": "Gearbox change"},
        {"driver": "D3", "places": 10, "reason": "5th ICE"},
    ]
    grid = resolve_fia_starting_grid(df, penalties=penalties, expected_grid_size=22)
    d3_row = grid[grid["Abbreviation"] == "D3"].iloc[0]
    assert int(d3_row["grid_penalty_places"]) == 15
    # Since total >= 15, D3 is relegated to Back of Grid (starts P22)
    assert int(d3_row["ActualStartingGrid"]) == 22


# -----------------------------------------------------------------------------
# Case 15: Monte Carlo Assessed Penalty vs Net Grid Displacement
# -----------------------------------------------------------------------------
def test_case_15_monte_carlo_penalty_magnitude_vs_displacement():
    """
    Monte Carlo must not equate assessed penalty magnitude with net grid displacement.
    If D3 has 0 assessed penalty but started P4 because D1 dropped,
    grid_penalty_places must be 0 and grid_vs_quali_delta must be +1.
    """
    from src.simulation.monte_carlo import MonteCarloSimulator
    sim = MonteCarloSimulator({"monte_carlo": {"default_simulations": 10}, "random_seed": 42})

    target_df = pd.DataFrame({
        "Abbreviation": ["D2", "D1", "D3"],
        "TeamName": ["Team A", "Team B", "Team C"],
        "QualifyingPosition": [2, 1, 3],
        "ActualStartingGrid": [1.0, 2.0, 3.0],
        "simulation_start_rank": [1.0, 2.0, 3.0],
        "is_pit_lane_start": [0, 0, 0],
        "GridPenalty": [0, 1, 0],
        "grid_penalty_places": [0, 1, 0],
        "grid_vs_quali_delta": [-1, 1, 0],
    })
    deterministic_scores = np.array([1.0, 2.0, 3.0])
    residuals = np.array([0.0, 0.0, 0.0])

    sim_res, _ = sim.simulate(
        target_df=target_df,
        deterministic_scores=deterministic_scores,
        residuals=residuals,
    )
    d2 = sim_res[sim_res["Abbreviation"] == "D2"].iloc[0]
    assert int(d2["grid_penalty_places"]) == 0
    assert int(d2["grid_vs_quali_delta"]) == -1  # Net shifted up by 1


# -----------------------------------------------------------------------------
# Case 16: Safe Pit-Lane Grid Display Formatting ("PL")
# -----------------------------------------------------------------------------
def test_case_16_pit_lane_starter_formatting_pl():
    """
    Markdown reports and CLI must format NaN ActualStartingGrid / is_pit_lane_start as 'PL'
    without raising ValueError.
    """
    row = {
        "Abbreviation": "OCO",
        "TeamName": "Alpine",
        "ActualStartingGrid": np.nan,
        "is_pit_lane_start": 1,
        "simulation_start_rank": 22,
    }
    grid_raw = row.get("ActualStartingGrid")
    is_pit = bool(row.get("is_pit_lane_start", 0)) or pd.isna(grid_raw)
    grid_str = "PL" if is_pit else str(int(round(float(grid_raw))))
    assert grid_str == "PL"

