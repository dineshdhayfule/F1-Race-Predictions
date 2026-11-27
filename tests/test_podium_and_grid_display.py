"""
Unit tests for Phase 7:
1. Separate ExpectedFinish ranking from probability-based podium showcase.
2. Provisional grid position visibility and formatting (~P<number>).
3. Preservation of underlying Monte Carlo ExpectedFinish rankings and DataFrame immutability.
"""

import numpy as np
import pandas as pd
import pytest

from apexgrid.cli.menu import format_grid_display, select_probability_podium
from apexgrid.simulation.monte_carlo import MonteCarloSimulator


# =====================================================================
# 1. Tests for select_probability_podium
# =====================================================================

def test_select_probability_podium_orders_by_win_and_podium():
    """
    Test that podium selection orders by WinProbability desc,
    then Top3Probability desc, then ExpectedFinish asc.
    """
    # Create sample DataFrame where ExpectedFinish order differs from WinProbability order
    data = {
        "Abbreviation": ["NOR", "ANT", "VER", "PIA", "RUS"],
        "TeamName": ["McLaren", "Mercedes", "Red Bull", "McLaren", "Mercedes"],
        "ExpectedFinish": [2.10, 2.50, 2.80, 4.00, 5.00],
        "PredictedPosition": [1, 2, 3, 4, 5],
        "WinProbability": [25.0, 35.0, 30.0, 5.0, 5.0],
        "Top3Probability": [80.0, 70.0, 75.0, 40.0, 30.0],
    }
    df = pd.DataFrame(data)
    df_copy = df.copy()

    podium = select_probability_podium(df)
    assert len(podium) == 3

    # Win% descending: ANT (35%), VER (30%), NOR (25%)
    assert podium[0]["Abbreviation"] == "ANT"
    assert podium[1]["Abbreviation"] == "VER"
    assert podium[2]["Abbreviation"] == "NOR"

    # Verify input DataFrame was NOT mutated or reordered
    pd.testing.assert_frame_equal(df, df_copy)
    assert df["Abbreviation"].tolist() == ["NOR", "ANT", "VER", "PIA", "RUS"]
    assert df["ExpectedFinish"].tolist() == [2.10, 2.50, 2.80, 4.00, 5.00]


def test_select_probability_podium_tie_breaking():
    """
    Test tie-breaking when WinProbability is identical:
    Tie-break 1: Top3Probability descending
    Tie-break 2: ExpectedFinish ascending
    """
    # Case A: Same Win%, different Top3%
    df_tie_win = pd.DataFrame({
        "Abbreviation": ["A", "B", "C"],
        "ExpectedFinish": [3.0, 2.0, 1.0],
        "WinProbability": [20.0, 20.0, 20.0],
        "Top3Probability": [60.0, 75.0, 70.0],
    })
    podium_a = select_probability_podium(df_tie_win)
    # B (75%), C (70%), A (60%)
    assert [d["Abbreviation"] for d in podium_a] == ["B", "C", "A"]

    # Case B: Same Win%, same Top3%, different ExpectedFinish
    df_tie_both = pd.DataFrame({
        "Abbreviation": ["X", "Y", "Z"],
        "ExpectedFinish": [4.5, 2.3, 3.1],
        "WinProbability": [10.0, 10.0, 10.0],
        "Top3Probability": [50.0, 50.0, 50.0],
    })
    podium_b = select_probability_podium(df_tie_both)
    # ExpectedFinish ascending: Y (2.3), Z (3.1), X (4.5)
    assert [d["Abbreviation"] for d in podium_b] == ["Y", "Z", "X"]


def test_select_probability_podium_edge_cases():
    """
    Test edge cases: empty DataFrame, fewer than 3 drivers, NaNs in columns, missing columns.
    """
    # Empty
    assert select_probability_podium(pd.DataFrame()) == []
    assert select_probability_podium(None) == []

    # Single driver
    df_one = pd.DataFrame({
        "Abbreviation": ["HAM"],
        "ExpectedFinish": [1.0],
        "WinProbability": [100.0],
        "Top3Probability": [100.0],
    })
    podium_one = select_probability_podium(df_one)
    assert len(podium_one) == 1
    assert podium_one[0]["Abbreviation"] == "HAM"

    # NaNs in probabilities
    df_nans = pd.DataFrame({
        "Abbreviation": ["A", "B", "C"],
        "ExpectedFinish": [1.0, 2.0, 3.0],
        "WinProbability": [np.nan, 20.0, np.nan],
        "Top3Probability": [50.0, np.nan, 40.0],
    })
    podium_nans = select_probability_podium(df_nans)
    assert len(podium_nans) == 3
    # B has 20% win (others 0.0), so B is P1
    assert podium_nans[0]["Abbreviation"] == "B"

    # Missing probability columns entirely
    df_missing = pd.DataFrame({
        "Abbreviation": ["A", "B", "C"],
        "ExpectedFinish": [3.0, 1.0, 2.0],
    })
    podium_missing = select_probability_podium(df_missing)
    assert len(podium_missing) == 3
    # Falls back to ExpectedFinish ascending: B, C, A
    assert [d["Abbreviation"] for d in podium_missing] == ["B", "C", "A"]


# =====================================================================
# 2. Tests for format_grid_display
# =====================================================================

def test_format_grid_display_priority():
    """
    Test grid display formatting:
    1) Pit lane start -> 'PL'
    2) Official ActualStartingGrid -> 'P<num>'
    3) Provisional grid -> '~P<num>'
    4) None -> '-'
    """
    # 1. Pit lane start takes precedence
    assert format_grid_display(grid_raw=5, prov_raw=4, raw_pit=1) == "PL"
    assert format_grid_display(grid_raw=np.nan, prov_raw=10, raw_pit=True) == "PL"

    # 2. Official grid available
    assert format_grid_display(grid_raw=1, prov_raw=3, raw_pit=0) == "P1"
    assert format_grid_display(grid_raw=14.0, prov_raw=np.nan, raw_pit=0) == "P14"

    # 3. Provisional grid only (pre-qualifying stages)
    assert format_grid_display(grid_raw=np.nan, prov_raw=2, raw_pit=0) == "~P2"
    assert format_grid_display(grid_raw=None, prov_raw=11.0, raw_pit=0) == "~P11"

    # 4. Neither available
    assert format_grid_display(grid_raw=np.nan, prov_raw=np.nan, raw_pit=0) == "-"
    assert format_grid_display(grid_raw=None, prov_raw=None, raw_pit=None) == "-"


# =====================================================================
# 3. Tests for Monte Carlo preservation of provisional grid position
# =====================================================================

def test_monte_carlo_preserves_provisional_grid():
    """
    Verify that MonteCarloSimulator preserves provisional_grid_position
    in the output predictions DataFrame without mutating official grid.
    """
    cfg = {
        "random_seed": 42,
        "monte_carlo": {
            "default_simulations": 500,
            "fallback_sigma": 2.85,
            "fallback_dnf_prob": 0.158,
        },
    }
    mc = MonteCarloSimulator(cfg)

    target_df = pd.DataFrame([
        {
            "Abbreviation": "VER",
            "TeamName": "Red Bull",
            "provisional_grid_position": 2,
            "simulation_start_rank": 2,
            "StartingGridStatus": "CLASSIFIED",
            "is_pit_lane_start": 0,
        },
        {
            "Abbreviation": "NOR",
            "TeamName": "McLaren",
            "provisional_grid_position": 1,
            "simulation_start_rank": 1,
            "StartingGridStatus": "CLASSIFIED",
            "is_pit_lane_start": 0,
        },
        {
            "Abbreviation": "BEA",
            "TeamName": "Haas",
            "provisional_grid_position": 18,
            "simulation_start_rank": 18,
            "StartingGridStatus": "DNS",
            "is_pit_lane_start": 0,
        },
    ])
    scores = np.array([1.5, 2.0, 18.0])

    results, summary = mc.simulate(
        target_df=target_df,
        deterministic_scores=scores,
        n_simulations=500,
        circuit_key="bahrain",
    )

    # Predictions DataFrame checks
    assert "provisional_grid_position" in results.columns
    assert "ActualStartingGrid" in results.columns

    # Active drivers have their provisional grid preserved
    ver_row = results[results["Abbreviation"] == "VER"].iloc[0]
    nor_row = results[results["Abbreviation"] == "NOR"].iloc[0]
    bea_row = results[results["Abbreviation"] == "BEA"].iloc[0]

    assert ver_row["provisional_grid_position"] == 2
    assert nor_row["provisional_grid_position"] == 1
    assert bea_row["provisional_grid_position"] == 18

    # ActualStartingGrid should remain NaN because official qualifying hasn't run
    assert pd.isna(ver_row["ActualStartingGrid"])
    assert pd.isna(nor_row["ActualStartingGrid"])

    # Main ranking is ordered by ExpectedFinish ascending
    active_preds = results.dropna(subset=["ExpectedFinish"])
    assert active_preds["ExpectedFinish"].is_monotonic_increasing
    assert active_preds["PredictedPosition"].tolist() == list(range(1, len(active_preds) + 1))
