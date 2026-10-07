"""
Unit Tests for Starting Grid and Penalty Logic
==============================================
Verifies FIA grid drop rules, back-of-grid relegation, and starting order guarantees.
"""

import pytest
import pandas as pd
import numpy as np

from src.features.grid_features import apply_grid_penalties


def test_no_penalties_preserves_order():
    df = pd.DataFrame({
        "Abbreviation": ["VER", "HAM", "LEC", "NOR"],
        "Position": [1, 2, 3, 4]
    })
    grid = apply_grid_penalties(df, penalties={}, expected_grid_size=4)
    assert grid["ActualStartingGrid"].tolist() == [1, 2, 3, 4]
    assert grid["Abbreviation"].tolist() == ["VER", "HAM", "LEC", "NOR"]


def test_sepang_penalty_scenario():
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
    # Qualifying positions 1 to 22
    df = pd.DataFrame({
        "Abbreviation": drivers,
        "Position": list(range(1, 23))
    })

    penalties = {"HAD": 5, "COL": 15, "LIN": 30}
    grid = apply_grid_penalties(df, penalties=penalties, expected_grid_size=22)

    # 1. Guaranteed unique 1..22
    assert sorted(grid["ActualStartingGrid"].tolist()) == list(range(1, 23))
    assert grid["Abbreviation"].nunique() == 22

    # 2. Check HAD starting position: P8
    had_row = grid[grid["Abbreviation"] == "HAD"]
    assert int(had_row.iloc[0]["ActualStartingGrid"]) == 8

    # 3. Check back of grid: COL and LIN at P21 and P22
    col_row = grid[grid["Abbreviation"] == "COL"]
    lin_row = grid[grid["Abbreviation"] == "LIN"]
    assert int(col_row.iloc[0]["ActualStartingGrid"]) == 21
    assert int(lin_row.iloc[0]["ActualStartingGrid"]) == 22

    # 4. Check front runners shifted up:
    # VER (P1), HAM (P2)
    # ANT (was Q4) -> P3
    # LEC (was Q5) -> P4
    # NOR (was Q6) -> P5
    # PIA (was Q7) -> P6
    # RUS (was Q8) -> P7
    assert grid.iloc[0]["Abbreviation"] == "VER"
    assert grid.iloc[1]["Abbreviation"] == "HAM"
    assert grid.iloc[2]["Abbreviation"] == "ANT"
    assert grid.iloc[3]["Abbreviation"] == "LEC"
    assert grid.iloc[4]["Abbreviation"] == "NOR"
    assert grid.iloc[5]["Abbreviation"] == "PIA"
    assert grid.iloc[6]["Abbreviation"] == "RUS"
    assert grid.iloc[7]["Abbreviation"] == "HAD"
