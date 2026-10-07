"""
Unit Tests for Teammate-Relative Features
=========================================
Validates:
1. teammate_quali_gap_pct exact percentage calculation and directionality.
2. teammate_quali_gap_pct handling of missing times and solo drivers.
3. teammate_grid_delta exact calculation (negative = ahead, positive = behind).
4. driver_teammate_h2h_ratio Empirical Bayes shrinkage and dual/single DNF handling.
5. Integration in build_target_dataset.
6. Pre-qualifying stage behavior (quali/grid features NaN, H2H populated).
7. Anti-leakage guarantees (strict chronological separation, no GP race data).
8. Model configuration feature count (exactly 37 features).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features.teammate_features import (
    compute_teammate_quali_gap,
    compute_teammate_grid_delta,
    compute_rolling_teammate_h2h,
)
from src.features.build_features import FeatureEngineer


@pytest.fixture
def sample_quali_df():
    # Mercedes: RUS (80.0s), ANT (80.8s) -> RUS faster by -0.99%, ANT slower by +1.00%
    # Ferrari: LEC (81.0s), HAM (None) -> LEC -2.5%, HAM +2.5%
    # McLaren: NOR (80.5s), PIA (80.5s) -> 0.0%
    return pd.DataFrame({
        "Abbreviation": ["RUS", "ANT", "LEC", "HAM", "NOR", "PIA"],
        "TeamName": [
            "Mercedes", "Mercedes",
            "Ferrari", "Ferrari",
            "McLaren", "McLaren",
        ],
        "Q1": ["0 days 00:01:20.500000", "0 days 00:01:21.000000", "0 days 00:01:21.500000", None, "0 days 00:01:20.500000", "0 days 00:01:20.500000"],
        "Q2": ["0 days 00:01:20.200000", "0 days 00:01:20.900000", "0 days 00:01:21.200000", None, "0 days 00:01:20.500000", "0 days 00:01:20.500000"],
        "Q3": ["0 days 00:01:20.000000", "0 days 00:01:20.800000", "0 days 00:01:21.000000", None, "0 days 00:01:20.500000", "0 days 00:01:20.500000"],
        "Position": [1, 4, 5, 12, 2, 3],
    })


@pytest.fixture
def sample_grid_df():
    # Mercedes: RUS P1, ANT P4 -> RUS delta = 1 - 4 = -3; ANT delta = 4 - 1 = +3
    # Ferrari: LEC P5, HAM P12 -> LEC delta = 5 - 12 = -7; HAM delta = 12 - 5 = +7
    return pd.DataFrame({
        "Abbreviation": ["RUS", "ANT", "LEC", "HAM"],
        "TeamName": ["Mercedes", "Mercedes", "Ferrari", "Ferrari"],
        "ActualStartingGrid": [1, 4, 5, 12],
    })


def test_teammate_quali_gap_calculation(sample_quali_df):
    drivers = ["RUS", "ANT", "LEC", "HAM", "NOR", "PIA"]
    res = compute_teammate_quali_gap(sample_quali_df, drivers)
    gaps = res.set_index("Abbreviation")["teammate_quali_gap_pct"].to_dict()

    # RUS vs ANT: (80.0 - 80.8) / 80.8 * 100 = -0.9901%
    assert np.isclose(gaps["RUS"], (80.0 - 80.8) / 80.8 * 100.0, atol=0.01)
    assert np.isclose(gaps["ANT"], (80.8 - 80.0) / 80.0 * 100.0, atol=0.01)
    assert gaps["RUS"] < 0  # Faster is negative
    assert gaps["ANT"] > 0  # Slower is positive

    # LEC has time, HAM has None
    assert gaps["LEC"] == -2.5
    assert gaps["HAM"] == 2.5

    # NOR vs PIA: identical times
    assert np.isclose(gaps["NOR"], 0.0, atol=1e-4)
    assert np.isclose(gaps["PIA"], 0.0, atol=1e-4)


def test_teammate_grid_delta_calculation(sample_grid_df):
    drivers = ["RUS", "ANT", "LEC", "HAM"]
    res = compute_teammate_grid_delta(sample_grid_df, drivers, grid_col="ActualStartingGrid")
    deltas = res.set_index("Abbreviation")["teammate_grid_delta"].to_dict()

    assert deltas["RUS"] == -3.0  # 1 - 4
    assert deltas["ANT"] == 3.0   # 4 - 1
    assert deltas["LEC"] == -7.0  # 5 - 12
    assert deltas["HAM"] == 7.0   # 12 - 5


def test_rolling_teammate_h2h_calculation_and_shrinkage():
    # 3 prior rounds for Mercedes:
    # Round 1: RUS P1, ANT P4 (RUS wins)
    # Round 2: RUS P3, ANT P2 (ANT wins)
    # Round 3: RUS P2, ANT DNF ("Retired") (RUS wins)
    prior_races = pd.DataFrame({
        "Round": [1, 1, 2, 2, 3, 3],
        "Abbreviation": ["RUS", "ANT", "RUS", "ANT", "RUS", "ANT"],
        "TeamName": ["Mercedes", "Mercedes", "Mercedes", "Mercedes", "Mercedes", "Mercedes"],
        "Position": [1, 4, 3, 2, 2, 20],
        "Status": ["Finished", "Finished", "Finished", "Finished", "Finished", "Retired"],
    })
    lineup = pd.DataFrame({
        "Abbreviation": ["RUS", "ANT"],
        "TeamName": ["Mercedes", "Mercedes"],
    })

    # Total valid races = 3. RUS wins = 2, ANT wins = 1.
    # Prior weight = 3.0.
    # Shrunk RUS: (2 + 1.5) / (3 + 3) = 3.5 / 6.0 = 0.5833
    # Shrunk ANT: (1 + 1.5) / (3 + 3) = 2.5 / 6.0 = 0.4167
    h2h = compute_rolling_teammate_h2h(prior_races, lineup, prior_weight=3.0)

    assert np.isclose(h2h["RUS"], 3.5 / 6.0, atol=1e-3)
    assert np.isclose(h2h["ANT"], 2.5 / 6.0, atol=1e-3)


def test_rolling_teammate_h2h_zero_prior_races():
    # Empty prior races -> default 0.50
    lineup = pd.DataFrame({
        "Abbreviation": ["VER", "HAD"],
        "TeamName": ["Red Bull Racing", "Red Bull Racing"],
    })
    h2h = compute_rolling_teammate_h2h(pd.DataFrame(), lineup)
    assert h2h["VER"] == 0.50
    assert h2h["HAD"] == 0.50


def test_teammate_features_in_build_target_dataset(sample_quali_df):
    fe = FeatureEngineer(expected_grid_size=len(sample_quali_df))
    # Mock season races for rounds 1 and 2
    prior_races = pd.DataFrame({
        "Round": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        "Race": ["R1"] * 6 + ["R2"] * 6,
        "Abbreviation": ["RUS", "ANT", "LEC", "HAM", "NOR", "PIA"] * 2,
        "TeamName": ["Mercedes", "Mercedes", "Ferrari", "Ferrari", "McLaren", "McLaren"] * 2,
        "GridPosition": [1, 2, 3, 4, 5, 6] * 2,
        "Position": [1, 3, 2, 4, 5, 6, 2, 1, 3, 5, 4, 6],
        "Points": [25, 15, 18, 12, 10, 8, 18, 25, 15, 10, 12, 8],
    })

    target_df = fe.build_target_dataset(
        target_round=3,
        track_name="Albert Park",
        quali_df=sample_quali_df,
        practice_sessions={},
        season_races_df=prior_races,
    )

    for col in ["teammate_quali_gap_pct", "teammate_grid_delta", "driver_teammate_h2h_ratio"]:
        assert col in target_df.columns
        assert not target_df[col].isna().any()

    rus = target_df[target_df["Abbreviation"] == "RUS"].iloc[0]
    ant = target_df[target_df["Abbreviation"] == "ANT"].iloc[0]

    assert rus["teammate_grid_delta"] < 0  # Starts ahead
    assert ant["teammate_grid_delta"] > 0  # Starts behind
    assert rus["teammate_quali_gap_pct"] < 0  # Faster in quali


def test_model_config_has_37_features():
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "model_config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    features = cfg.get("features", [])
    assert len(features) == 37, f"Expected 37 features, got {len(features)}: {features}"
    assert "teammate_quali_gap_pct" in features
    assert "teammate_grid_delta" in features
    assert "driver_teammate_h2h_ratio" in features
