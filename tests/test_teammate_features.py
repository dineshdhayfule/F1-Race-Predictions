"""
Unit Tests for Phase 3A: Teammate-Relative Feature Engineering
==============================================================
Validates all 26 required scenarios:
 1. Faster driver has negative percentage gap.
 2. Slower driver has positive percentage gap.
 3. Equal lap times evaluate to 0.0%.
 4. Missing qualifying lap times -> strictly NaN (Fix 1).
 5. Invalid lap times (< 50s or invalid strings) -> strictly NaN.
 6. Solo driver without teammate in qualifying -> strictly NaN.
 7. Unexpected team size (> 2 drivers) -> strictly NaN.
 8. Grid delta: starting ahead evaluates to negative integer delta.
 9. Grid delta: starting behind evaluates to positive integer delta.
10. Grid delta: tied starting position evaluates to 0.0.
11. Grid penalties correctly reflected via ActualStartingGrid.
12. Missing starting grid or unheld qualifying -> strictly NaN (Fix 2).
13. Solo driver on starting grid -> strictly NaN (Fix 2).
14. Rolling H2H: Head-to-head win counted (pos_drv < pos_tm).
15. Rolling H2H: Head-to-head loss counted (pos_drv > pos_tm).
16. Rolling H2H: Head-to-head tie gives 0.5 win.
17. Rolling H2H: Dual DNF is neutral (excluded from valid races).
18. Rolling H2H: Single DNF gives win to finishing teammate.
19. Rolling H2H: Single DNF gives loss to retiring driver.
20. Empirical Bayes shrinkage with prior M=3.0 and prior mean=0.50.
21. Bounding: Dominant win rate clipped to 0.85 (Fix 3).
22. Bounding: Inferior win rate clipped to 0.15 (Fix 3).
23. Mid-season driver changes: only matches prior races when on the same team.
24. Pre-qualifying vs post-qualifying stage behavior (quali & grid delta NaN, H2H populated).
25. Weekend compatibility: functions seamlessly on both Normal and Sprint weekend formats.
26. Anti-leakage guarantees: target and future race results are strictly excluded.
27. Model configuration has exactly 37 features.
28. Median imputer gracefully handles NaNs for training and inference.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from src.features.teammate_features import (
    compute_teammate_quali_gap,
    compute_teammate_grid_delta,
    compute_rolling_teammate_h2h,
)
from src.features.build_features import FeatureEngineer


@pytest.fixture
def clean_quali_df():
    """Qualifying session with complete lap times across 3 teams."""
    return pd.DataFrame({
        "Abbreviation": ["RUS", "ANT", "LEC", "HAM", "NOR", "PIA"],
        "TeamName": ["Mercedes", "Mercedes", "Ferrari", "Ferrari", "McLaren", "McLaren"],
        "Q1": ["0 days 00:01:20.500000", "0 days 00:01:21.000000", "0 days 00:01:20.200000", "0 days 00:01:20.400000", "0 days 00:01:20.000000", "0 days 00:01:20.000000"],
        "Q2": ["0 days 00:01:20.200000", "0 days 00:01:20.900000", "0 days 00:01:20.000000", "0 days 00:01:20.300000", "0 days 00:01:19.800000", "0 days 00:01:19.800000"],
        "Q3": ["0 days 00:01:20.000000", "0 days 00:01:20.800000", "0 days 00:01:19.800000", "0 days 00:01:20.100000", "0 days 00:01:19.500000", "0 days 00:01:19.500000"],
        "Position": [1, 4, 2, 5, 3, 3],
    })


@pytest.fixture
def sample_grid_df():
    """Starting grid with penalty adjustments reflected in ActualStartingGrid."""
    return pd.DataFrame({
        "Abbreviation": ["RUS", "ANT", "LEC", "HAM", "NOR", "PIA"],
        "TeamName": ["Mercedes", "Mercedes", "Ferrari", "Ferrari", "McLaren", "McLaren"],
        "GridPosition": [1, 4, 2, 5, 3, 6],
        "ActualStartingGrid": [1, 4, 7, 5, 3, 3],  # LEC took 5-place penalty (2 -> 7)
    })


# -------------------------------------------------------------------------
# 1-7: teammate_quali_gap_pct tests
# -------------------------------------------------------------------------
def test_teammate_quali_gap_faster_and_slower(clean_quali_df):
    """Faster driver gets negative %, slower driver gets positive %."""
    drivers = ["RUS", "ANT"]
    res = compute_teammate_quali_gap(clean_quali_df, drivers)
    gaps = res.set_index("Abbreviation")["teammate_quali_gap_pct"].to_dict()

    # RUS best: 80.0s, ANT best: 80.8s
    # RUS gap = (80.0 - 80.8) / 80.8 * 100 = -0.9901%
    # ANT gap = (80.8 - 80.0) / 80.0 * 100 = +1.0000%
    assert np.isclose(gaps["RUS"], (80.0 - 80.8) / 80.8 * 100.0, atol=1e-3)
    assert np.isclose(gaps["ANT"], (80.8 - 80.0) / 80.0 * 100.0, atol=1e-3)
    assert gaps["RUS"] < 0.0
    assert gaps["ANT"] > 0.0


def test_teammate_quali_gap_equal_times(clean_quali_df):
    """Equal best lap times evaluate to 0.0%."""
    drivers = ["NOR", "PIA"]
    res = compute_teammate_quali_gap(clean_quali_df, drivers)
    gaps = res.set_index("Abbreviation")["teammate_quali_gap_pct"].to_dict()
    assert np.isclose(gaps["NOR"], 0.0, atol=1e-4)
    assert np.isclose(gaps["PIA"], 0.0, atol=1e-4)


def test_teammate_quali_gap_missing_lap_times_evaluates_to_nan():
    """Missing qualifying time for either driver evaluates strictly to NaN (Fix 1)."""
    quali_df = pd.DataFrame({
        "Abbreviation": ["LEC", "HAM"],
        "TeamName": ["Ferrari", "Ferrari"],
        "Q1": ["0 days 00:01:21.000000", None],
        "Q2": ["0 days 00:01:20.500000", None],
        "Q3": ["0 days 00:01:20.000000", None],
    })
    res = compute_teammate_quali_gap(quali_df, ["LEC", "HAM"])
    gaps = res.set_index("Abbreviation")["teammate_quali_gap_pct"].to_dict()

    assert np.isnan(gaps["LEC"])
    assert np.isnan(gaps["HAM"])


def test_teammate_quali_gap_invalid_lap_times():
    """Unrealistic or invalid lap times (< 50s or corrupt) evaluate to NaN."""
    quali_df = pd.DataFrame({
        "Abbreviation": ["VER", "HAD"],
        "TeamName": ["Red Bull Racing", "Red Bull Racing"],
        "Q1": ["0 days 00:00:30.000000", "0 days 00:01:20.000000"],  # 30s is invalid
        "Q2": [None, None],
        "Q3": [None, None],
    })
    res = compute_teammate_quali_gap(quali_df, ["VER", "HAD"])
    gaps = res.set_index("Abbreviation")["teammate_quali_gap_pct"].to_dict()
    assert np.isnan(gaps["VER"])
    assert np.isnan(gaps["HAD"])


def test_teammate_quali_gap_solo_driver():
    """A solo driver with no teammate in qualifying evaluates to NaN."""
    quali_df = pd.DataFrame({
        "Abbreviation": ["ALB"],
        "TeamName": ["Williams"],
        "Q1": ["0 days 00:01:21.000000"],
        "Q2": ["0 days 00:01:20.500000"],
        "Q3": ["0 days 00:01:20.200000"],
    })
    res = compute_teammate_quali_gap(quali_df, ["ALB"])
    assert np.isnan(res.loc[res["Abbreviation"] == "ALB", "teammate_quali_gap_pct"].values[0])


def test_teammate_quali_gap_unexpected_team_size():
    """A team with > 2 drivers in the DataFrame evaluates gracefully to NaN."""
    quali_df = pd.DataFrame({
        "Abbreviation": ["DRV1", "DRV2", "DRV3"],
        "TeamName": ["TestTeam", "TestTeam", "TestTeam"],
        "Q1": ["0 days 00:01:20.000000", "0 days 00:01:20.500000", "0 days 00:01:21.000000"],
    })
    res = compute_teammate_quali_gap(quali_df, ["DRV1", "DRV2", "DRV3"])
    assert res["teammate_quali_gap_pct"].isna().all()


def test_teammate_quali_gap_empty_or_none():
    """Empty or None qualifying data evaluates to all NaNs without crash."""
    res_none = compute_teammate_quali_gap(None, ["RUS", "ANT"])
    assert res_none["teammate_quali_gap_pct"].isna().all()

    res_empty = compute_teammate_quali_gap(pd.DataFrame(), ["RUS", "ANT"])
    assert res_empty["teammate_quali_gap_pct"].isna().all()


# -------------------------------------------------------------------------
# 8-13: teammate_grid_delta tests
# -------------------------------------------------------------------------
def test_teammate_grid_delta_ahead_and_behind(sample_grid_df):
    """Ahead = negative delta, behind = positive delta."""
    drivers = ["RUS", "ANT"]
    res = compute_teammate_grid_delta(sample_grid_df, drivers)
    deltas = res.set_index("Abbreviation")["teammate_grid_delta"].to_dict()

    # RUS P1, ANT P4 -> RUS: 1 - 4 = -3; ANT: 4 - 1 = +3
    assert deltas["RUS"] == -3.0
    assert deltas["ANT"] == 3.0


def test_teammate_grid_delta_penalties_via_actual_starting_grid(sample_grid_df):
    """Grid penalties reflected in ActualStartingGrid are accurately used."""
    drivers = ["LEC", "HAM"]
    res = compute_teammate_grid_delta(sample_grid_df, drivers)
    deltas = res.set_index("Abbreviation")["teammate_grid_delta"].to_dict()

    # LEC ActualStartingGrid = 7 (penalized), HAM ActualStartingGrid = 5
    # LEC: 7 - 5 = +2 (starts behind teammate due to penalty)
    # HAM: 5 - 7 = -2 (starts ahead of teammate)
    assert deltas["LEC"] == 2.0
    assert deltas["HAM"] == -2.0


def test_teammate_grid_delta_tied(sample_grid_df):
    """Tied grid positions yield 0.0 delta."""
    drivers = ["NOR", "PIA"]
    res = compute_teammate_grid_delta(sample_grid_df, drivers)
    deltas = res.set_index("Abbreviation")["teammate_grid_delta"].to_dict()
    assert deltas["NOR"] == 0.0
    assert deltas["PIA"] == 0.0


def test_teammate_grid_delta_missing_and_solo_evaluates_to_nan():
    """Missing starting grid or solo driver evaluates strictly to NaN (Fix 2)."""
    grid_df = pd.DataFrame({
        "Abbreviation": ["ALB", "OCO", "GAS"],
        "TeamName": ["Williams", "Alpine", "Alpine"],
        "ActualStartingGrid": [10, 12, np.nan],  # GAS has missing grid, ALB is solo
    })
    res = compute_teammate_grid_delta(grid_df, ["ALB", "OCO", "GAS"])
    deltas = res.set_index("Abbreviation")["teammate_grid_delta"].to_dict()

    assert np.isnan(deltas["ALB"])  # Solo driver
    assert np.isnan(deltas["OCO"])  # Teammate has NaN
    assert np.isnan(deltas["GAS"])  # Self has NaN


def test_teammate_grid_delta_empty_or_none():
    """None or empty grid dataframe evaluates to NaN."""
    res_none = compute_teammate_grid_delta(None, ["RUS", "ANT"])
    assert res_none["teammate_grid_delta"].isna().all()


# -------------------------------------------------------------------------
# 14-22: driver_teammate_h2h_ratio tests
# -------------------------------------------------------------------------
def test_rolling_h2h_win_loss_tie():
    """Validates win, loss, and tie calculation in head-to-head racing."""
    prior_races = pd.DataFrame({
        "Round": [1, 1, 2, 2, 3, 3],
        "Abbreviation": ["RUS", "ANT", "RUS", "ANT", "RUS", "ANT"],
        "TeamName": ["Mercedes"] * 6,
        "Position": [1, 4, 4, 2, 3, 3],  # R1: RUS win; R2: ANT win; R3: tie
        "Status": ["Finished"] * 6,
    })
    lineup = pd.DataFrame({"Abbreviation": ["RUS", "ANT"], "TeamName": ["Mercedes", "Mercedes"]})

    # R1: RUS wins (1.0). R2: ANT wins (RUS 0.0). R3: tie (0.5).
    # RUS wins = 1.5, valid races = 3.
    # Shrunk = (1.5 + 3 * 0.5) / (3 + 3) = 3.0 / 6.0 = 0.50
    h2h = compute_rolling_teammate_h2h(prior_races, lineup, prior_weight=3.0)
    assert np.isclose(h2h["RUS"], 0.50, atol=1e-3)
    assert np.isclose(h2h["ANT"], 0.50, atol=1e-3)


def test_rolling_h2h_dual_dnf_neutral():
    """Dual-DNF rounds are neutral and excluded from valid races."""
    prior_races = pd.DataFrame({
        "Round": [1, 1, 2, 2],
        "Abbreviation": ["RUS", "ANT", "RUS", "ANT"],
        "TeamName": ["Mercedes"] * 4,
        "Position": [1, 4, 19, 20],
        "Status": ["Finished", "Finished", "Collision", "Retired"],  # R2: both DNF
    })
    lineup = pd.DataFrame({"Abbreviation": ["RUS", "ANT"], "TeamName": ["Mercedes", "Mercedes"]})

    # Only Round 1 counts. Valid races = 1, RUS wins = 1.
    # Shrunk = (1 + 1.5) / (1 + 3) = 2.5 / 4.0 = 0.625
    h2h = compute_rolling_teammate_h2h(prior_races, lineup, prior_weight=3.0)
    assert np.isclose(h2h["RUS"], 0.625, atol=1e-3)
    assert np.isclose(h2h["ANT"], 0.375, atol=1e-3)


def test_rolling_h2h_single_dnf():
    """Single DNF gives a win to the finishing teammate and loss to the retiree."""
    prior_races = pd.DataFrame({
        "Round": [1, 1],
        "Abbreviation": ["RUS", "ANT"],
        "TeamName": ["Mercedes", "Mercedes"],
        "Position": [2, 20],
        "Status": ["Finished", "Engine"],  # ANT DNF
    })
    lineup = pd.DataFrame({"Abbreviation": ["RUS", "ANT"], "TeamName": ["Mercedes", "Mercedes"]})

    # Valid races = 1. RUS wins = 1, ANT wins = 0.
    h2h = compute_rolling_teammate_h2h(prior_races, lineup, prior_weight=3.0)
    assert np.isclose(h2h["RUS"], 2.5 / 4.0, atol=1e-3)
    assert np.isclose(h2h["ANT"], 1.5 / 4.0, atol=1e-3)


def test_rolling_h2h_shrinkage_schedule():
    """Validates Empirical Bayes shrinkage progression with prior M=3.0."""
    lineup = pd.DataFrame({"Abbreviation": ["VER", "HAD"], "TeamName": ["Red Bull Racing", "Red Bull Racing"]})

    # 0 prior races -> 0.50
    h2h_0 = compute_rolling_teammate_h2h(pd.DataFrame(), lineup, prior_weight=3.0)
    assert h2h_0["VER"] == 0.50

    # 10 prior races, VER wins 8 -> (8 + 1.5) / (10 + 3) = 9.5 / 13 = 0.7308
    races_10 = []
    for r in range(1, 11):
        races_10.append({"Round": r, "Abbreviation": "VER", "TeamName": "Red Bull Racing", "Position": 1 if r <= 8 else 4, "Status": "Finished"})
        races_10.append({"Round": r, "Abbreviation": "HAD", "TeamName": "Red Bull Racing", "Position": 2 if r <= 8 else 1, "Status": "Finished"})
    h2h_10 = compute_rolling_teammate_h2h(pd.DataFrame(races_10), lineup, prior_weight=3.0)
    assert np.isclose(h2h_10["VER"], 9.5 / 13.0, atol=1e-3)
    assert np.isclose(h2h_10["HAD"], 3.5 / 13.0, atol=1e-3)


def test_rolling_h2h_bounding_clamps():
    """Validates bounding clamps [0.15, 0.85] (Fix 3)."""
    lineup = pd.DataFrame({"Abbreviation": ["VER", "HAD"], "TeamName": ["Red Bull Racing", "Red Bull Racing"]})

    # 25 races, VER wins 25 -> raw shrunk = (25 + 1.5)/28 = 0.9464 -> clamped to 0.85
    races_25 = []
    for r in range(1, 26):
        races_25.append({"Round": r, "Abbreviation": "VER", "TeamName": "Red Bull Racing", "Position": 1, "Status": "Finished"})
        races_25.append({"Round": r, "Abbreviation": "HAD", "TeamName": "Red Bull Racing", "Position": 10, "Status": "Finished"})

    h2h = compute_rolling_teammate_h2h(pd.DataFrame(races_25), lineup, prior_weight=3.0, lower_bound=0.15, upper_bound=0.85)
    assert h2h["VER"] == 0.85
    assert h2h["HAD"] == 0.15


def test_rolling_h2h_midseason_driver_change():
    """Only matches prior races when both drivers raced for the same team."""
    # Round 1: DRV_A and DRV_B were on Team_Alpha.
    # Round 2: DRV_A was on Team_Alpha, DRV_B was on Team_Beta.
    # Round 3: DRV_A and DRV_B are on Team_Alpha.
    prior_races = pd.DataFrame({
        "Round": [1, 1, 2, 2],
        "Abbreviation": ["DRV_A", "DRV_B", "DRV_A", "DRV_B"],
        "TeamName": ["Team_Alpha", "Team_Alpha", "Team_Alpha", "Team_Beta"],
        "Position": [1, 2, 1, 10],
        "Status": ["Finished"] * 4,
    })
    lineup = pd.DataFrame({
        "Abbreviation": ["DRV_A", "DRV_B"],
        "TeamName": ["Team_Alpha", "Team_Alpha"],
    })

    # Only Round 1 is a valid head-to-head on the same team!
    h2h = compute_rolling_teammate_h2h(prior_races, lineup, prior_weight=3.0)
    # 1 valid race, DRV_A wins 1: (1 + 1.5)/4 = 0.625
    assert np.isclose(h2h["DRV_A"], 0.625, atol=1e-3)
    assert np.isclose(h2h["DRV_B"], 0.375, atol=1e-3)


# -------------------------------------------------------------------------
# 23-26: Integration, weekend compatibility, anti-leakage, and imputer
# -------------------------------------------------------------------------
def test_pre_qualifying_stage_behavior():
    """In pre-qualifying stages before qualifying is held, gap and grid delta evaluate to NaN, while rolling H2H is computable."""
    drivers = ["RUS", "ANT"]
    # 1. Quali gap with None -> all NaN
    q_gap = compute_teammate_quali_gap(None, drivers)
    assert q_gap["teammate_quali_gap_pct"].isna().all()

    # 2. Grid delta with None -> all NaN
    g_delta = compute_teammate_grid_delta(None, drivers)
    assert g_delta["teammate_grid_delta"].isna().all()

    # 3. Rolling H2H is still populated from prior races
    prior_races = pd.DataFrame({
        "Round": [1, 1],
        "Abbreviation": ["RUS", "ANT"],
        "TeamName": ["Mercedes", "Mercedes"],
        "Position": [1, 3],
        "Status": ["Finished", "Finished"],
    })
    lineup = pd.DataFrame({"Abbreviation": ["RUS", "ANT"], "TeamName": ["Mercedes", "Mercedes"]})
    h2h = compute_rolling_teammate_h2h(prior_races, lineup)
    assert h2h["RUS"] == 0.625
    assert h2h["ANT"] == 0.375


def test_sprint_and_normal_weekend_compatibility(clean_quali_df):
    """Pipeline operates seamlessly on both sprint and normal weekend structures."""
    fe = FeatureEngineer(expected_grid_size=len(clean_quali_df))
    prior_races = pd.DataFrame({
        "Round": [1] * 6,
        "Abbreviation": ["RUS", "ANT", "LEC", "HAM", "NOR", "PIA"],
        "TeamName": ["Mercedes", "Mercedes", "Ferrari", "Ferrari", "McLaren", "McLaren"],
        "Position": [1, 2, 3, 4, 5, 6],
        "GridPosition": [1, 2, 3, 4, 5, 6],
        "Points": [25, 18, 15, 12, 10, 8],
        "Status": ["Finished"] * 6,
    })

    # Normal weekend (3 practice sessions)
    df_normal = fe.build_target_dataset(
        target_round=2,
        track_name="Sepang",
        quali_df=clean_quali_df,
        practice_sessions={"fp1": pd.DataFrame(), "fp2": pd.DataFrame(), "fp3": pd.DataFrame()},
        season_races_df=prior_races,
    )
    for col in ["teammate_quali_gap_pct", "teammate_grid_delta", "driver_teammate_h2h_ratio"]:
        assert col in df_normal.columns

    # Sprint weekend (sprint_qualifying, sprint_race)
    df_sprint = fe.build_target_dataset(
        target_round=2,
        track_name="Shanghai",
        quali_df=clean_quali_df,
        practice_sessions={"fp1": pd.DataFrame(), "sprint_qualifying": pd.DataFrame(), "sprint_race": pd.DataFrame()},
        season_races_df=prior_races,
    )
    for col in ["teammate_quali_gap_pct", "teammate_grid_delta", "driver_teammate_h2h_ratio"]:
        assert col in df_sprint.columns


def test_anti_leakage_target_race_results_excluded():
    """Target race and future rounds in season_races_df are strictly excluded from H2H."""
    fe = FeatureEngineer(expected_grid_size=2)
    # Include target round 2 and future round 3 in season_races_df
    season_df = pd.DataFrame({
        "Round": [1, 1, 2, 2, 3, 3],
        "Abbreviation": ["RUS", "ANT", "RUS", "ANT", "RUS", "ANT"],
        "TeamName": ["Mercedes"] * 6,
        "Position": [1, 4, 10, 1, 12, 1],  # In R2 and R3 ANT wins
        "Status": ["Finished"] * 6,
        "GridPosition": [1, 2, 3, 4, 5, 6],
        "Points": [25, 12, 1, 25, 0, 25],
    })

    lineup_quali = pd.DataFrame({
        "Abbreviation": ["RUS", "ANT"],
        "TeamName": ["Mercedes", "Mercedes"],
        "Position": [1, 2],
        "Q1": ["0 days 00:01:20.000000", "0 days 00:01:20.500000"],
    })

    # Build target dataset for Round 2. Only Round 1 (< 2) must be considered for H2H!
    target_df = fe.build_target_dataset(
        target_round=2,
        track_name="Sepang",
        quali_df=lineup_quali,
        practice_sessions={},
        season_races_df=season_df,
    )

    rus = target_df[target_df["Abbreviation"] == "RUS"].iloc[0]
    ant = target_df[target_df["Abbreviation"] == "ANT"].iloc[0]

    # In Round 1, RUS won. If R2 or R3 leaked, ANT would have more wins.
    # Therefore, RUS must have > 0.50 win rate!
    assert rus["driver_teammate_h2h_ratio"] == 0.625
    assert ant["driver_teammate_h2h_ratio"] == 0.375


def test_model_config_has_37_features():
    """Verify config/model_config.json contains exactly 37 configured features."""
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "model_config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    features = cfg.get("features", [])
    assert len(features) == 37, f"Expected 37 features, found {len(features)}: {features}"
    assert "teammate_quali_gap_pct" in features
    assert "teammate_grid_delta" in features
    assert "driver_teammate_h2h_ratio" in features


def test_simple_imputer_handles_teammate_nans():
    """Scikit-learn SimpleImputer transforms teammate feature NaNs without errors."""
    data = pd.DataFrame({
        "teammate_quali_gap_pct": [-0.5, np.nan, 0.5, np.nan],
        "teammate_grid_delta": [-2.0, np.nan, 2.0, np.nan],
        "driver_teammate_h2h_ratio": [0.65, 0.50, 0.35, 0.50],
    })
    imputer = SimpleImputer(strategy="median")
    transformed = imputer.fit_transform(data)
    assert not np.isnan(transformed).any()
    assert transformed.shape == (4, 3)
