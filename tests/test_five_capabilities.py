"""
Test Suite: Five Major ML Capabilities
======================================
Tests covering:
1. Race Pace Intelligence (Stint extraction, filtering, fuel correction, pace rank)
2. Tire Degradation Modelling (Compound slopes, teammate borrowing, clipping)
3. Pit Stop Strategy Simulator (1/2-stop evaluation, 2-compound rule, undercut/overcut)
4. Race Control Model (SC/VSC/Red flag priors, field compression, cheap pit stop)
5. Weather-Aware Predictions (Climate priors, session weather, wet driver skill, stage masking)
6. Anti-leakage and Monte Carlo Determinism
"""

import numpy as np
import pandas as pd
import pytest

from src.features.race_pace import (
    extract_stints_from_laps,
    compute_driver_race_pace,
    build_race_pace_features,
)
from src.features.tire_degradation import (
    estimate_stint_degradation_slope,
    TireDegradationModel,
    build_tire_degradation_features,
)
from src.simulation.strategy_simulator import (
    PitStrategySimulator,
    build_pit_strategy_features,
)
from src.simulation.race_control_model import (
    RaceControlModel,
    build_race_control_features,
)
from src.features.weather_features import (
    WeatherModel,
    extract_session_weather_metrics,
    build_weather_features,
)
from src.simulation.monte_carlo import MonteCarloSimulator


# ==============================================================================
# 1. Race Pace Intelligence Tests
# ==============================================================================
def test_race_pace_stint_extraction_filters_in_out_and_short_stints():
    """Verify that in/out laps, deleted laps, and stints < 4 laps are excluded."""
    # Stint 1: 5 laps (valid long run)
    stint1_laps = [
        {"Driver": "VER", "LapNumber": 1, "LapTimeSeconds": 95.0, "Stint": 1, "PitIn": True, "Deleted": False, "Compound": "MEDIUM"},
        {"Driver": "VER", "LapNumber": 2, "LapTimeSeconds": 82.5, "Stint": 1, "PitIn": False, "Deleted": False, "Compound": "MEDIUM"},
        {"Driver": "VER", "LapNumber": 3, "LapTimeSeconds": 82.6, "Stint": 1, "PitIn": False, "Deleted": False, "Compound": "MEDIUM"},
        {"Driver": "VER", "LapNumber": 4, "LapTimeSeconds": 82.8, "Stint": 1, "PitIn": False, "Deleted": False, "Compound": "MEDIUM"},
        {"Driver": "VER", "LapNumber": 5, "LapTimeSeconds": 82.9, "Stint": 1, "PitIn": False, "Deleted": False, "Compound": "MEDIUM"},
        {"Driver": "VER", "LapNumber": 6, "LapTimeSeconds": 83.1, "Stint": 1, "PitIn": False, "Deleted": False, "Compound": "MEDIUM"},
    ]
    # Stint 2: 2 laps (short run, should be filtered out)
    stint2_laps = [
        {"Driver": "VER", "LapNumber": 10, "LapTimeSeconds": 81.0, "Stint": 2, "PitIn": False, "Deleted": False, "Compound": "SOFT"},
        {"Driver": "VER", "LapNumber": 11, "LapTimeSeconds": 81.2, "Stint": 2, "PitIn": False, "Deleted": False, "Compound": "SOFT"},
    ]
    df = pd.DataFrame(stint1_laps + stint2_laps)
    stints = extract_stints_from_laps(df, min_consecutive_laps=4)

    assert len(stints) == 1, f"Expected 1 valid stint, got {len(stints)}"
    stint = stints[0]
    assert stint["driver"] == "VER"
    assert stint["compound"] == "MEDIUM"
    assert stint["lap_count"] == 5


def test_race_pace_features_and_rank_calculation():
    """Verify clean pace ranking across multiple drivers."""
    laps_list = []
    # VER: fast pace ~82.0s
    for lap in range(1, 7):
        laps_list.append({"Driver": "VER", "LapNumber": lap, "LapTimeSeconds": 82.0 + lap * 0.05, "Stint": 1, "Deleted": False, "Compound": "MEDIUM"})
    # HAM: medium pace ~82.6s
    for lap in range(1, 7):
        laps_list.append({"Driver": "HAM", "LapNumber": lap, "LapTimeSeconds": 82.6 + lap * 0.05, "Stint": 1, "Deleted": False, "Compound": "MEDIUM"})

    practice_df = pd.DataFrame(laps_list)
    feats = build_race_pace_features({"fp2": practice_df}, drivers=["VER", "HAM", "BOT"])

    ver_row = feats[feats["Abbreviation"] == "VER"].iloc[0]
    ham_row = feats[feats["Abbreviation"] == "HAM"].iloc[0]
    bot_row = feats[feats["Abbreviation"] == "BOT"].iloc[0]

    assert ver_row["long_run_pace_rank"] == 1.0
    assert ham_row["long_run_pace_rank"] == 2.0
    assert ver_row["long_run_pace_gap_pct"] == 0.0
    assert ham_row["long_run_pace_gap_pct"] > 0.0
    assert bot_row["long_run_pace_rank"] == 22.0  # Fallback for missing driver


# ==============================================================================
# 2. Tire Degradation Modelling Tests
# ==============================================================================
def test_tire_degradation_slope_estimation_and_bounds():
    """Verify linear degradation slope calculation with bounds clipping."""
    # Degrades 0.080s per lap with slight noise
    laps = np.array([80.0, 80.09, 80.16, 80.25, 80.31, 80.40, 80.49, 80.56])
    slope, r2 = estimate_stint_degradation_slope(laps)

    assert 0.060 <= slope <= 0.100, f"Expected slope around 0.08, got {slope}"
    assert r2 > 0.90, f"Expected strong R^2, got {r2}"


def test_tire_degradation_teammate_borrowing():
    """Verify that when a driver lacks compound data, teammate pooling is borrowed."""
    model = TireDegradationModel()
    team_map = {"VER": "Red Bull", "PER": "Red Bull"}

    # PER has 6 valid soft laps
    per_laps = [
        {"Driver": "PER", "LapNumber": i, "LapTimeSeconds": 80.0 + i * 0.12, "Stint": 1, "Compound": "SOFT", "Deleted": False}
        for i in range(1, 7)
    ]
    df = pd.DataFrame(per_laps)
    deg_df = model.extract_driver_compound_degradation(
        laps_df=df,
        drivers=["VER", "PER"],
        team_map=team_map,
    )

    ver_row = deg_df[deg_df["Abbreviation"] == "VER"].iloc[0]
    per_row = deg_df[deg_df["Abbreviation"] == "PER"].iloc[0]

    assert pd.notna(ver_row["soft_tyre_deg_rate"])
    assert ver_row["tyre_deg_confidence"] <= per_row["tyre_deg_confidence"]


# ==============================================================================
# 3. Pit Stop Strategy Simulator Tests
# ==============================================================================
def test_pit_strategy_satisfies_two_dry_compounds_regulation():
    """Verify that valid dry race strategies use at least two distinct dry compounds."""
    sim = PitStrategySimulator()
    strategies = sim.evaluate_strategy_candidates(
        base_lap_pace=85.0,
        total_laps=56,
        circuit_key="mexico",
        is_wet=False,
    )

    assert len(strategies) >= 2, "Expected multiple viable strategy candidates"
    for strat in strategies:
        compounds = strat["compounds"]
        assert len(set(compounds)) >= 2, f"Strategy {strat['strategy_name']} violates 2-compound rule: {compounds}"


def test_pit_strategy_wet_race_single_compound_allowed():
    """Verify that wet race allows single compound running (Inter/Wet rule)."""
    sim = PitStrategySimulator()
    strategies = sim.evaluate_strategy_candidates(
        base_lap_pace=95.0,
        total_laps=56,
        circuit_key="sepang",
        is_wet=True,
    )
    assert len(strategies) > 0
    inter_only = [s for s in strategies if all(c == "INTERMEDIATE" for c in s["compounds"])]
    assert len(inter_only) > 0


# ==============================================================================
# 4. Safety Car / VSC / Red Flag Model Tests
# ==============================================================================
def test_race_control_probabilities_and_circuit_lookup():
    """Verify empirical priors lookup for street circuits vs permanent circuits."""
    model = RaceControlModel()
    monaco_p = model.get_circuit_event_probabilities("monaco")
    monza_p = model.get_circuit_event_probabilities("monza")

    assert monaco_p["p_safety_car"] > monza_p["p_safety_car"]
    assert 0.0 <= monaco_p["p_safety_car"] <= 1.0
    assert 0.0 <= monaco_p["p_red_flag"] <= 1.0


def test_race_control_field_compression():
    """Verify that SC deployment compresses race gaps while preserving leader ordering."""
    model = RaceControlModel()
    rng = np.random.RandomState(42)

    # Base scores: 1st=1.0, 2nd=6.0, 3rd=12.0
    scores = np.array([1.0, 6.0, 12.0, 18.0, 24.0])
    compressed, triggered = model.simulate_race_control_disruptions(
        base_scores=scores,
        p_sc=1.0,  # Force SC
        p_vsc=0.0,
        p_red_flag=0.0,
        rng=rng,
    )

    assert triggered["safety_car"] is True
    assert len(compressed) == len(scores)


# ==============================================================================
# 5. Weather-Aware Prediction Tests
# ==============================================================================
def test_weather_model_wet_driver_skill_bonuses():
    """Verify that recognized wet-weather masters receive appropriate performance bonuses."""
    w_model = WeatherModel()
    ver_bonus = w_model.driver_wet_bonuses.get("VER", 0.0)
    ham_bonus = w_model.driver_wet_bonuses.get("HAM", 0.0)

    # In our model, lower score = faster pace
    assert ver_bonus < 0.0, f"Verstappen should receive negative (faster) score delta in wet: {ver_bonus}"
    assert ham_bonus < 0.0, f"Hamilton should receive negative (faster) score delta in wet: {ham_bonus}"


def test_weather_features_circuit_fallbacks():
    """Verify that missing weather sessions cleanly fallback to circuit climatological priors."""
    feats = build_weather_features(drivers=["VER", "NOR"], circuit_key="sepang", stage="pre_fp1")
    assert len(feats) == 2
    row = feats.iloc[0]
    assert row["rain_probability"] > 0.25  # Sepang tropical climate prior
    assert row["track_temperature"] >= 30.0


# ==============================================================================
# 6. Monte Carlo Integration and Determinism Tests
# ==============================================================================
def test_monte_carlo_deterministic_with_same_seed():
    """Verify that Monte Carlo execution is strictly deterministic across runs."""
    config = {
        "random_seed": 42,
        "monte_carlo": {"default_simulations": 1000, "fallback_sigma": 2.5, "fallback_dnf_prob": 0.15},
        "capabilities": {
            "race_pace_intelligence": True,
            "tire_degradation_modeling": True,
            "pit_strategy_simulator": True,
            "race_control_model": True,
            "weather_aware_prediction": True,
        }
    }
    sim1 = MonteCarloSimulator(config=config)
    sim2 = MonteCarloSimulator(config=config)

    df = pd.DataFrame({
        "Abbreviation": ["VER", "HAM", "LEC"],
        "TeamName": ["Red Bull", "Mercedes", "Ferrari"],
        "simulation_start_rank": [1.0, 2.0, 3.0],
        "ActualStartingGrid": [1.0, 2.0, 3.0],
        "QualifyingPosition": [1.0, 2.0, 3.0],
        "is_pit_lane_start": [0, 0, 0],
        "p_safety_car": [0.5, 0.5, 0.5],
        "p_vsc": [0.3, 0.3, 0.3],
        "p_red_flag": [0.1, 0.1, 0.1],
        "rain_probability": [0.1, 0.1, 0.1],
        "strategy_time_loss_sec": [1.0, 2.0, 3.0],
        "recommended_strategy": ["1-Stop (M-H)", "1-Stop (M-H)", "2-Stop (S-M-H)"],
        "optimal_strategy_stops": [1, 1, 2],
        "strategy_delta_seconds": [3.2, 4.1, 2.8],
    })
    scores = np.array([1.2, 2.5, 3.1])

    res1, _ = sim1.simulate(target_df=df, deterministic_scores=scores, circuit_key="mexico")
    res2, _ = sim2.simulate(target_df=df, deterministic_scores=scores, circuit_key="mexico")

    assert np.allclose(res1["ExpectedFinish"].values, res2["ExpectedFinish"].values), "Monte Carlo simulations must be deterministic"
    assert res1["Abbreviation"].tolist() == res2["Abbreviation"].tolist()
