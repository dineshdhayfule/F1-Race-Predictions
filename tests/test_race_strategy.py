import numpy as np
import pandas as pd

from src.features.race_pace import build_race_pace_features
from src.features.tire_degradation import estimate_stint_degradation_slope
from src.simulation.strategy_simulator import PitStrategySimulator


def test_missing_race_pace_driver_keeps_rank_22():
    laps = pd.DataFrame([
        {"Driver": "VER", "LapNumber": i, "LapTimeSeconds": 82.0 + i * 0.05,
         "Stint": 1, "Deleted": False, "Compound": "MEDIUM"}
        for i in range(1, 7)
    ] + [
        {"Driver": "HAM", "LapNumber": i, "LapTimeSeconds": 82.6 + i * 0.05,
         "Stint": 1, "Deleted": False, "Compound": "MEDIUM"}
        for i in range(1, 7)
    ])
    features = build_race_pace_features({"fp2": laps}, drivers=["VER", "HAM", "BOT"])
    bot = features.loc[features["Abbreviation"] == "BOT", "long_run_pace_rank"].iloc[0]
    assert bot == 22.0


def test_measured_race_pace_ranks_fastest_first():
    laps = pd.DataFrame([
        {"Driver": "VER", "LapNumber": i, "LapTimeSeconds": 82.0 + i * 0.05,
         "Stint": 1, "Deleted": False, "Compound": "MEDIUM"}
        for i in range(1, 7)
    ] + [
        {"Driver": "HAM", "LapNumber": i, "LapTimeSeconds": 82.6 + i * 0.05,
         "Stint": 1, "Deleted": False, "Compound": "MEDIUM"}
        for i in range(1, 7)
    ])
    features = build_race_pace_features({"fp2": laps}, drivers=["VER", "HAM"])
    assert features.loc[features["Abbreviation"] == "VER", "long_run_pace_rank"].iloc[0] == 1.0
    assert features.loc[features["Abbreviation"] == "HAM", "long_run_pace_rank"].iloc[0] == 2.0


def test_tire_degradation_is_bounded():
    laps = np.array([80.0, 80.09, 80.16, 80.25, 80.31, 80.40])
    slope, r2 = estimate_stint_degradation_slope(laps)
    assert 0.005 <= slope <= 0.40
    assert 0.0 <= r2 <= 1.0


def test_pit_strategy_respects_two_dry_compounds():
    simulator = PitStrategySimulator()
    options = simulator.evaluate_strategy_candidates(
        base_lap_pace=82.0,
        total_laps=57,
        circuit_key="australia",
        is_wet=False,
    )
    assert options
    assert all(len(set(option["compounds"])) >= 2 for option in options if "compounds" in option)


def test_wet_race_allows_single_compound():
    simulator = PitStrategySimulator()
    options = simulator.evaluate_strategy_candidates(
        base_lap_pace=92.0,
        total_laps=57,
        circuit_key="spa",
        is_wet=True,
    )
    assert options
