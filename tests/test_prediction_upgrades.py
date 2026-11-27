"""
Targeted unit tests for prediction, Monte Carlo, strategy priors, and CLI defaults.
"""

import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from apexgrid.data.data_manager import DataManager
from apexgrid.features.build_features import apply_stage_feature_mask
from apexgrid.features.season_features import build_season_form
from apexgrid.features.teammate_features import compute_rolling_teammate_h2h
from apexgrid.models.evaluate import ModelEvaluator
from apexgrid.models.predict import Predictor
from apexgrid.pipeline.race_prediction_pipeline import RacePredictionPipeline
from apexgrid.simulation.monte_carlo import MonteCarloSimulator
from apexgrid.cli.typer_app import cmd_predict
from apexgrid.utils.paths import get_config_path, get_runtime_root, is_dev_mode


def test_provisional_grid_discount_vs_actual_grid():
    cfg = {
        "blending": {
            "ml_weight": 0.65,
            "grid_weight": 0.35,
            "provisional_grid_discount": 0.5,
            "circuit_aware": True,
            "circuit_weights": {
                "permanent": {"grid_weight": 0.35, "ml_weight": 0.65},
            },
        },
        "model_selection": {"mode": "ensemble"},
    }
    predictor = Predictor(cfg)
    predictor.feature_cols = ["grid_norm"]

    imputer = MagicMock()
    imputer.transform.side_effect = lambda X: X.values
    rf = MagicMock()
    rf.predict.return_value = np.array([10.0, 10.0])

    # 1. Actual GridPosition -> full 0.35 grid / 0.65 ML: 0.65*10 + 0.35*2 = 7.20
    df_actual = pd.DataFrame([
        {"Abbreviation": "VER", "TeamName": "Red Bull Racing", "GridPosition": 2, "grid_norm": 0.05},
        {"Abbreviation": "NOR", "TeamName": "McLaren", "GridPosition": 4, "grid_norm": 0.15},
    ])
    _, scores_actual = predictor.predict_deterministic(
        df_actual, imputer=imputer, rf_model=rf, circuit_type="permanent", model_mode="rf"
    )
    assert np.isclose(scores_actual[0], 7.20, atol=1e-4)

    # 2. Provisional grid -> discounted 0.175 grid / 0.825 ML: 0.825*10 + 0.175*2 = 8.60
    df_prov = pd.DataFrame([
        {"Abbreviation": "VER", "TeamName": "Red Bull Racing", "provisional_grid_position": 2, "grid_norm": 0.05},
        {"Abbreviation": "NOR", "TeamName": "McLaren", "provisional_grid_position": 4, "grid_norm": 0.15},
    ])
    _, scores_prov = predictor.predict_deterministic(
        df_prov, imputer=imputer, rf_model=rf, circuit_type="permanent", model_mode="rf"
    )
    assert np.isclose(scores_prov[0], 8.60, atol=1e-4)

    # 3. Explicit pre-2018 override (0.75 grid / 0.25 ML) is preserved
    _, scores_override = predictor.predict_deterministic(
        df_actual, imputer=imputer, rf_model=rf, circuit_type="permanent",
        grid_weight=0.75, ml_weight=0.25, model_mode="rf",
    )
    assert np.isclose(scores_override[0], 4.00, atol=1e-4)


def test_constructor_sigma_shrinkage_and_exp_finish_classified():
    cfg = {
        "random_seed": 42,
        "monte_carlo": {"default_simulations": 500, "fallback_sigma": 2.5},
        "capabilities": {"race_control_model": False, "weather_aware_prediction": False, "pit_strategy_simulator": False},
    }
    sim = MonteCarloSimulator(cfg)

    target_df = pd.DataFrame([
        {"Abbreviation": f"D{i}", "TeamName": "TeamA" if i < 2 else "TeamB", "GridPosition": i + 1}
        for i in range(20)
    ])
    scores = np.arange(1.0, 21.0)
    # TeamA has >= 8 tight residuals (near 0 -> hits 1.20 floor); TeamB has < 8 residuals -> falls back to global_sigma
    team_residuals = {
        "TeamA": np.array([0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.0, 0.15]),
        "TeamB": np.array([5.0, -5.0, 4.0]),
    }

    res_df, meta = sim.simulate(
        target_df=target_df,
        deterministic_scores=scores,
        pace_sigma=2.5,
        empirical_dnf_rate=0.10,
        n_simulations=500,
        team_residuals=team_residuals,
    )
    assert "ExpFinishClassified" in res_df.columns
    assert res_df["ExpFinishClassified"].notna().all()
    assert meta["calibrated_sigma"] == 2.5


def test_all_tracks_have_strategy_circuit_priors():
    tracks = json.loads(Path("src/apexgrid/config/tracks.json").read_text(encoding="utf-8"))["tracks"]
    priors = json.loads(Path("src/apexgrid/config/strategy_config.json").read_text(encoding="utf-8"))["circuit_priors"]
    missing = set(tracks.keys()) - set(priors.keys())
    assert missing == set(), f"Missing circuit_priors for: {sorted(missing)}"


def test_cli_default_model_is_ensemble():
    sig = inspect.signature(cmd_predict)
    assert sig.parameters["model"].default.default == "ensemble"


def test_pipeline_run_default_model_is_ensemble():
    sig = inspect.signature(RacePredictionPipeline.run)
    assert sig.parameters["model"].default == "ensemble"


@pytest.mark.slow
def test_evaluate_walk_forward_returns_team_residuals_and_ensemble_fallback():
    cfg = {
        "random_seed": 42,
        "random_forest": {"n_estimators": 10, "max_depth": 3, "min_samples_leaf": 1},
        "xgboost": {"n_estimators": 10, "max_depth": 2, "learning_rate": 0.1},
        "ridge": {"alpha": 10.0},
        "blending": {"ml_weight": 0.65, "grid_weight": 0.35, "circuit_aware": False},
        "model_selection": {"mode": "auto", "min_validation_priors": 4},
    }
    evaluator = ModelEvaluator(cfg)
    evaluator.feature_cols = ["grid_norm", "driver_avg_finish"]

    # Build 4 rounds x 4 drivers with TeamName labels
    rows = []
    for rnd in range(1, 5):
        for i, (drv, team) in enumerate([("VER", "Red Bull"), ("NOR", "McLaren"), ("LEC", "Ferrari"), ("RUS", "Mercedes")]):
            rows.append({
                "Round": rnd,
                "Abbreviation": drv,
                "TeamName": team,
                "ActualFinish": float(i + 1),
                "GridPosition": float(i + 1),
                "ActualStartingGrid": float(i + 1),
                "grid_norm": (i + 1) / 20.0,
                "driver_avg_finish": float(i + 1),
                "driver_dnf_rate": 0.10,
            })
    df = pd.DataFrame(rows)

    res = evaluator.evaluate_walk_forward(df)
    assert "team_residuals" in res
    assert isinstance(res["team_residuals"], dict)
    assert set(["Red Bull", "McLaren", "Ferrari", "Mercedes"]).issubset(res["team_residuals"].keys())

    # Empty df fallback should default to 0.40 / 0.30 / 0.30 and empty team_residuals
    empty_res = evaluator.evaluate_walk_forward(df.iloc[0:0])
    assert empty_res["best_w_rf"] == 0.40
    assert empty_res["best_w_xgb"] == 0.30
    assert empty_res["best_w_ridge"] == 0.30
    assert empty_res["team_residuals"] == {}


def test_predict_qualifying_uses_three_way_ensemble(tmp_path):
    pipeline = RacePredictionPipeline()
    pipeline.output_dir = tmp_path

    drivers = [("VER", "Red Bull Racing"), ("NOR", "McLaren"), ("LEC", "Ferrari"), ("RUS", "Mercedes")]
    race_rows, quali_rows, fp_rows = [], [], []
    for rnd in (1, 2):
        for idx, (drv, team) in enumerate(drivers, start=1):
            race_rows.append({"Round": rnd, "Abbreviation": drv, "TeamName": team, "Position": idx, "GridPosition": idx, "Status": "Finished", "Points": max(0, 26 - idx * 5)})
            quali_rows.append({"Round": rnd, "Abbreviation": drv, "TeamName": team, "Position": idx, "Q1": 80.0 + idx * 0.2, "Q2": 79.5 + idx * 0.2, "Q3": 79.0 + idx * 0.2})
            fp_rows.append({"Round": rnd, "Abbreviation": drv, "TeamName": team, "LapNumber": 5, "LapTimeSeconds": 81.0 + idx * 0.2})

    races_df = pd.DataFrame(race_rows)
    qualis_df = pd.DataFrame(quali_rows)
    sessions_by_round = {1: {"fp1": pd.DataFrame(fp_rows[:4])}, 2: {"fp1": pd.DataFrame(fp_rows[4:])}}
    target_fp = pd.DataFrame([
        {"Abbreviation": drv, "TeamName": team, "LapNumber": 5, "LapTimeSeconds": 80.5 + idx * 0.2}
        for idx, (drv, team) in enumerate(drivers, start=1)
    ])

    pipeline.data_loader.load_season_rounds = MagicMock(return_value=(races_df, qualis_df))
    pipeline.data_loader.load_all_practice_sessions = MagicMock(return_value=sessions_by_round)
    pipeline.data_loader.load_weekend_sessions = MagicMock(return_value={"fp1": target_fp, "fp2": target_fp, "fp3": target_fp})

    with patch("apexgrid.data.roster_resolver.RosterResolver.resolve_roster", return_value=pd.DataFrame({"Abbreviation": [d[0] for d in drivers], "TeamName": [d[1] for d in drivers]})):
        res = pipeline.predict_qualifying(track_key="monza", year=2026, round_num=3, stage="after_fp3")

    df = res["predictions"]
    assert len(df) == 4
    for col in ("Abbreviation", "ExpectedQualifyingPosition", "PoleProbability", "DeterministicScore"):
        assert col in df.columns
        assert df[col].notna().all()
    assert df["ExpectedQualifyingPosition"].is_monotonic_increasing


def test_rookie_zero_race_history_no_division_by_zero():
    empty_races = pd.DataFrame(columns=["Round", "Abbreviation", "TeamName", "Position", "GridPosition", "Status", "Points"])
    driver_form, team_form = build_season_form(target_round=1, races_df=empty_races)
    assert driver_form == {}
    assert team_form == {}

    h2h = compute_rolling_teammate_h2h(empty_races, pd.DataFrame([{"Abbreviation": "ROOK", "TeamName": "NewTeam"}]))
    assert h2h == {"ROOK": 0.5}

    sim = MonteCarloSimulator({"random_seed": 42, "monte_carlo": {"default_simulations": 100, "fallback_sigma": 2.5}})
    target = pd.DataFrame([{"Abbreviation": "ROOK", "TeamName": "NewTeam", "GridPosition": 1}])
    res_df, meta = sim.simulate(target, np.array([10.0]), residuals=np.array([]), team_residuals={}, n_simulations=100)
    assert np.isfinite(meta["calibrated_sigma"])
    assert res_df["ExpectedFinish"].notna().all()


def test_sprint_stages_mask_sunday_grid():
    df = pd.DataFrame([{
        "Abbreviation": "VER",
        "GridPosition": 1.0,
        "ActualStartingGrid": 1.0,
        "QualifyingPosition": 1.0,
        "simulation_start_rank": 1.0,
        "sprint_quali_pos": 2.0,
        "sprint_finish_pos": 1.0,
    }])

    masked_sq = apply_stage_feature_mask(df.copy(), "after_sprint_qualifying")
    for col in ("GridPosition", "ActualStartingGrid", "QualifyingPosition", "simulation_start_rank"):
        assert pd.isna(masked_sq[col].iloc[0])
    assert masked_sq["sprint_quali_pos"].iloc[0] == 2.0

    masked_sr = apply_stage_feature_mask(df.copy(), "after_sprint")
    for col in ("GridPosition", "ActualStartingGrid", "QualifyingPosition", "simulation_start_rank"):
        assert pd.isna(masked_sr[col].iloc[0])
    assert masked_sr["sprint_finish_pos"].iloc[0] == 1.0


def test_cold_start_fresh_home_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr("apexgrid.utils.paths.is_dev_mode", lambda: False)
    runtime_root = get_runtime_root()
    assert runtime_root == tmp_path / ".apexgrid"
    assert runtime_root.is_dir()
    assert get_config_path("tracks.json").is_file()
    dm = DataManager(root_dir=runtime_root)
    assert dm.root_dir == runtime_root


def test_driver_wet_skill_bonuses_coverage():
    bonuses = json.loads(Path("src/apexgrid/config/strategy_config.json").read_text(encoding="utf-8"))["wet_weather"]["driver_wet_skill_bonuses"]
    expected = {
        "PER": 0.15, "BOT": 0.10, "STR": 0.08, "TSU": 0.05, "ALB": 0.05,
        "ANT": 0.00, "HAD": 0.00, "LAW": 0.00, "BEA": 0.00,
        "BOR": 0.00, "COL": 0.00, "DOO": 0.00, "LIN": 0.00,
    }
    for drv, val in expected.items():
        assert drv in bonuses, f"Missing driver {drv} in driver_wet_skill_bonuses"
        assert bonuses[drv] == val


def test_single_config_source_in_src_config():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "config").exists(), "Duplicate root config/ directory should be removed"
    for fname in ["model_config.json", "settings.json", "strategy_config.json", "tracks.json"]:
        assert (root / "src" / "apexgrid" / "config" / fname).is_file(), f"Missing src/apexgrid/config/{fname}"
    assert is_dev_mode() is True
