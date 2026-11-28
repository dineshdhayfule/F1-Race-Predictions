import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from apexgrid.models.train import ModelTrainer, create_feature_imputer
from apexgrid.models.predict import Predictor
from apexgrid.models.evaluate import ModelEvaluator


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return json.loads((ROOT / "src" / "apexgrid" / "config" / "model_config.json").read_text(encoding="utf-8"))


def _synthetic(config):
    np.random.seed(42)
    features = config["features"]
    df = pd.DataFrame({feature: np.random.randn(66) for feature in features})
    df["ActualFinish"] = np.random.randint(1, 23, size=66)
    df["ActualStartingGrid"] = np.random.randint(1, 23, size=66)
    df["GridPosition"] = df["ActualStartingGrid"]
    df["Round"] = np.repeat([1, 2, 3], 22)
    df["Abbreviation"] = [f"D{i % 22}" for i in range(66)]
    df["TeamName"] = [f"T{i % 11}" for i in range(66)]
    df["StartingGridStatus"] = ["CLASSIFIED"] * 66
    df["is_pit_lane_start"] = [1 if i % 22 == 21 else 0 for i in range(66)]
    return df


def test_training_inference_feature_parity():
    config = _config()
    assert list(ModelTrainer(config).feature_cols) == config["features"]
    assert list(ModelEvaluator(config).feature_cols) == config["features"]
    assert list(Predictor(config).feature_cols) == config["features"]


def test_ridge_architecture_and_scaling():
    config = _config()
    df = _synthetic(config)
    trainer = ModelTrainer(config)
    imputer = create_feature_imputer(strategy="median")
    X = imputer.fit_transform(df[config["features"]])
    pipe = trainer.fit_ridge(X, df["ActualFinish"].values)
    assert isinstance(pipe, Pipeline)
    assert isinstance(pipe.named_steps["scaler"], StandardScaler)
    assert isinstance(pipe.named_steps["ridge"], Ridge)
    assert pipe.named_steps["ridge"].alpha == 100.0


def test_random_forest_hyperparameters():
    config = _config()
    df = _synthetic(config)
    trainer = ModelTrainer(config)
    _, rf, _, _, _ = trainer.fit(df, return_ridge=True)
    assert rf.n_estimators == 500
    assert rf.max_depth == 8
    assert rf.min_samples_leaf == 2
    assert rf.random_state == 42


def test_xgboost_hyperparameters():
    config = _config()
    df = _synthetic(config)
    trainer = ModelTrainer(config)
    _, _, xgb, _, _ = trainer.fit(df, return_ridge=True)
    params = xgb.get_params()
    assert params["n_estimators"] == 250
    assert params["max_depth"] == 3
    assert abs(params["learning_rate"] - 0.015) < 1e-5
    assert params["min_child_weight"] == 2
    assert params["reg_alpha"] == 1.0
    assert params["reg_lambda"] == 2.0


def test_circuit_blending_weights():
    predictor = Predictor(_config())
    assert predictor.resolve_blending_weights("street") == (0.50, 0.50)
    assert predictor.resolve_blending_weights("permanent") == (0.65, 0.35)
    assert predictor.resolve_blending_weights("temporary") == (0.55, 0.45)
    assert predictor.resolve_blending_weights("unknown") == (0.65, 0.35)


def test_tri_ensemble_is_40_30_30():
    config = _config()
    assert config["model_selection"]["promoted_ensemble"] == {
        "rf_weight": 0.40,
        "xgb_weight": 0.30,
        "ridge_weight": 0.30,
    }


def test_default_hybrid_is_60_40():
    config = _config()
    assert config["model_selection"]["fixed_rf_weight"] == 0.60
    assert config["model_selection"]["fixed_xgb_weight"] == 0.40


def test_stage_masking_is_lossless_at_after_qualifying():
    from apexgrid.features.build_features import apply_stage_feature_mask
    config = _config()
    df = pd.DataFrame([{c: 1.0 for c in config["features"]}])
    masked = apply_stage_feature_mask(df, "after_qualifying")
    assert not np.isnan(masked["QualifyingPosition"].iloc[0])
    assert not np.isnan(masked["GridPosition"].iloc[0])


def test_monte_carlo_draw_depth():
    config = _config()
    assert config["monte_carlo"]["default_simulations"] == 10000
    assert config["monte_carlo"]["precision_simulations"] == 25000


def test_custom_weight_pair_must_sum_to_one_hundred_percent():
    from typer.testing import CliRunner
    from apexgrid.cli.typer_app import app

    runner = CliRunner()
    bad_res = runner.invoke(
        app,
        ["predict", "-t", "melbourne", "-m", "hybrid", "--rf-weight", "70", "--xgb-weight", "20"],
    )
    assert bad_res.exit_code == 2
    assert "Weight sum must equal 100%" in bad_res.output
