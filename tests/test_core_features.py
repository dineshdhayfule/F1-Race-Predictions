import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.session_validation import validate_session_dataframe
from src.features.build_features import apply_stage_feature_mask
from src.models.train import create_feature_imputer
from src.models.train import ModelTrainer
from src.models.predict import Predictor
from src.models.evaluate import ModelEvaluator


ROOT = Path(__file__).resolve().parents[1]


def _load_config():
    return json.loads((ROOT / "config" / "model_config.json").read_text(encoding="utf-8"))


def test_exact_40_feature_configuration():
    config = _load_config()
    assert len(config["features"]) == 40
    assert "sprint_quali_pos" not in config["features"]
    assert "is_pit_lane_start" in config["features"]
    assert "long_run_pace_gap_pct" in config["features"]
    assert "long_run_pace_rank" in config["features"]


def test_feature_pruning_flags_are_authoritative():
    config = _load_config()
    assert config["feature_pruning"]["enabled"] is True
    assert "sprint_quali_pos" in config["feature_pruning"]["validated_pruned_features"]


def test_trainer_evaluator_predictor_feature_parity():
    config = _load_config()
    trainer = ModelTrainer(config)
    evaluator = ModelEvaluator(config)
    predictor = Predictor(config)
    assert trainer.feature_cols == config["features"]
    assert evaluator.feature_cols == config["features"]
    assert predictor.feature_cols == config["features"]


def test_stage_feature_mask_pre_fp1():
    config = _load_config()
    df = pd.DataFrame([{c: 1.0 for c in config["features"]}])
    masked = apply_stage_feature_mask(df, "pre_fp1")
    assert np.isnan(masked["practice_gap_pct"].iloc[0])
    assert np.isnan(masked["QualifyingPosition"].iloc[0])
    assert np.isnan(masked["GridPosition"].iloc[0])


def test_stage_feature_mask_practice_cutoffs():
    config = _load_config()
    df = pd.DataFrame([{c: 1.0 for c in config["features"]}])
    for stage, observed in [
        ("after_fp1", ["fp1_gap_pct"]),
        ("after_fp2", ["fp1_gap_pct", "fp2_gap_pct"]),
        ("after_fp3", ["fp1_gap_pct", "fp2_gap_pct", "fp3_gap_pct"]),
    ]:
        masked = apply_stage_feature_mask(df.copy(), stage)
        assert all(not np.isnan(masked[c].iloc[0]) for c in observed)
        assert np.isnan(masked["QualifyingPosition"].iloc[0])
        assert np.isnan(masked["GridPosition"].iloc[0])


def test_after_qualifying_reveals_grid_features():
    config = _load_config()
    df = pd.DataFrame([{c: 1.0 for c in config["features"]}])
    masked = apply_stage_feature_mask(df, "after_qualifying")
    assert not np.isnan(masked["QualifyingPosition"].iloc[0])
    assert not np.isnan(masked["GridPosition"].iloc[0])
    assert not np.isnan(masked["is_pit_lane_start"].iloc[0])


def test_all_nan_feature_is_imputed_without_crashing():
    df = pd.DataFrame({
        "all_nan": [np.nan, np.nan, np.nan],
        "usable": [1.0, 2.0, 3.0],
    })
    imputer = create_feature_imputer(strategy="median")
    result = imputer.fit_transform(df)
    assert result.shape == (3, 2)
    assert np.isfinite(result).all()


def test_teammate_features_remain_in_40_feature_config():
    config = _load_config()
    assert "driver_teammate_h2h_ratio" in config["features"]
    assert "teammate_grid_delta" in config["features"]


def test_untimed_practice_dataset_is_valid():
    df = pd.DataFrame({
        "Abbreviation": ["VER", "NOR"],
        "LapNumber": [1, 1],
        "LapTimeSeconds": [np.nan, np.nan],
    })
    valid, errors = validate_session_dataframe(df, "fp1")
    assert valid is True
    assert errors == []


def test_arbitrary_csv_is_not_a_session():
    df = pd.DataFrame({"name": ["sample"], "value": [1]})
    valid, errors = validate_session_dataframe(df, "fp2")
    assert valid is False
    assert "Missing driver identifier column." in errors


def test_race_validation_requires_usable_result():
    df = pd.DataFrame({
        "Abbreviation": ["VER"],
        "Position": [np.nan],
        "Status": [""],
    })
    valid, errors = validate_session_dataframe(df, "race")
    assert valid is False
    assert "Dataset contains no usable result observations." in errors


def test_empty_session_dataframe_is_rejected():
    valid, errors = validate_session_dataframe(pd.DataFrame(), "qualifying")
    assert valid is False
    assert "Dataset is empty." in errors


def test_malformed_qualifying_observation_is_rejected():
    df = pd.DataFrame({"Abbreviation": ["VER"], "Position": [np.nan]})
    valid, errors = validate_session_dataframe(df, "qualifying")
    assert valid is False
    assert "Dataset contains no usable result observations." in errors


def test_qualifying_observation_with_position_is_valid():
    df = pd.DataFrame({"Abbreviation": ["VER"], "Position": [1.0]})
    valid, errors = validate_session_dataframe(df, "qualifying")
    assert valid is True
    assert errors == []
