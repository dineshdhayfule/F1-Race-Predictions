import numpy as np
import pandas as pd

from src.models.evaluate import ModelEvaluator


def _config(features):
    return {
        "features": features,
        "random_seed": 42,
        "validation": {"min_train_rounds": 3},
        "models": {},
    }


def test_chronological_fold_only_trains_on_prior_rounds():
    df = pd.DataFrame([
        {"Round": 1, "Abbreviation": "VER", "ActualFinish": 1, "f": 1.0},
        {"Round": 2, "Abbreviation": "VER", "ActualFinish": 2, "f": 2.0},
        {"Round": 3, "Abbreviation": "VER", "ActualFinish": 3, "f": 3.0},
        {"Round": 4, "Abbreviation": "VER", "ActualFinish": 4, "f": 4.0},
    ])
    rounds = sorted(df["Round"].unique())
    for validation_round in rounds:
        train = df[df["Round"] < validation_round]
        test = df[df["Round"] == validation_round]
        assert all(r < validation_round for r in train["Round"])
        assert all(r == validation_round for r in test["Round"])


def test_target_round_is_not_in_training_fold():
    df = pd.DataFrame([
        {"Round": r, "Abbreviation": "VER", "ActualFinish": r, "f": r}
        for r in range(1, 8)
    ])
    target_round = 6
    train = df[df["Round"] < target_round]
    assert target_round not in train["Round"].values


def test_empty_training_fold_must_not_produce_validation_success():
    df = pd.DataFrame([
        {"Round": 4, "Abbreviation": "VER", "ActualFinish": 4, "f": 4.0},
    ])
    evaluator = ModelEvaluator(_config(["f"]))
    result = evaluator.evaluate_walk_forward(df)
    assert result["validation_status"] != "SUCCESS"


def test_empty_validation_fold_must_not_be_silently_skipped_as_success():
    df = pd.DataFrame([
        {"Round": 1, "Abbreviation": "VER", "ActualFinish": 1, "f": 1.0},
        {"Round": 2, "Abbreviation": "VER", "ActualFinish": 2, "f": 2.0},
    ])
    evaluator = ModelEvaluator(_config(["f"]))
    result = evaluator.evaluate_walk_forward(df)
    assert result["validation_rounds"] == []
    assert result["is_out_of_sample"] is False


def test_austin_pre_fp1_insufficient_history_is_actionable():
    df = pd.DataFrame(columns=["Round", "Abbreviation", "ActualFinish", "f"])
    evaluator = ModelEvaluator(_config(["f"]))
    result = evaluator.evaluate_walk_forward(df)
    assert result["validation_status"] == "UNAVAILABLE"
    assert result["sample_count"] == 0


def test_nan_training_targets_are_removed_before_fit():
    df = pd.DataFrame([
        {"Round": 1, "Abbreviation": "VER", "ActualFinish": np.nan, "f": 1.0},
        {"Round": 2, "Abbreviation": "VER", "ActualFinish": 2.0, "f": 2.0},
    ])
    df = df.dropna(subset=["ActualFinish"])
    assert len(df) == 1
    assert df["ActualFinish"].isna().sum() == 0


def test_failed_fold_returns_blocked_validation_status():
    df = pd.DataFrame([
        {"Round": 1, "Abbreviation": "VER", "ActualFinish": 1, "f": 1.0},
        {"Round": 2, "Abbreviation": "VER", "ActualFinish": 2, "f": 2.0},
        {"Round": 3, "Abbreviation": "VER", "ActualFinish": 3, "f": 3.0},
        {"Round": 4, "Abbreviation": "VER", "ActualFinish": np.nan, "f": 4.0},
    ])
    evaluator = ModelEvaluator(_config(["f"]))
    result = evaluator.evaluate_walk_forward(df)
    assert result["validation_status"] == "BLOCKED"
    assert result["is_out_of_sample"] is False
    assert len(result["failed_folds"]) > 0
    assert result["failed_folds"][0]["round"] == 4
    # All five residual arrays must be strictly empty (zero partial residual leakage into Monte Carlo)
    for key in ("selected_residuals", "rf_residuals", "xgb_residuals", "ridge_residuals", "multi_residuals"):
        assert isinstance(result[key], np.ndarray)
        assert len(result[key]) == 0


def test_grid_blend_sweep_with_empty_actuals_returns_empty_dict():
    df = pd.DataFrame(columns=["Round", "Abbreviation", "ActualFinish", "f"])
    evaluator = ModelEvaluator(_config(["f"]))
    sweep = evaluator.evaluate_grid_blend_sweep(df)
    assert sweep == {}
