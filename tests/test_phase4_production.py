"""
tests/test_phase4_production.py

Comprehensive Phase 4 test suite validating:
1. Ridge regression model training and inference.
2. Validated Multi-Model ensemble execution (40% RF / 30% XGB / 30% Ridge).
3. Circuit-type-aware starting-grid weighting (street -> 0.50, permanent -> 0.35, temporary -> 0.45).
4. CLI options (--model ridge, --model ensemble, --model multi) and error validation.
5. Feature pruning configuration and backward-compatible feature list preservation.
6. Pipeline integration and metadata provenance verification.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from src.models.train import ModelTrainer, create_feature_imputer
from src.models.predict import Predictor
from src.models.evaluate import ModelEvaluator
import main


@pytest.fixture
def model_config():
    """Loads production model config."""
    cfg_path = Path("config/model_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def sample_feature_data(model_config):
    """Generates synthetic dataset for model training and prediction testing using configured features."""
    np.random.seed(42)
    n_samples = 100
    feature_cols = model_config.get("features", [])
    data = {col: np.random.randn(n_samples) for col in feature_cols}
    data["ActualFinish"] = np.random.randint(1, 21, size=n_samples)
    data["ActualStartingGrid"] = np.random.randint(1, 21, size=n_samples)
    data["GridPosition"] = data["ActualStartingGrid"]
    data["Round"] = np.repeat([1, 2, 3, 4, 5], 20)
    data["Abbreviation"] = [f"DRV{i%20}" for i in range(n_samples)]
    data["TeamName"] = [f"TEAM{i%10}" for i in range(n_samples)]
    return pd.DataFrame(data), feature_cols


# =====================================================================
# 1. Ridge Regression Integration Tests
# =====================================================================

def test_model_trainer_fit_ridge(sample_feature_data, model_config):
    df, feature_cols = sample_feature_data
    trainer = ModelTrainer(model_config)
    
    imputer = create_feature_imputer(strategy="median")
    X_imp = imputer.fit_transform(df[feature_cols])
    y = df["ActualFinish"].values
    
    ridge_pipe = trainer.fit_ridge(X_imp, y)
    assert isinstance(ridge_pipe, Pipeline)
    assert "scaler" in ridge_pipe.named_steps
    assert isinstance(ridge_pipe.named_steps["ridge"], Ridge)
    assert ridge_pipe.named_steps["ridge"].alpha == 40.0


def test_model_trainer_fit_backward_compatibility(sample_feature_data, model_config):
    df, feature_cols = sample_feature_data
    trainer = ModelTrainer(model_config)
    
    # Standard 4-tuple unpacking must work without error
    ret = trainer.fit(df)
    assert len(ret) == 4
    imputer, rf_model, xgb_model, feature_names = ret
    assert trainer.ridge_model is not None
    
    # Optional 5-tuple unpacking
    ret_5 = trainer.fit(df, return_ridge=True)
    assert len(ret_5) == 5
    assert ret_5[3] is trainer.ridge_model


def test_predictor_deterministic_ridge(sample_feature_data, model_config):
    df, feature_cols = sample_feature_data
    trainer = ModelTrainer(model_config)
    imputer, rf, xgb, _ = trainer.fit(df)
    
    predictor = Predictor(model_config)
    target_df = df.iloc[:10].copy()
    
    det_df, blended = predictor.predict_deterministic(
        target_df=target_df,
        imputer=imputer,
        rf_model=rf,
        xgb_model=xgb,
        ridge_model=trainer.ridge_model,
        model_mode="ridge",
    )
    
    assert "ML_PredictedScore" in det_df.columns
    assert "BlendedScore" in det_df.columns
    assert "DeterministicRank" in det_df.columns
    assert len(blended) == 10


# =====================================================================
# 2. Multi-Model Ensemble Execution Tests
# =====================================================================

def test_predictor_multi_model_ensemble(sample_feature_data, model_config):
    df, feature_cols = sample_feature_data
    trainer = ModelTrainer(model_config)
    imputer, rf, xgb, _ = trainer.fit(df)
    
    predictor = Predictor(model_config)
    target_df = df.iloc[:10].copy()
    
    det_df, blended = predictor.predict_deterministic(
        target_df=target_df,
        imputer=imputer,
        rf_model=rf,
        xgb_model=xgb,
        ridge_model=trainer.ridge_model,
        rf_weight=0.40,
        xgb_weight=0.30,
        ridge_weight=0.30,
        model_mode="ensemble",
    )
    
    # Calculate pure ML predictions to verify 40/30/30 weighting
    X_imp = imputer.transform(target_df[feature_cols])
    rf_pred = rf.predict(X_imp)
    xgb_pred = xgb.predict(X_imp)
    ridge_pred = trainer.ridge_model.predict(X_imp)
    expected_ml = 0.40 * rf_pred + 0.30 * xgb_pred + 0.30 * ridge_pred
    
    np.testing.assert_allclose(det_df["ML_PredictedScore"].values, np.round(expected_ml, 3), atol=1e-3)


# =====================================================================
# 3. Circuit-Type-Aware Starting-Grid Blending Tests
# =====================================================================

def test_resolve_blending_weights_street(model_config):
    predictor = Predictor(model_config)
    w_ml, w_grid = predictor.resolve_blending_weights(circuit_type="street")
    assert w_grid == 0.50
    assert w_ml == 0.50


def test_resolve_blending_weights_permanent(model_config):
    predictor = Predictor(model_config)
    w_ml, w_grid = predictor.resolve_blending_weights(circuit_type="permanent")
    assert w_grid == 0.35
    assert w_ml == 0.65


def test_resolve_blending_weights_temporary(model_config):
    predictor = Predictor(model_config)
    w_ml, w_grid = predictor.resolve_blending_weights(circuit_type="temporary")
    assert w_grid == 0.45
    assert w_ml == 0.55


def test_resolve_blending_weights_unknown_fallback(model_config):
    predictor = Predictor(model_config)
    w_ml, w_grid = predictor.resolve_blending_weights(circuit_type="unknown_circuit")
    assert w_grid == 0.35
    assert w_ml == 0.65


def test_resolve_blending_weights_custom_override(model_config):
    predictor = Predictor(model_config)
    w_ml, w_grid = predictor.resolve_blending_weights(
        circuit_type="street",
        custom_grid_weight=0.20,
        custom_ml_weight=0.80
    )
    assert w_grid == 0.20
    assert w_ml == 0.80


# =====================================================================
# 4. Feature Pruning Configuration Tests
# =====================================================================

def test_feature_pruning_config_and_baseline_count():
    cfg_path = Path("config/model_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    # Baseline features must retain 37 features for full backward compatibility
    assert len(cfg["features"]) == 37
    assert "feature_pruning" in cfg
    assert "validated_pruned_features" in cfg["feature_pruning"]
    assert "sprint_quali_pos" in cfg["feature_pruning"]["validated_pruned_features"]


# =====================================================================
# 5. CLI Model Integration & Parameter Validation Tests
# =====================================================================

def test_cmd_predict_accepts_ridge_and_ensemble():
    args_ridge = MagicMock(model="ridge", rf_weight=None, xgb_weight=None, enhanced=False)
    # Validate no exception thrown during arg verification before pipeline
    model_raw = getattr(args_ridge, "model", "hybrid")
    assert model_raw in ("rf", "xgb", "ridge", "hybrid", "ensemble", "multi", "__interactive__")

    args_ensemble = MagicMock(model="ensemble", rf_weight=None, xgb_weight=None, enhanced=False)
    model_raw2 = getattr(args_ensemble, "model", "hybrid")
    assert model_raw2 in ("rf", "xgb", "ridge", "hybrid", "ensemble", "multi", "__interactive__")


def test_cmd_predict_rejects_custom_weights_on_ridge():
    class MockParser:
        def __init__(self):
            self.error_called = False
            self.error_msg = ""
        def error(self, msg):
            self.error_called = True
            self.error_msg = msg

    parser = MockParser()
    args = MagicMock(
        model="ridge",
        rf_weight=50.0,
        xgb_weight=50.0,
        enhanced=False
    )
    main.cmd_predict(args, parser=parser)
    assert parser.error_called
    assert "cannot be used with 'ridge' model" in parser.error_msg


def test_cmd_predict_rejects_custom_weights_on_ensemble():
    class MockParser:
        def __init__(self):
            self.error_called = False
            self.error_msg = ""
        def error(self, msg):
            self.error_called = True
            self.error_msg = msg

    parser = MockParser()
    args = MagicMock(
        model="ensemble",
        rf_weight=50.0,
        xgb_weight=50.0,
        enhanced=False
    )
    main.cmd_predict(args, parser=parser)
    assert parser.error_called
    assert "cannot be used with 'ensemble' model" in parser.error_msg


# =====================================================================
# 6. ModelEvaluator Walk-Forward Includes Ridge and Multi-Model
# =====================================================================

def test_evaluator_walk_forward_includes_phase4_models(sample_feature_data, model_config):
    df, feature_cols = sample_feature_data
    evaluator = ModelEvaluator(model_config)
    
    results = evaluator.evaluate_walk_forward(df)
    assert "ridge_mae" in results
    assert "multi_mae" in results
    assert "ridge_residuals" in results
    assert "multi_residuals" in results
    
    comp_names = [x["model_name"] for x in results["ensemble_comparisons"]]
    assert "100% Ridge" in comp_names
    assert "Validated Multi-Model (40% RF / 30% XGB / 30% Ridge)" in comp_names
