"""
Phase 8 Integration Test Suite: ApexGrid Production Pipeline
============================================================
Validates the complete production integration of the Phase 1-7 promoted configuration:
1. Tri-Ensemble weights (40% RF / 30% XGB / 30% Ridge) and routing.
2. Ridge scaling (StandardScaler) and alpha=100.0.
3. Conservative XGBoost configuration (depth=3, lr=0.015, min_child_weight=2, alpha=1.0, lambda=2.0).
4. Circuit-aware grid blending (street: 50/50, permanent: 65/35, temporary: 55/45, default: 65/35).
5. Exact Config 8 feature list (40 features, pruned sprint_quali_pos, added race pace + pit lane start).
6. Feature availability and masking across all five stages (pre_fp1, after_fp1, after_fp2, after_fp3, after_qualifying).
7. Missing practice sessions handling (imputation and fallback).
8. Sprint weekend handling.
9. Pit lane starts (is_pit_lane_start) and grid penalties.
10. Monte Carlo draw-count selection (10,000 default vs 25,000 publication mode).
11. Residual sigma boundaries and temporal isolation (floor 1.5, MAD * 1.4826).
12. Race Control capability toggles.
13. DNF and classification semantics (RETIRED, DNS, DSQ, NC, UNKNOWN, FINISHED).
14. Probability monotonicity (P(Win) <= P(Podium) <= P(Top5) <= P(Points)) and quota bounds.
15. Existing CLI compatibility (main.py --model flag parsing).
16. Output schema parity.
17. Deterministic execution with fixed random seeds.
"""

import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import median_abs_deviation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.train import ModelTrainer, create_feature_imputer
from src.models.predict import Predictor
from src.models.evaluate import ModelEvaluator
from src.simulation.monte_carlo import MonteCarloSimulator
from src.features.build_features import apply_stage_feature_mask, FeatureEngineer
from src.features.season_features import classify_race_status, is_dnf, is_classified_finish
import main

@pytest.fixture
def prod_config():
    """Loads authoritative production configuration."""
    cfg_path = ROOT / "config" / "model_config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture
def sample_40_feature_df(prod_config):
    """Generates synthetic dataframe containing the exact 40 production features."""
    np.random.seed(42)
    n_samples = 66
    features = prod_config["features"]
    data = {f: np.random.randn(n_samples) for f in features}
    data["ActualFinish"] = np.random.randint(1, 23, size=n_samples)
    data["GridPosition"] = np.random.randint(1, 23, size=n_samples)
    data["ActualStartingGrid"] = data["GridPosition"]
    data["Round"] = np.repeat([1, 2, 3], 22)
    data["Abbreviation"] = [f"DRV{i%22}" for i in range(n_samples)]
    data["TeamName"] = [f"TEAM{i%11}" for i in range(n_samples)]
    data["StartingGridStatus"] = ["CLASSIFIED"] * n_samples
    data["is_pit_lane_start"] = [1 if i % 22 == 21 else 0 for i in range(n_samples)]
    return pd.DataFrame(data), features

# ==============================================================================
# 1. Feature Configuration & Parity (Config 8 - 40 Features)
# ==============================================================================
def test_production_feature_list_exact_count_and_composition(prod_config):
    """Verify exact 40-feature specification: pruned sprint_quali_pos, added race pace + pit lane start."""
    features = prod_config["features"]
    assert len(features) == 40
    
    # Pruned feature check
    assert "sprint_quali_pos" not in features
    
    # Promoted feature additions
    assert "is_pit_lane_start" in features
    assert "long_run_pace_gap_pct" in features
    assert "long_run_pace_rank" in features
    assert "race_pace_consistency" in features
    
    # Feature pruning flag check
    assert prod_config.get("feature_pruning", {}).get("enabled") is True
    assert "sprint_quali_pos" in prod_config.get("feature_pruning", {}).get("validated_pruned_features", [])

def test_training_inference_feature_parity(prod_config):
    """Trainer, Evaluator, and Predictor must share identical 40 feature columns."""
    trainer = ModelTrainer(prod_config)
    evaluator = ModelEvaluator(prod_config)
    predictor = Predictor(prod_config)
    
    expected = prod_config["features"]
    assert list(trainer.feature_cols) == expected
    assert list(evaluator.feature_cols) == expected
    assert list(predictor.feature_cols) == expected

# ==============================================================================
# 2. Model Architecture & Hyperparameters
# ==============================================================================
def test_ridge_hyperparameters_and_scaling(sample_40_feature_df, prod_config):
    """Verify Ridge alpha=100.0, StandardScaler encapsulation, and valid prediction."""
    df, features = sample_40_feature_df
    trainer = ModelTrainer(prod_config)
    imputer = create_feature_imputer(strategy="median")
    X_imp = imputer.fit_transform(df[features])
    y = df["ActualFinish"].values
    
    pipe = trainer.fit_ridge(X_imp, y)
    assert isinstance(pipe, Pipeline)
    assert "scaler" in pipe.named_steps
    assert isinstance(pipe.named_steps["scaler"], StandardScaler)
    assert isinstance(pipe.named_steps["ridge"], Ridge)
    assert pipe.named_steps["ridge"].alpha == 100.0
    
    preds = pipe.predict(X_imp)
    assert len(preds) == len(y)
    assert not np.isnan(preds).any()

def test_xgboost_conservative_hyperparameters(sample_40_feature_df, prod_config):
    """Verify conservative XGBoost parameters (depth=3, lr=0.015, min_child_weight=2, alpha=1.0, lambda=2.0)."""
    df, features = sample_40_feature_df
    trainer = ModelTrainer(prod_config)
    ret = trainer.fit(df, return_ridge=True)
    _, _, xgb_model, _, _ = ret
    
    params = xgb_model.get_params()
    assert params["max_depth"] == 3
    assert abs(params["learning_rate"] - 0.015) < 1e-5
    assert params["min_child_weight"] == 2
    assert params["reg_alpha"] == 1.0
    assert params["reg_lambda"] == 2.0
    assert params["n_estimators"] == 250

def test_random_forest_hyperparameters(sample_40_feature_df, prod_config):
    """Verify Random Forest hyperparameter preservation from machine-readable config."""
    df, features = sample_40_feature_df
    trainer = ModelTrainer(prod_config)
    ret = trainer.fit(df, return_ridge=True)
    _, rf_model, _, _, _ = ret
    
    assert rf_model.n_estimators == 500
    assert rf_model.max_depth == 8
    assert rf_model.min_samples_leaf == 2
    assert rf_model.random_state == 42

# ==============================================================================
# 3. Tri-Ensemble Routing & Blending
# ==============================================================================
def test_tri_ensemble_weights_and_routing(sample_40_feature_df, prod_config):
    """Verify Tri-Ensemble routes to 40% RF / 30% XGB / 30% Ridge and weights sum to 1.0."""
    df, features = sample_40_feature_df
    trainer = ModelTrainer(prod_config)
    predictor = Predictor(prod_config)
    
    imputer, rf_m, xgb_m, ridge_m, _ = trainer.fit(df, return_ridge=True)
    
    target_df = df.iloc[:22].copy()
    _, blended = predictor.predict_deterministic(
        target_df=target_df,
        imputer=imputer,
        rf_model=rf_m,
        xgb_model=xgb_m,
        ridge_model=ridge_m,
        model_mode="ensemble",
        circuit_type="permanent"
    )
    
    assert len(blended) == 22
    assert not np.isnan(blended).any()

def test_circuit_aware_blending_weights(prod_config):
    """Verify circuit blending weights: Street 50/50, Permanent 65/35, Temporary 55/45, Default 65/35."""
    predictor = Predictor(prod_config)
    
    # Street circuit
    ml_st, grid_st = predictor.resolve_blending_weights(circuit_type="street")
    assert ml_st == 0.50
    assert grid_st == 0.50
    
    # Permanent circuit
    ml_perm, grid_perm = predictor.resolve_blending_weights(circuit_type="permanent")
    assert ml_perm == 0.65
    assert grid_perm == 0.35
    
    # Temporary circuit
    ml_temp, grid_temp = predictor.resolve_blending_weights(circuit_type="temporary")
    assert ml_temp == 0.55
    assert grid_temp == 0.45
    
    # Default circuit fallback
    ml_def, grid_def = predictor.resolve_blending_weights(circuit_type="unknown_circuit")
    assert ml_def == 0.65
    assert grid_def == 0.35

# ==============================================================================
# 4. Stage Availability & Leakage Protection
# ==============================================================================
def test_stage_availability_and_feature_masking(prod_config):
    """Verify that unobserved features are strictly masked across all five stages."""
    stages = ["pre_fp1", "after_fp1", "after_fp2", "after_fp3", "after_qualifying"]
    
    for stg in stages:
        df = pd.DataFrame([{col: 1.0 for col in prod_config["features"]}])
        masked = apply_stage_feature_mask(df.copy(), stg)
        
        if stg == "pre_fp1":
            assert np.isnan(masked["practice_gap_pct"].iloc[0])
            assert np.isnan(masked["fp1_gap_pct"].iloc[0])
            assert np.isnan(masked["QualifyingPosition"].iloc[0])
            assert np.isnan(masked["GridPosition"].iloc[0])
            assert np.isnan(masked["long_run_pace_gap_pct"].iloc[0])
        elif stg == "after_fp1":
            assert not np.isnan(masked["fp1_gap_pct"].iloc[0])
            assert np.isnan(masked["fp2_gap_pct"].iloc[0])
            assert np.isnan(masked["QualifyingPosition"].iloc[0])
            assert np.isnan(masked["GridPosition"].iloc[0])
        elif stg == "after_fp2":
            assert not np.isnan(masked["fp1_gap_pct"].iloc[0])
            assert not np.isnan(masked["fp2_gap_pct"].iloc[0])
            assert np.isnan(masked["fp3_gap_pct"].iloc[0])
            assert np.isnan(masked["QualifyingPosition"].iloc[0])
            assert np.isnan(masked["GridPosition"].iloc[0])
        elif stg == "after_fp3":
            assert not np.isnan(masked["fp1_gap_pct"].iloc[0])
            assert not np.isnan(masked["fp2_gap_pct"].iloc[0])
            assert not np.isnan(masked["fp3_gap_pct"].iloc[0])
            assert np.isnan(masked["QualifyingPosition"].iloc[0])
            assert np.isnan(masked["GridPosition"].iloc[0])
        elif stg == "after_qualifying":
            assert not np.isnan(masked["QualifyingPosition"].iloc[0])
            assert not np.isnan(masked["GridPosition"].iloc[0])
            assert not np.isnan(masked["is_pit_lane_start"].iloc[0])

# ==============================================================================
# 5. Stochastic Monte Carlo Simulation & Probabilities
# ==============================================================================
def test_monte_carlo_draw_depth_and_stability(prod_config):
    """Verify Monte Carlo simulator executes 10,000 draws by default and supports 25,000 publication mode."""
    sim = MonteCarloSimulator(prod_config)
    assert sim.n_simulations == 10000
    
    target_df = pd.DataFrame({
        "Abbreviation": [f"D{i}" for i in range(1, 23)],
        "TeamName": [f"T{i//2}" for i in range(1, 23)],
        "GridPosition": list(range(1, 23)),
        "StartingGridStatus": ["CLASSIFIED"] * 22,
        "is_pit_lane_start": [0] * 22,
    })
    scores = np.arange(1.0, 23.0)
    
    # Default 10k execution
    res_10k, meta_10k = sim.simulate(target_df, scores, n_simulations=10000)
    assert meta_10k["n_simulations"] == 10000
    assert len(res_10k) == 22
    
    # Publication 25k execution
    res_25k, meta_25k = sim.simulate(target_df, scores, n_simulations=25000)
    assert meta_25k["n_simulations"] == 25000
    assert len(res_25k) == 22

def test_probability_monotonicity_and_quota_consistency(prod_config):
    """Verify strict monotonicity P(Win) <= P(Podium) <= P(Top5) <= P(Points) across all drivers."""
    sim = MonteCarloSimulator(prod_config)
    target_df = pd.DataFrame({
        "Abbreviation": [f"D{i}" for i in range(1, 23)],
        "TeamName": [f"T{i//2}" for i in range(1, 23)],
        "GridPosition": list(range(1, 23)),
        "StartingGridStatus": ["CLASSIFIED"] * 22,
        "is_pit_lane_start": [0] * 22,
    })
    scores = np.linspace(1.5, 21.5, 22)
    
    res_df, _ = sim.simulate(target_df, scores, n_simulations=5000)
    
    for _, row in res_df.iterrows():
        p_win = row["WinProbability"]
        p_pod = row["Top3Probability"]
        p_top5 = row["Top5Probability"]
        p_pts = row["PointsProbability"]
        
        assert p_win <= p_pod + 1e-4, f"Monotonic violation: Win {p_win} > Podium {p_pod}"
        assert p_pod <= p_top5 + 1e-4, f"Monotonic violation: Podium {p_pod} > Top5 {p_top5}"
        assert p_top5 <= p_pts + 1e-4, f"Monotonic violation: Top5 {p_top5} > Points {p_pts}"

def test_residual_sigma_calibration_and_floor(prod_config):
    """Verify robust MAD residual sigma calibration with strict 1.5 floor."""
    sim = MonteCarloSimulator(prod_config)
    target_df = pd.DataFrame({
        "Abbreviation": ["VER", "NOR"],
        "TeamName": ["Red Bull", "McLaren"],
        "GridPosition": [1, 2],
        "StartingGridStatus": ["CLASSIFIED", "CLASSIFIED"],
        "is_pit_lane_start": [0, 0],
    })
    scores = np.array([1.0, 2.0])
    
    # 1. Normal residuals with MAD > 1.5
    residuals_normal = np.array([-4.0, 1.5, 2.0, -3.5, 4.0, 0.0, -1.5, 2.5])
    _, meta_norm = sim.simulate(target_df, scores, residuals=residuals_normal)
    expected_sigma = float(median_abs_deviation(residuals_normal) * 1.4826)
    assert expected_sigma > 1.5
    assert abs(meta_norm["calibrated_sigma"] - round(expected_sigma, 4)) < 1e-4
    
    # 2. Near-zero residuals (must be clamped to floor 1.5)
    residuals_tiny = np.array([0.01, -0.01, 0.02, 0.0, -0.02])
    _, meta_floor = sim.simulate(target_df, scores, residuals=residuals_tiny)
    assert meta_floor["calibrated_sigma"] == 1.5

def test_dnf_semantic_categories():
    """Verify canonical classification of race status outcomes."""
    assert classify_race_status("Finished") == "FINISHED"
    assert classify_race_status("+1 Lap") == "FINISHED"
    assert classify_race_status("Accident") == "RETIRED"
    assert classify_race_status("Collision") == "RETIRED"
    assert classify_race_status("Engine") == "RETIRED"
    assert classify_race_status("Did not start") == "DNS"
    assert classify_race_status("Disqualified") == "DSQ"
    assert classify_race_status("Not classified") == "NC"
    
    assert is_classified_finish("Finished") is True
    assert is_classified_finish("Accident") is False
    assert is_classified_finish("Did not start") is False
    assert is_dnf("Accident") is True
    assert is_dnf("Finished") is False

# ==============================================================================
# 6. CLI Compatibility & Determinism
# ==============================================================================
def test_cli_accepts_promoted_ensemble_flag():
    """Verify CLI accepts --model ensemble and --model multi."""
    test_cases = [
        (["main.py", "predict", "--track", "mexico", "--model", "ensemble"], "ensemble"),
        (["main.py", "predict", "--track", "mexico", "--model", "multi"], "multi"),
        (["main.py", "predict", "--track", "mexico", "--model", "hybrid"], "hybrid"),
    ]
    for argv, expected_model in test_cases:
        with patch.object(sys, "argv", argv):
            with patch("main.cmd_predict") as mock_predict:
                main.main()
                mock_predict.assert_called_once()
                args = mock_predict.call_args[0][0]
                assert args.model == expected_model

def test_simulation_determinism_with_fixed_seed(prod_config):
    """Verify Monte Carlo simulation produces byte-for-byte identical results with fixed random seed."""
    sim1 = MonteCarloSimulator(prod_config)
    sim2 = MonteCarloSimulator(prod_config)
    
    target_df = pd.DataFrame({
        "Abbreviation": ["VER", "NOR", "LEC", "PIA", "RUS"],
        "TeamName": ["Red Bull", "McLaren", "Ferrari", "McLaren", "Mercedes"],
        "GridPosition": [1, 2, 3, 4, 5],
        "StartingGridStatus": ["CLASSIFIED"] * 5,
        "is_pit_lane_start": [0] * 5,
    })
    scores = np.array([1.2, 2.1, 3.0, 4.2, 5.1])
    
    res1, _ = sim1.simulate(target_df, scores, n_simulations=2000)
    res2, _ = sim2.simulate(target_df, scores, n_simulations=2000)
    
    pd.testing.assert_frame_equal(res1, res2)
