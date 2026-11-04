"""
Inference and Deterministic Prediction Module
=============================================
Generates point predictions from trained estimators and produces blended grid expectations.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import xgboost as xgb

from src.utils.logging_utils import get_logger

logger = get_logger("model_predict")


class Predictor:
    """Generates deterministic finishing predictions for target race."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_cols = config.get("features", config.get("feature_columns", []))
        self.ens_cfg = config.get("models", {}).get("ensemble", {})
        blend_cfg = config.get("blending", {})
        self.rf_weight = self.ens_cfg.get("rf_weight", 1.0)
        self.xgb_weight = self.ens_cfg.get("xgb_weight", 0.0)
        self.grid_weight = blend_cfg.get("grid_weight", self.ens_cfg.get("grid_weight", 0.35))
        self.ml_weight = blend_cfg.get("ml_weight", self.ens_cfg.get("ml_weight", 0.65))
        self.circuit_aware = blend_cfg.get("circuit_aware", True)
        self.circuit_weights = blend_cfg.get("circuit_weights", {
            "street": {"grid_weight": 0.50, "ml_weight": 0.50},
            "permanent": {"grid_weight": 0.35, "ml_weight": 0.65},
            "temporary": {"grid_weight": 0.45, "ml_weight": 0.55},
            "default": {"grid_weight": 0.35, "ml_weight": 0.65}
        })

    def resolve_blending_weights(
        self,
        circuit_type: Optional[str] = None,
        custom_grid_weight: Optional[float] = None,
        custom_ml_weight: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Resolves ML and starting-grid blending weights.
        Prioritizes custom caller weights, then circuit-type specific rules, then config defaults.
        """
        if custom_grid_weight is not None and custom_ml_weight is not None:
            return float(custom_ml_weight), float(custom_grid_weight)

        if self.circuit_aware and circuit_type:
            c_key = str(circuit_type).lower().strip()
            if c_key in self.circuit_weights:
                cw = self.circuit_weights[c_key]
                return float(cw.get("ml_weight", self.ml_weight)), float(cw.get("grid_weight", self.grid_weight))

        return float(self.ml_weight), float(self.grid_weight)

    def predict_deterministic(
        self,
        target_df: pd.DataFrame,
        imputer: SimpleImputer,
        rf_model: RandomForestRegressor,
        xgb_model: Optional[xgb.XGBRegressor] = None,
        rf_weight: Optional[float] = None,
        xgb_weight: Optional[float] = None,
        ridge_model: Optional[Any] = None,
        ridge_weight: Optional[float] = None,
        circuit_type: Optional[str] = None,
        grid_weight: Optional[float] = None,
        ml_weight: Optional[float] = None,
        model_mode: Optional[str] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Calculates deterministic finishing position expectations for each driver.
        Supports RF, XGBoost, Ridge Regression, 2-way Hybrids, and 3-way Multi-Model Ensembles.

        Parameters
        ----------
        target_df : pd.DataFrame
            Target features for upcoming race.
        imputer : SimpleImputer
            Fitted imputer from training pipeline.
        rf_model : RandomForestRegressor
            Fitted Random Forest model.
        xgb_model : XGBRegressor, optional
            Fitted XGBoost model.
        rf_weight : float, optional
            Dynamic or configured Random Forest weight.
        xgb_weight : float, optional
            Dynamic or configured XGBoost weight.
        ridge_model : Pipeline or Ridge, optional
            Fitted Ridge Regression model pipeline.
        ridge_weight : float, optional
            Ridge Regression weight in multi-model ensemble.
        circuit_type : str, optional
            Circuit type ('permanent', 'street', 'temporary') for circuit-aware grid blending.
        grid_weight : float, optional
            Override starting-grid blending weight.
        ml_weight : float, optional
            Override ML blending weight.
        model_mode : str, optional
            Explicit model mode ('rf', 'xgb', 'ridge', 'hybrid', 'ensemble', 'multi').

        Returns
        -------
        tuple of (results_df, blended_deterministic_scores)
        """
        sel_cfg = self.config.get("model_selection", {})
        mode = (model_mode or sel_cfg.get("mode", "fixed_ensemble")).lower().strip()

        for col in self.feature_cols:
            if col not in target_df.columns:
                target_df[col] = np.nan

        X = target_df[self.feature_cols].copy()
        X_imputed = imputer.transform(X)

        rf_pred = rf_model.predict(X_imputed)
        xgb_pred = xgb_model.predict(X_imputed) if xgb_model is not None else rf_pred
        ridge_pred = ridge_model.predict(X_imputed) if ridge_model is not None else rf_pred

        # Determine ML prediction based on mode and weights
        if mode == "ridge" or (ridge_weight is not None and ridge_weight >= 0.99):
            ml_pred = ridge_pred
        elif mode in ("ensemble", "multi"):
            w_rf = 0.40 if rf_weight is None else float(rf_weight)
            w_xgb = 0.30 if xgb_weight is None else float(xgb_weight)
            w_ridge = 0.30 if ridge_weight is None else float(ridge_weight)
            total_w = w_rf + w_xgb + w_ridge
            if total_w > 0:
                w_rf, w_xgb, w_ridge = w_rf / total_w, w_xgb / total_w, w_ridge / total_w
            ml_pred = w_rf * rf_pred + w_xgb * xgb_pred + w_ridge * ridge_pred
        elif mode == "fixed_rf" or mode == "rf":
            ml_pred = rf_pred
        elif mode == "fixed_xgb" or mode == "xgb":
            ml_pred = xgb_pred
        elif rf_weight is not None and xgb_weight is not None:
            try:
                w_rf = float(rf_weight)
                w_xgb = float(xgb_weight)
                w_ridge = float(ridge_weight) if ridge_weight is not None else 0.0
                if np.isnan(w_rf) or w_rf < 0:
                    w_rf = 0.0
                if np.isnan(w_xgb) or w_xgb < 0:
                    w_xgb = 0.0
                if np.isnan(w_ridge) or w_ridge < 0:
                    w_ridge = 0.0
            except (ValueError, TypeError):
                w_rf, w_xgb, w_ridge = 0.60, 0.40, 0.0

            total_w = w_rf + w_xgb + w_ridge
            if total_w <= 1e-6:
                logger.warning("Deterministic weights sum to 0; falling back to 0.60 RF / 0.40 XGB")
                w_rf, w_xgb, w_ridge = 0.60, 0.40, 0.0
                total_w = 1.0

            w_rf, w_xgb, w_ridge = w_rf / total_w, w_xgb / total_w, w_ridge / total_w
            ml_pred = w_rf * rf_pred + w_xgb * xgb_pred + w_ridge * ridge_pred
        elif mode == "fixed_ensemble":
            w_rf = float(sel_cfg.get("fixed_rf_weight", 0.6))
            w_xgb = float(sel_cfg.get("fixed_xgb_weight", 0.4))
            ml_pred = w_rf * rf_pred + w_xgb * xgb_pred
        else:
            w_rf = float(self.ens_cfg.get("rf_weight", 1.0))
            w_xgb = float(self.ens_cfg.get("xgb_weight", 0.0))
            ml_pred = w_rf * rf_pred + w_xgb * xgb_pred if w_xgb > 0 else rf_pred

        # Resolve circuit-type-aware blending weights
        eff_ml_w, eff_grid_w = self.resolve_blending_weights(
            circuit_type=circuit_type,
            custom_grid_weight=grid_weight,
            custom_ml_weight=ml_weight
        )

        # Blended deterministic baseline: eff_ml_w * ML + eff_grid_w * GridPosition
        # Use simulation_start_rank if present (handles pit-lane starters), else GridPosition
        grid_col = "simulation_start_rank" if "simulation_start_rank" in target_df.columns else "GridPosition"
        grid_positions = pd.to_numeric(target_df.get(grid_col, target_df.get("GridPosition")), errors="coerce").fillna(12.0).values
        blended = eff_ml_w * ml_pred + eff_grid_w * grid_positions


        out = target_df[["Abbreviation", "TeamName", "GridPosition", "QualifyingPosition"]].copy()
        out["ML_PredictedScore"] = np.round(ml_pred, 3)
        out["BlendedScore"] = np.round(blended, 3)
        out["DeterministicRank"] = np.argsort(np.argsort(blended)) + 1

        return out, blended
