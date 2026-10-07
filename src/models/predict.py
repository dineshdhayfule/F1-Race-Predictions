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
        self.grid_weight = blend_cfg.get("grid_weight", self.ens_cfg.get("grid_weight", 0.30))
        self.ml_weight = blend_cfg.get("ml_weight", self.ens_cfg.get("ml_weight", 0.70))


    def predict_deterministic(
        self,
        target_df: pd.DataFrame,
        imputer: SimpleImputer,
        rf_model: RandomForestRegressor,
        xgb_model: Optional[xgb.XGBRegressor] = None,
        rf_weight: Optional[float] = None,
        xgb_weight: Optional[float] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Calculates deterministic finishing position expectations for each driver.

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

        Returns
        -------
        tuple of (results_df, blended_deterministic_scores)
        """
        sel_cfg = self.config.get("model_selection", {})
        mode = sel_cfg.get("mode", "walk_forward_auto")

        if rf_weight is not None and xgb_weight is not None:
            w_rf = float(rf_weight)
            w_xgb = float(xgb_weight)
        elif mode == "fixed_rf":
            w_rf, w_xgb = 1.0, 0.0
        elif mode == "fixed_xgb":
            w_rf, w_xgb = 0.0, 1.0
        elif mode == "fixed_ensemble":
            w_rf = float(sel_cfg.get("fixed_rf_weight", 0.6))
            w_xgb = float(sel_cfg.get("fixed_xgb_weight", 0.4))
        else:
            w_rf = float(self.ens_cfg.get("rf_weight", 1.0))
            w_xgb = float(self.ens_cfg.get("xgb_weight", 0.0))

        for col in self.feature_cols:
            if col not in target_df.columns:
                target_df[col] = np.nan

        X = target_df[self.feature_cols].copy()
        X_imputed = imputer.transform(X)

        rf_pred = rf_model.predict(X_imputed)

        if xgb_model is not None and w_xgb > 0:
            xgb_pred = xgb_model.predict(X_imputed)
            ml_pred = w_rf * rf_pred + w_xgb * xgb_pred
        else:
            ml_pred = rf_pred

        grid_positions = pd.to_numeric(target_df["GridPosition"], errors="coerce").fillna(12.0).values
        # Blended deterministic baseline: ml_weight * ML + grid_weight * GridPosition
        blended = self.ml_weight * ml_pred + self.grid_weight * grid_positions


        out = target_df[["Abbreviation", "TeamName", "GridPosition", "QualifyingPosition"]].copy()
        out["ML_PredictedScore"] = np.round(ml_pred, 3)
        out["BlendedScore"] = np.round(blended, 3)
        out["DeterministicRank"] = np.argsort(np.argsort(blended)) + 1

        return out, blended
