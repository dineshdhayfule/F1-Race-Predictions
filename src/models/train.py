"""
Model Training Module
=====================
Fits machine learning models (Random Forest and XGBoost) on completed historical rounds.
Uses SimpleImputer for robust handling of missing weekend telemetry (e.g. Q3 laps for non-Q3 drivers).
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import xgboost as xgb

from src.utils.logging_utils import get_logger

logger = get_logger("model_train")


class ModelTrainer:
    """Trains and manages predictive models for F1 race finishing positions."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_cfg = config.get("models", {})
        self.feature_cols = config.get("features", config.get("feature_columns", []))
        self.random_state = config.get("random_seed", 42)


    def fit(
        self,
        training_df: pd.DataFrame
    ) -> Tuple[SimpleImputer, RandomForestRegressor, xgb.XGBRegressor, np.ndarray]:
        """
        Fits imputer, Random Forest, and XGBoost on the training dataframe.

        Parameters
        ----------
        training_df : pd.DataFrame
            Historical training features with 'ActualFinish' target.

        Returns
        -------
        tuple of (imputer, rf_model, xgb_model, feature_names)
        """
        if "ActualFinish" not in training_df.columns:
            raise ValueError("Training DataFrame must contain 'ActualFinish' target column.")

        # Ensure all required feature columns exist, filling missing with NaN for imputer
        for col in self.feature_cols:
            if col not in training_df.columns:
                training_df[col] = np.nan

        X = training_df[self.feature_cols].copy()
        y = pd.to_numeric(training_df["ActualFinish"], errors="coerce").values

        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]

        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)

        rf_cfg = self.model_cfg.get("random_forest", {})
        rf = RandomForestRegressor(
            n_estimators=rf_cfg.get("n_estimators", 500),
            max_depth=rf_cfg.get("max_depth", 8),
            min_samples_leaf=rf_cfg.get("min_samples_leaf", 2),
            random_state=self.random_state,
            n_jobs=-1,
        )
        logger.info(f"Fitting Random Forest on {len(X_imputed)} samples...")
        rf.fit(X_imputed, y)

        xgb_cfg = self.model_cfg.get("xgboost", {})
        n_est = xgb_cfg.get("final_n_estimators", xgb_cfg.get("n_estimators", 300))
        xgb_model = xgb.XGBRegressor(
            n_estimators=n_est,
            max_depth=xgb_cfg.get("max_depth", 4),
            learning_rate=xgb_cfg.get("learning_rate", 0.03),
            subsample=xgb_cfg.get("subsample", 0.8),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
            random_state=self.random_state,
            objective="reg:squarederror",
        )
        logger.info(f"Fitting XGBoost ({n_est} estimators) on {len(X_imputed)} samples...")

        xgb_model.fit(X_imputed, y)

        return imputer, rf, xgb_model, np.array(self.feature_cols)
