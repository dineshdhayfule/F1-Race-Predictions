"""
Model Training Module
=====================
Fits machine learning models (Random Forest, XGBoost, and Ridge Regression) on completed historical rounds.
Uses SimpleImputer for robust handling of missing weekend telemetry (e.g. Q3 laps for non-Q3 drivers).
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.utils.logging_utils import get_logger

logger = get_logger("model_train")


def create_feature_imputer(strategy: str = "median") -> SimpleImputer:
    """
    Creates a SimpleImputer configured to retain empty (all-NaN) feature columns,
    preventing dimension mismatches and scikit-learn all-NaN UserWarnings across
    early prediction stages and walk-forward training folds.
    """
    try:
        return SimpleImputer(strategy=strategy, keep_empty_features=True)
    except TypeError:
        return SimpleImputer(strategy=strategy)


class ModelTrainer:
    """Trains and manages predictive models for F1 race finishing positions."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_cfg = config.get("models", {})
        self.feature_cols = config.get("features", config.get("feature_columns", []))
        self.random_state = config.get("random_seed", 42)
        self.ridge_model: Optional[Pipeline] = None

    def fit_ridge(self, X_imputed: np.ndarray, y: np.ndarray) -> Pipeline:
        """
        Fits standardized L2-regularized Ridge Regression pipeline on imputed features.
        Scaling is fitted strictly on training data; alpha is pulled from validated config.
        """
        ridge_cfg = self.model_cfg.get("ridge", {})
        alpha = float(ridge_cfg.get("alpha", 40.0))
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, random_state=self.random_state))
        ])
        pipe.fit(X_imputed, y)
        return pipe

    def fit(
        self,
        training_df: pd.DataFrame,
        return_ridge: bool = False
    ) -> Union[
        Tuple[SimpleImputer, RandomForestRegressor, xgb.XGBRegressor, np.ndarray],
        Tuple[SimpleImputer, RandomForestRegressor, xgb.XGBRegressor, Pipeline, np.ndarray]
    ]:
        """
        Fits imputer, Random Forest, XGBoost, and Ridge Regression on the training dataframe.

        Parameters
        ----------
        training_df : pd.DataFrame
            Historical training features with 'ActualFinish' target.
        return_ridge : bool, optional
            If True, returns 5-tuple including fitted Ridge pipeline.
            Defaults to False for strict 4-tuple backward compatibility.

        Returns
        -------
        tuple of (imputer, rf_model, xgb_model, feature_names) if return_ridge=False,
        or (imputer, rf_model, xgb_model, ridge_model, feature_names) if return_ridge=True.
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

        imputer = create_feature_imputer(strategy="median")
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

        # Fit Ridge Regression
        self.ridge_model = self.fit_ridge(X_imputed, y)
        logger.info(f"Fitting Ridge Regression (alpha={self.model_cfg.get('ridge', {}).get('alpha', 40.0)}) on {len(X_imputed)} samples...")

        if return_ridge:
            return imputer, rf, xgb_model, self.ridge_model, np.array(self.feature_cols)

        return imputer, rf, xgb_model, np.array(self.feature_cols)
