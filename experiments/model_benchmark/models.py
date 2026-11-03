"""
Isolated Model Implementations and Wrappers for F1 Benchmark.
Ensures identical train/test data flow and strictly reproducible estimators.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from experiments.model_benchmark.config import (
    RANDOM_SEED,
    DEFAULT_RF_WEIGHT,
    DEFAULT_XGB_WEIGHT,
    DEFAULT_ML_WEIGHT,
    DEFAULT_GRID_WEIGHT,
    RIDGE_ALPHA,
    RF_PARAMS,
    XGB_PARAMS,
    MC_PARAMS,
)
from src.simulation.monte_carlo import MonteCarloSimulator


def get_grid_positions(df: pd.DataFrame) -> np.ndarray:
    """
    Extracts starting grid positions from a test dataframe following production rules:
    Prefers 'simulation_start_rank' (which places pit-lane starters at back),
    then 'ActualStartingGrid', then 'GridPosition', falling back to 12.0 for missing.
    """
    if "simulation_start_rank" in df.columns:
        s = pd.to_numeric(df["simulation_start_rank"], errors="coerce")
    elif "ActualStartingGrid" in df.columns:
        s = pd.to_numeric(df["ActualStartingGrid"], errors="coerce")
    elif "GridPosition" in df.columns:
        s = pd.to_numeric(df["GridPosition"], errors="coerce")
    else:
        return np.full(len(df), 12.0)

    # For pit lane starters not yet demoted in ActualStartingGrid
    if "is_pit_lane_start" in df.columns:
        is_pit = pd.to_numeric(df["is_pit_lane_start"], errors="coerce").fillna(0).astype(int) == 1
        if is_pit.any():
            max_grid = float(s.max() if pd.notna(s.max()) else 20.0)
            pit_indices = np.where(is_pit)[0]
            for idx_offset, pit_idx in enumerate(pit_indices):
                s.iloc[pit_idx] = max_grid + 1.0 + idx_offset

    return s.fillna(12.0).values


class StartingGridBaseline:
    """Model 1: Deterministic Starting Grid Baseline."""

    def __init__(self):
        self.name = "Starting Grid Baseline"

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray):
        """Zero parameter baseline; no training required."""
        return self

    def predict(self, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns continuous grid positions and ordinal starting ranks (1 to N).
        """
        grid_pos = get_grid_positions(X_test)
        ordinal_ranks = np.argsort(np.argsort(grid_pos)) + 1
        return grid_pos, ordinal_ranks


class RandomForestBenchmark:
    """Model 2: Independent Production Random Forest."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.name = "Random Forest"
        self.params = params or RF_PARAMS
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.model = RandomForestRegressor(**self.params)

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray):
        X_imp = self.imputer.fit_transform(X_train)
        self.model.fit(X_imp, y_train)
        return self

    def predict(self, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X_imp = self.imputer.transform(X_test)
        pred = self.model.predict(X_imp)
        ordinal_ranks = np.argsort(np.argsort(pred)) + 1
        return pred, ordinal_ranks


class XGBoostBenchmark:
    """Model 3: Independent Production XGBoost."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.name = "XGBoost"
        self.params = params or XGB_PARAMS
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.model = xgb.XGBRegressor(**self.params)

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray):
        X_imp = self.imputer.fit_transform(X_train)
        self.model.fit(X_imp, y_train)
        return self

    def predict(self, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X_imp = self.imputer.transform(X_test)
        pred = self.model.predict(X_imp)
        ordinal_ranks = np.argsort(np.argsort(pred)) + 1
        return pred, ordinal_ranks


class RidgeRegressionBenchmark:
    """Model 4: Standardized L2-Regularized Ridge Regression Pipeline."""

    def __init__(self, alpha: float = RIDGE_ALPHA, random_state: int = RANDOM_SEED):
        self.name = "Ridge Regression"
        self.alpha = alpha
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, random_state=random_state))
        ])

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray):
        self.pipeline.fit(X_train, y_train)
        return self

    def predict(self, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        pred = self.pipeline.predict(X_test)
        ordinal_ranks = np.argsort(np.argsort(pred)) + 1
        return pred, ordinal_ranks


class MLOnlyEnsembleBenchmark:
    """Model 5: Random Forest + XGBoost ML-only Ensemble (No Starting-Grid Blending)."""

    def __init__(
        self,
        rf_weight: float = DEFAULT_RF_WEIGHT,
        xgb_weight: float = DEFAULT_XGB_WEIGHT,
        rf_params: Optional[Dict[str, Any]] = None,
        xgb_params: Optional[Dict[str, Any]] = None
    ):
        self.name = "RF + XGBoost Ensemble (ML-only)"
        self.rf_weight = rf_weight
        self.xgb_weight = xgb_weight
        self.rf = RandomForestBenchmark(params=rf_params)
        self.xgb = XGBoostBenchmark(params=xgb_params)

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray):
        self.rf.fit(X_train, y_train)
        self.xgb.fit(X_train, y_train)
        return self

    def predict(self, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        rf_pred, _ = self.rf.predict(X_test)
        xgb_pred, _ = self.xgb.predict(X_test)
        ml_pred = self.rf_weight * rf_pred + self.xgb_weight * xgb_pred
        ordinal_ranks = np.argsort(np.argsort(ml_pred)) + 1
        return ml_pred, ordinal_ranks


class ProductionHybridBenchmark:
    """Model 6: Existing Production Hybrid (ML Ensemble + Grid Blending + Optional Monte Carlo)."""

    def __init__(
        self,
        rf_weight: float = DEFAULT_RF_WEIGHT,
        xgb_weight: float = DEFAULT_XGB_WEIGHT,
        ml_weight: float = DEFAULT_ML_WEIGHT,
        grid_weight: float = DEFAULT_GRID_WEIGHT,
        rf_params: Optional[Dict[str, Any]] = None,
        xgb_params: Optional[Dict[str, Any]] = None,
        mc_config: Optional[Dict[str, Any]] = None
    ):
        self.name = "Production Hybrid (ML + Grid)"
        self.ml_ensemble = MLOnlyEnsembleBenchmark(
            rf_weight=rf_weight,
            xgb_weight=xgb_weight,
            rf_params=rf_params,
            xgb_params=xgb_params
        )
        self.ml_weight = ml_weight
        self.grid_weight = grid_weight
        self.mc_config = mc_config or {
            "monte_carlo": MC_PARAMS,
            "random_seed": RANDOM_SEED
        }
        self.simulator = MonteCarloSimulator(self.mc_config)
        self.train_residuals: Optional[np.ndarray] = None
        self.empirical_dnf_rate: float = MC_PARAMS["fallback_dnf_prob"]

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, train_meta: Optional[pd.DataFrame] = None):
        self.ml_ensemble.fit(X_train, y_train)
        # Compute training residuals for Monte Carlo noise scaling
        rf_p, _ = self.ml_ensemble.rf.predict(X_train)
        xgb_p, _ = self.ml_ensemble.xgb.predict(X_train)
        ml_train = self.ml_ensemble.rf_weight * rf_p + self.ml_ensemble.xgb_weight * xgb_p
        grid_source = train_meta if (train_meta is not None and any(c in train_meta.columns for c in ["simulation_start_rank", "ActualStartingGrid", "GridPosition"])) else X_train
        grid_train = get_grid_positions(grid_source)
        blended_train = self.ml_weight * ml_train + self.grid_weight * grid_train
        self.train_residuals = y_train - blended_train

        if train_meta is not None and "driver_dnf_rate" in train_meta.columns:
            self.empirical_dnf_rate = float(train_meta["driver_dnf_rate"].mean())

        return self

    def predict(self, X_test: pd.DataFrame, test_meta: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray]:
        ml_pred, _ = self.ml_ensemble.predict(X_test)
        grid_source = test_meta if (test_meta is not None and any(c in test_meta.columns for c in ["simulation_start_rank", "ActualStartingGrid", "GridPosition"])) else X_test
        grid_pos = get_grid_positions(grid_source)
        blended = self.ml_weight * ml_pred + self.grid_weight * grid_pos
        ordinal_ranks = np.argsort(np.argsort(blended)) + 1
        return blended, ordinal_ranks

    def simulate_probabilities(
        self,
        X_test: pd.DataFrame,
        blended_scores: np.ndarray,
        test_meta: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Executes production Monte Carlo race simulation (25k iterations)
        yielding calibrated probabilities.
        """
        meta = test_meta.copy() if test_meta is not None else X_test.copy()
        if "Abbreviation" not in meta.columns and "Abbreviation" in X_test.columns:
            meta["Abbreviation"] = X_test["Abbreviation"]
        if "TeamName" not in meta.columns and "TeamName" in X_test.columns:
            meta["TeamName"] = X_test["TeamName"]

        sim_df, _ = self.simulator.simulate(
            target_df=meta,
            deterministic_scores=blended_scores,
            residuals=self.train_residuals,
            empirical_dnf_rate=self.empirical_dnf_rate
        )
        return sim_df
