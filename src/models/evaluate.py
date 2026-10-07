"""
Walk-Forward Model Evaluation Module
====================================
Performs chronological out-of-fold walk-forward validation across season rounds.
Calculates MAE, RMSE, Spearman correlation, Top-K overlap, and calibration residuals.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import xgboost as xgb

from src.utils.logging_utils import get_logger

logger = get_logger("model_evaluate")


def top_k_overlap(pred_series: pd.Series, actual_series: pd.Series, k: int) -> float:
    """Computes percentage overlap between predicted top-k and actual top-k drivers."""
    pred_top = set(pred_series.nsmallest(k).index)
    actual_top = set(actual_series.nsmallest(k).index)
    return len(pred_top & actual_top) / float(k)


class ModelEvaluator:
    """Evaluates predictive models using chronological walk-forward cross-validation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_cols = config.get("features", config.get("feature_columns", []))
        self.random_state = config.get("random_seed", 42)
        self.val_cfg = config.get("validation", {})

        self.min_train_rounds = self.val_cfg.get("min_train_rounds", 3)
        self.model_cfg = config.get("models", {})

    def evaluate_walk_forward(
        self,
        training_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Executes chronological walk-forward validation.
        For each round r (where r > min_train_rounds):
          - Train on rounds < r
          - Test on round r
        """
        rounds = sorted([int(r) for r in training_df["Round"].dropna().unique()])
        val_rounds = [r for r in rounds if r > self.min_train_rounds]

        logger.info(f"Starting Walk-Forward Validation across rounds: {val_rounds}")

        tested_weights = [
            (1.0, 0.0, "100% RF"),
            (0.75, 0.25, "75% RF / 25% XGB"),
            (0.60, 0.40, "60% RF / 40% XGB"),
            (0.50, 0.50, "50% RF / 50% XGB"),
            (0.40, 0.60, "40% RF / 60% XGB"),
            (0.25, 0.75, "25% RF / 75% XGB"),
            (0.0, 1.0, "100% XGB"),
        ]

        rf_residuals = []
        xgb_residuals = []

        round_predictions = {name: [] for _, _, name in tested_weights}
        round_actuals = []

        for r in val_rounds:
            train = training_df[training_df["Round"] < r].copy()
            test = training_df[training_df["Round"] == r].copy()

            if train.empty or test.empty:
                continue

            for col in self.feature_cols:
                if col not in train.columns:
                    train[col] = np.nan
                if col not in test.columns:
                    test[col] = np.nan

            imputer = SimpleImputer(strategy="median")
            X_train = imputer.fit_transform(train[self.feature_cols])
            X_test = imputer.transform(test[self.feature_cols])

            y_train = train["ActualFinish"].values
            y_test = test["ActualFinish"].values

            # Random Forest
            rf_cfg = self.model_cfg.get("random_forest", {})
            rf = RandomForestRegressor(
                n_estimators=rf_cfg.get("n_estimators", 500),
                max_depth=rf_cfg.get("max_depth", 8),
                min_samples_leaf=rf_cfg.get("min_samples_leaf", 2),
                random_state=self.random_state,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)

            # XGBoost
            xgb_cfg = self.model_cfg.get("xgboost", {})
            model_xgb = xgb.XGBRegressor(
                n_estimators=xgb_cfg.get("n_estimators", 300),
                max_depth=xgb_cfg.get("max_depth", 4),
                learning_rate=xgb_cfg.get("learning_rate", 0.03),
                subsample=xgb_cfg.get("subsample", 0.8),
                colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
                random_state=self.random_state,
                objective="reg:squarederror",
            )
            model_xgb.fit(X_train, y_train)
            xgb_pred = model_xgb.predict(X_test)

            rf_residuals.extend(y_test - rf_pred)
            xgb_residuals.extend(y_test - xgb_pred)
            round_actuals.extend(y_test)

            for w_rf, w_xgb, name in tested_weights:
                blend_pred = w_rf * rf_pred + w_xgb * xgb_pred
                round_predictions[name].extend(blend_pred)

        # Compute summary metrics across all validation rounds
        actuals = np.array(round_actuals)
        ensemble_results = []

        for w_rf, w_xgb, name in tested_weights:
            preds = np.array(round_predictions[name])
            mae = float(mean_absolute_error(actuals, preds))
            medae = float(median_absolute_error(actuals, preds))
            rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
            spearman_corr = float(spearmanr(actuals, preds).statistic)

            ensemble_results.append({
                "model_name": name,
                "rf_weight": w_rf,
                "xgb_weight": w_xgb,
                "mae": round(mae, 4),
                "medae": round(medae, 4),
                "rmse": round(rmse, 4),
                "spearman": round(spearman_corr, 4),
            })

        # Rank configurations strictly by walk-forward MAE (Option 1: dynamic auto-selection)
        sorted_ensembles = sorted(ensemble_results, key=lambda x: x["mae"])
        best_cfg = sorted_ensembles[0]
        best_w_rf = best_cfg["rf_weight"]
        best_w_xgb = best_cfg["xgb_weight"]
        best_name = best_cfg["model_name"]
        best_mae = best_cfg["mae"]

        selected_residuals = np.array(rf_residuals if best_w_rf >= 0.5 else xgb_residuals)
        empirical_dnf_rate = float(training_df["driver_dnf_rate"].mean()) if "driver_dnf_rate" in training_df.columns else 0.158
        logger.info(f"Dynamic walk-forward selection winner: {best_name} (MAE={best_mae:.4f})")

        return {
            "validation_rounds": val_rounds,
            "sample_count": len(actuals),
            "ensemble_comparisons": ensemble_results,
            "best_w_rf": best_w_rf,
            "best_w_xgb": best_w_xgb,
            "best_model_name": best_name,
            "best_mae": best_mae,
            "selected_residuals": selected_residuals,
            "rf_residuals": np.array(rf_residuals),
            "xgb_residuals": np.array(xgb_residuals),
            "rf_mae": ensemble_results[0]["mae"],
            "xgb_mae": ensemble_results[-1]["mae"],
            "empirical_dnf_rate": round(empirical_dnf_rate, 4),
        }

