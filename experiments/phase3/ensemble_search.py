"""
Phase 3 Ensemble Search and Weight Optimization Engine.
Performs strictly chronological inner-fold validation to optimize model blending weights
and evaluate 9 model combinations without test-fold leakage.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

from experiments.model_benchmark.config import (
    BASELINE_FEATURES,
    DEFAULT_RF_WEIGHT,
    DEFAULT_XGB_WEIGHT,
    DEFAULT_ML_WEIGHT,
    DEFAULT_GRID_WEIGHT,
    RIDGE_ALPHA,
    RANDOM_SEED,
)
from experiments.model_benchmark.models import (
    RandomForestBenchmark,
    XGBoostBenchmark,
    RidgeRegressionBenchmark,
    get_grid_positions,
)
from experiments.model_benchmark.metrics import compute_race_metrics


class EnsembleSearchOptimizer:
    """
    Optimizes ensemble combinations and grid blending weights using
    strict inner chronological walk-forward cross-validation.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state

    def evaluate_all_combinations_on_outer_folds(
        self,
        folds: List[Dict[str, Any]]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
        """
        Evaluates 9 model combinations across outer test folds.
        For adaptive ensemble, weights are chosen strictly via inner walk-forward validation!
        """
        combinations = [
            "Random Forest",
            "XGBoost",
            "Ridge Regression",
            "RF + XGBoost (60/40)",
            "RF + Ridge (60/40)",
            "XGBoost + Ridge (50/50)",
            "RF + XGBoost + Ridge (40/30/30)",
            "Production Hybrid (0.65 ML + 0.35 Grid)",
            "Inner-CV Optimized Hybrid"
        ]

        fold_race_metrics = []
        optimized_weight_records = []
        all_pred_records = []

        for f_idx, fold in enumerate(folds):
            rnd = fold["round"]
            t_name = fold["track_name"]
            X_train = fold["X_train"]
            y_train = fold["y_train"]
            X_test = fold["X_test"]
            y_test = fold["y_test"]
            drivers = fold["driver_ids"]

            # Train base estimators on training fold
            rf = RandomForestBenchmark().fit(X_train, y_train)
            xgb_m = XGBoostBenchmark().fit(X_train, y_train)
            ridge = RidgeRegressionBenchmark(alpha=RIDGE_ALPHA).fit(X_train, y_train)

            rf_pred, _ = rf.predict(X_test)
            xgb_pred, _ = xgb_m.predict(X_test)
            ridge_pred, _ = ridge.predict(X_test)
            grid_pos = get_grid_positions(X_test)

            # Inner-fold optimization on training history (if enough training rounds exist)
            train_rounds = sorted(list(fold["meta_train"]["Round"].dropna().unique()))
            if len(train_rounds) >= 4:
                opt_rf, opt_xgb, opt_ridge, opt_grid = self._optimize_weights_inner_cv(
                    meta_train=fold["meta_train"]
                )
            else:
                # Fallback to configured defaults for early rounds
                opt_rf, opt_xgb, opt_ridge, opt_grid = 0.50, 0.25, 0.25, 0.35

            optimized_weight_records.append({
                "OuterRound": rnd,
                "TrackName": t_name,
                "TrainRoundsAvailable": len(train_rounds),
                "Opt_RF_Weight": round(opt_rf, 3),
                "Opt_XGB_Weight": round(opt_xgb, 3),
                "Opt_Ridge_Weight": round(opt_ridge, 3),
                "Opt_Grid_Weight": round(opt_grid, 3),
            })

            # Calculate predictions for all 9 combinations
            preds_dict = {
                "Random Forest": rf_pred,
                "XGBoost": xgb_pred,
                "Ridge Regression": ridge_pred,
                "RF + XGBoost (60/40)": 0.60 * rf_pred + 0.40 * xgb_pred,
                "RF + Ridge (60/40)": 0.60 * rf_pred + 0.40 * ridge_pred,
                "XGBoost + Ridge (50/50)": 0.50 * xgb_pred + 0.50 * ridge_pred,
                "RF + XGBoost + Ridge (40/30/30)": 0.40 * rf_pred + 0.30 * xgb_pred + 0.30 * ridge_pred,
                "Production Hybrid (0.65 ML + 0.35 Grid)": (
                    0.65 * (0.60 * rf_pred + 0.40 * xgb_pred) + 0.35 * grid_pos
                ),
                "Inner-CV Optimized Hybrid": (
                    (1.0 - opt_grid) * (opt_rf * rf_pred + opt_xgb * xgb_pred + opt_ridge * ridge_pred) +
                    opt_grid * grid_pos
                )
            }

            for combo_name, pred_score in preds_dict.items():
                pred_rank = np.argsort(np.argsort(pred_score)) + 1
                metrics = compute_race_metrics(pred_score, pred_rank, y_test, drivers)
                metrics.update({
                    "Round": rnd,
                    "TrackName": t_name,
                    "Model": combo_name,
                    "TrackType": fold.get("circuit_type", "permanent")
                })
                fold_race_metrics.append(metrics)

                for d_idx, drv in enumerate(drivers):
                    all_pred_records.append({
                        "Round": rnd,
                        "TrackName": t_name,
                        "Driver": drv,
                        "ActualFinish": y_test[d_idx],
                        "Model": combo_name,
                        "PredictedScore": round(float(pred_score[d_idx]), 3),
                        "PredictedRank": int(pred_rank[d_idx])
                    })

        df_metrics = pd.DataFrame(fold_race_metrics)
        df_opt_weights = pd.DataFrame(optimized_weight_records)

        # Aggregate comparison table across combinations
        agg_rows = []
        for combo_name in combinations:
            grp = df_metrics[df_metrics["Model"] == combo_name]
            agg_rows.append({
                "ModelCombination": combo_name,
                "RacesEvaluated": len(grp),
                "MAE_Continuous": round(float(grp["mae_continuous"].mean()), 4),
                "MAE_Ordinal": round(float(grp["mae_ordinal"].mean()), 4),
                "RMSE": round(float(grp["rmse"].mean()), 4),
                "Spearman_r": round(float(grp["spearman"].mean()), 4),
                "Mean_Top3_Overlap": round(float(grp["top3_overlap"].mean()), 2),
                "Podium_F1": round(float(grp["podium_f1"].mean()), 3),
                "Exact_Podium_Pct": round(float(grp["exact_podium"].mean()) * 100.0, 1),
                "Winner_Accuracy_Pct": round(float(grp["winner_accuracy"].mean()) * 100.0, 1),
                "Mean_Top5_Overlap": round(float(grp["top5_overlap"].mean()), 2),
                "Points_Accuracy_Pct": round(float(grp["points_accuracy"].mean()) * 100.0, 1)
            })

        df_comparison = pd.DataFrame(agg_rows)
        return df_comparison, df_opt_weights, all_pred_records

    def _optimize_weights_inner_cv(
        self,
        meta_train: pd.DataFrame
    ) -> Tuple[float, float, float, float]:
        """
        Sweeps candidate ML and grid weights using strictly prior inner rounds.
        Selects weights minimizing inner walk-forward continuous MAE.
        """
        rounds = sorted(list(meta_train["Round"].dropna().unique()))
        inner_test_rounds = rounds[3:] if len(rounds) > 3 else rounds[-1:]

        # Candidate grid blending weights
        candidate_grid_weights = [0.15, 0.25, 0.35, 0.45]
        # Candidate 3-way ML weights (RF, XGB, Ridge)
        candidate_ml_weights = [
            (0.60, 0.40, 0.00),  # Production ML
            (0.50, 0.30, 0.20),
            (0.40, 0.30, 0.30),
            (0.50, 0.00, 0.50),  # RF + Ridge
            (0.34, 0.33, 0.33),  # Equal 3-way
        ]

        inner_predictions = {}
        for inner_rnd in inner_test_rounds:
            in_tr = meta_train[meta_train["Round"] < inner_rnd]
            in_te = meta_train[meta_train["Round"] == inner_rnd]
            if in_tr.empty or in_te.empty:
                continue

            X_tr = in_tr[BASELINE_FEATURES]
            y_tr = in_tr["ActualFinish"].values
            X_te = in_te[BASELINE_FEATURES]
            y_te = in_te["ActualFinish"].values

            rf = RandomForestBenchmark(params={"n_estimators": 100, "random_state": self.random_state, "n_jobs": -1}).fit(X_tr, y_tr)
            xgb_m = XGBoostBenchmark(params={"n_estimators": 80, "max_depth": 4, "learning_rate": 0.05, "random_state": self.random_state}).fit(X_tr, y_tr)
            ridge = RidgeRegressionBenchmark(alpha=RIDGE_ALPHA).fit(X_tr, y_tr)

            p_rf, _ = rf.predict(X_te)
            p_xgb, _ = xgb_m.predict(X_te)
            p_rdg, _ = ridge.predict(X_te)
            p_grid = get_grid_positions(in_te)

            inner_predictions[inner_rnd] = {
                "p_rf": p_rf,
                "p_xgb": p_xgb,
                "p_rdg": p_rdg,
                "p_grid": p_grid,
                "y_te": y_te
            }

        best_score = float("inf")
        best_cfg = (0.50, 0.25, 0.25, 0.35)

        for w_rf, w_xgb, w_rdg in candidate_ml_weights:
            for w_grid in candidate_grid_weights:
                w_ml = 1.0 - w_grid
                fold_maes = []

                for inner_rnd, preds in inner_predictions.items():
                    ml_blend = w_rf * preds["p_rf"] + w_xgb * preds["p_xgb"] + w_rdg * preds["p_rdg"]
                    total_blend = w_ml * ml_blend + w_grid * preds["p_grid"]
                    mae = float(np.mean(np.abs(total_blend - preds["y_te"])))
                    fold_maes.append(mae)

                if fold_maes:
                    avg_mae = float(np.mean(fold_maes))
                    if avg_mae < best_score:
                        best_score = avg_mae
                        best_cfg = (w_rf, w_xgb, w_rdg, w_grid)

        return best_cfg
