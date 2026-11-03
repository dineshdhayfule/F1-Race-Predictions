"""
Phase 3 Podium-Focused Investigation and Analysis Engine.
Analyzes why Ridge Regression achieves superior podium capture (1.81 / 3.0),
evaluates driver ordering deltas, and tests Ridge + RF and Ridge + Grid combinations.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from experiments.model_benchmark.config import (
    RIDGE_ALPHA,
    RANDOM_SEED,
)
from experiments.model_benchmark.models import (
    StartingGridBaseline,
    RandomForestBenchmark,
    XGBoostBenchmark,
    RidgeRegressionBenchmark,
    get_grid_positions,
)
from experiments.model_benchmark.metrics import compute_race_metrics


def run_podium_investigation(folds: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compares podium capture performance race-by-race and tests Ridge-augmented configurations.
    """
    race_comparison_records = []
    combo_records = []

    for fold in folds:
        rnd = fold["round"]
        t_name = fold["track_name"]
        X_train = fold["X_train"]
        y_train = fold["y_train"]
        X_test = fold["X_test"]
        y_test = fold["y_test"]
        drivers = fold["driver_ids"]

        grid_pos = get_grid_positions(X_test)
        rf = RandomForestBenchmark().fit(X_train, y_train)
        xgb_m = XGBoostBenchmark().fit(X_train, y_train)
        ridge = RidgeRegressionBenchmark(alpha=RIDGE_ALPHA).fit(X_train, y_train)

        rf_p, _ = rf.predict(X_test)
        xgb_p, _ = xgb_m.predict(X_test)
        ridge_p, _ = ridge.predict(X_test)

        # Candidate configurations for podium prediction
        candidates = {
            "Grid Baseline": grid_pos,
            "Pure Random Forest": rf_p,
            "Pure XGBoost": xgb_p,
            "Pure Ridge Regression": ridge_p,
            "Ridge + RF (50/50)": 0.50 * ridge_p + 0.50 * rf_p,
            "Ridge + Grid (65/35)": 0.65 * ridge_p + 0.35 * grid_pos,
            "Ridge + RF + Grid (35/35/30)": 0.35 * ridge_p + 0.35 * rf_p + 0.30 * grid_pos,
            "Production Hybrid (ML 60/40 + Grid 35)": 0.65 * (0.60 * rf_p + 0.40 * xgb_p) + 0.35 * grid_pos,
        }

        # Actual podium drivers
        order_act = np.argsort(y_test)
        actual_podium = [drivers[i] for i in order_act[:3]]

        race_eval = {
            "Round": rnd,
            "TrackName": t_name,
            "Actual_P1": actual_podium[0],
            "Actual_P2": actual_podium[1],
            "Actual_P3": actual_podium[2],
        }

        for c_name, p_scores in candidates.items():
            order_pred = np.argsort(p_scores)
            pred_podium = [drivers[i] for i in order_pred[:3]]
            overlap = len(set(pred_podium) & set(actual_podium))
            is_exact = 1.0 if pred_podium == actual_podium else 0.0

            race_eval[f"{c_name}_Top3"] = ", ".join(pred_podium)
            race_eval[f"{c_name}_Overlap"] = overlap

            p_ranks = np.argsort(order_pred) + 1
            m = compute_race_metrics(p_scores, p_ranks, y_test, drivers)
            combo_records.append({
                "Configuration": c_name,
                "Round": rnd,
                "TrackName": t_name,
                "MAE_Continuous": m["mae_continuous"],
                "MAE_Ordinal": m["mae_ordinal"],
                "Top3_Overlap": m["top3_overlap"],
                "Exact_Podium": m["exact_podium"],
                "Winner_Accuracy": m["winner_accuracy"],
            })

        race_comparison_records.append(race_eval)

    df_race_comp = pd.DataFrame(race_comparison_records)
    df_combo = pd.DataFrame(combo_records)

    # Aggregate by configuration
    agg_rows = []
    for c_name, grp in df_combo.groupby("Configuration"):
        agg_rows.append({
            "PodiumConfiguration": c_name,
            "RacesEvaluated": len(grp),
            "Mean_Top3_Overlap": round(float(grp["Top3_Overlap"].mean()), 2),
            "Podium_F1": round(float(grp["Top3_Overlap"].mean() / 3.0), 3),
            "Exact_Podium_Pct": round(float(grp["Exact_Podium"].mean()) * 100.0, 1),
            "Winner_Accuracy_Pct": round(float(grp["Winner_Accuracy"].mean()) * 100.0, 1),
            "MAE_Continuous": round(float(grp["MAE_Continuous"].mean()), 4),
            "MAE_Ordinal": round(float(grp["MAE_Ordinal"].mean()), 4),
        })

    df_summary = pd.DataFrame(agg_rows).sort_values("Mean_Top3_Overlap", ascending=False).reset_index(drop=True)
    return df_summary, df_race_comp
