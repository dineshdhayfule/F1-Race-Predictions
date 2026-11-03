"""
Phase 3 Circuit-Type Performance and Blending Sensitivity Analysis.
Evaluates model behavior across street, permanent, and temporary circuits,
and tests circuit-specific vs global blending weights.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd

from experiments.model_benchmark.models import (
    StartingGridBaseline,
    RandomForestBenchmark,
    XGBoostBenchmark,
    RidgeRegressionBenchmark,
    ProductionHybridBenchmark,
    get_grid_positions,
)
from experiments.model_benchmark.metrics import compute_race_metrics


def run_circuit_type_evaluation(folds: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Evaluates baseline models and blending sensitivities segmented by circuit type.
    """
    records = []
    # Grid weights to test per circuit type
    test_grid_weights = [0.0, 0.20, 0.35, 0.50]

    for fold in folds:
        c_type = fold.get("circuit_type", "permanent").capitalize()
        t_name = fold["track_name"]
        rnd = fold["round"]

        X_train = fold["X_train"]
        y_train = fold["y_train"]
        X_test = fold["X_test"]
        y_test = fold["y_test"]
        drivers = fold["driver_ids"]

        grid_pos = get_grid_positions(X_test)
        rf = RandomForestBenchmark().fit(X_train, y_train)
        xgb_m = XGBoostBenchmark().fit(X_train, y_train)
        ridge = RidgeRegressionBenchmark().fit(X_train, y_train)

        rf_pred, _ = rf.predict(X_test)
        xgb_pred, _ = xgb_m.predict(X_test)
        ridge_pred, _ = ridge.predict(X_test)

        ml_pred = 0.60 * rf_pred + 0.40 * xgb_pred

        # 1. Starting Grid alone
        m_grid = compute_race_metrics(grid_pos, np.argsort(np.argsort(grid_pos)) + 1, y_test, drivers)
        m_grid.update({"CircuitType": c_type, "Configuration": "Pure Grid Baseline", "GridWeight": 1.0, "Round": rnd})
        records.append(m_grid)

        # 2. Pure ML alone
        m_ml = compute_race_metrics(ml_pred, np.argsort(np.argsort(ml_pred)) + 1, y_test, drivers)
        m_ml.update({"CircuitType": c_type, "Configuration": "Pure ML Ensemble (60/40)", "GridWeight": 0.0, "Round": rnd})
        records.append(m_ml)

        # 3. Pure Ridge alone
        m_ridge = compute_race_metrics(ridge_pred, np.argsort(np.argsort(ridge_pred)) + 1, y_test, drivers)
        m_ridge.update({"CircuitType": c_type, "Configuration": "Pure Ridge Regression", "GridWeight": 0.0, "Round": rnd})
        records.append(m_ridge)

        # 4. Grid Blends across weights
        for gw in test_grid_weights:
            blend = (1.0 - gw) * ml_pred + gw * grid_pos
            m_blend = compute_race_metrics(blend, np.argsort(np.argsort(blend)) + 1, y_test, drivers)
            m_blend.update({
                "CircuitType": c_type,
                "Configuration": f"ML + {int(gw*100)}% Grid",
                "GridWeight": gw,
                "Round": rnd
            })
            records.append(m_blend)

    df_raw = pd.DataFrame(records)

    # Aggregate by CircuitType and Configuration
    agg_rows = []
    for (c_type, cfg), grp in df_raw.groupby(["CircuitType", "Configuration"]):
        agg_rows.append({
            "CircuitType": c_type,
            "Configuration": cfg,
            "GridWeight": float(grp["GridWeight"].iloc[0]),
            "RacesEvaluated": len(grp),
            "MAE_Continuous": round(float(grp["mae_continuous"].mean()), 4),
            "MAE_Ordinal": round(float(grp["mae_ordinal"].mean()), 4),
            "RMSE": round(float(grp["rmse"].mean()), 4),
            "Spearman_r": round(float(grp["spearman"].mean()), 4),
            "Mean_Top3_Overlap": round(float(grp["top3_overlap"].mean()), 2),
            "Winner_Accuracy_Pct": round(float(grp["winner_accuracy"].mean()) * 100.0, 1),
            "Points_Accuracy_Pct": round(float(grp["points_accuracy"].mean()) * 100.0, 1),
        })

    df_summary = pd.DataFrame(agg_rows).sort_values(["CircuitType", "MAE_Continuous"]).reset_index(drop=True)
    return df_summary
