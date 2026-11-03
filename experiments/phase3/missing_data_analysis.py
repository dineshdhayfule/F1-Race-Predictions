"""
Phase 3 Missing Data and Common Feature Subset Evaluation Engine.
Investigates telemetry availability discrepancies between seasons,
and runs a controlled common-subset experiment across Track A and Track B.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from experiments.model_benchmark.config import (
    BASELINE_FEATURES,
    RIDGE_ALPHA,
    RANDOM_SEED,
)
from experiments.model_benchmark.models import (
    StartingGridBaseline,
    RandomForestBenchmark,
    XGBoostBenchmark,
    RidgeRegressionBenchmark,
    ProductionHybridBenchmark,
    get_grid_positions,
)
from experiments.model_benchmark.metrics import compute_race_metrics

# Common feature subset present with real observed data in BOTH historical and 2026 datasets
COMMON_FEATURE_SUBSET = [
    "GridPosition",
    "grid_norm",
    "expected_grid_finish",
    "grid_retention_prob",
    "QualifyingPosition",
    "driver_finish_ewma",
    "driver_quali_ewma",
    "driver_dnf_rate",
    "recent_finish",
    "recent_grid",
    "recent_points",
    "avg_finish",
    "avg_grid",
    "total_points",
    "races_completed",
    "team_finish_ewma",
    "team_points_rate",
    "team_avg_finish",
    "team_points",
    "driver_teammate_h2h_ratio",
]


def run_common_subset_evaluation(
    folds_2026: List[Dict[str, Any]],
    folds_hist: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Evaluates models on the Common Feature Subset (excluding practice-session telemetry)
    to establish an apples-to-apples cross-season comparison.
    """
    records = []

    for track_name, fold_list in [("Track B (2026 Common Subset)", folds_2026), ("Track A (Historical Common Subset)", folds_hist)]:
        for fold in fold_list:
            rnd = fold["round"]
            t_name = fold["track_name"]
            X_train = fold["X_train"][COMMON_FEATURE_SUBSET].copy()
            y_train = fold["y_train"]
            X_test = fold["X_test"][COMMON_FEATURE_SUBSET].copy()
            y_test = fold["y_test"]
            drivers = fold["driver_ids"]

            grid_pos = get_grid_positions(X_test)
            rf = RandomForestBenchmark().fit(X_train, y_train)
            xgb_m = XGBoostBenchmark().fit(X_train, y_train)
            ridge = RidgeRegressionBenchmark(alpha=RIDGE_ALPHA).fit(X_train, y_train)

            rf_p, _ = rf.predict(X_test)
            xgb_p, _ = xgb_m.predict(X_test)
            ridge_p, _ = ridge.predict(X_test)

            ml_blend = 0.60 * rf_p + 0.40 * xgb_p
            hybrid_p = 0.65 * ml_blend + 0.35 * grid_pos

            cand_preds = {
                "Grid Baseline": grid_pos,
                "Random Forest (Common)": rf_p,
                "XGBoost (Common)": xgb_p,
                "Ridge Regression (Common)": ridge_p,
                "Production Hybrid (Common)": hybrid_p,
            }

            for c_name, p_scores in cand_preds.items():
                p_ranks = np.argsort(np.argsort(p_scores)) + 1
                m = compute_race_metrics(p_scores, p_ranks, y_test, drivers)
                m.update({
                    "Track": track_name,
                    "Model": c_name,
                    "Round": rnd,
                    "TrackName": t_name
                })
                records.append(m)

    df_raw = pd.DataFrame(records)

    agg_rows = []
    for (t_label, m_name), grp in df_raw.groupby(["Track", "Model"]):
        agg_rows.append({
            "Track": t_label,
            "Model": m_name,
            "FeaturesUsed": len(COMMON_FEATURE_SUBSET),
            "RacesEvaluated": len(grp),
            "MAE_Continuous": round(float(grp["mae_continuous"].mean()), 4),
            "MAE_Ordinal": round(float(grp["mae_ordinal"].mean()), 4),
            "RMSE": round(float(grp["rmse"].mean()), 4),
            "Spearman_r": round(float(grp["spearman"].mean()), 4),
            "Mean_Top3_Overlap": round(float(grp["top3_overlap"].mean()), 2),
            "Winner_Accuracy_Pct": round(float(grp["winner_accuracy"].mean()) * 100.0, 1),
        })

    return pd.DataFrame(agg_rows).sort_values(["Track", "MAE_Continuous"]).reset_index(drop=True)
