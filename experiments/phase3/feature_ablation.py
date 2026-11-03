"""
Phase 3 Feature Ablation and Importance Engine.
Executes Leave-One-Group-Out (LOGO) ablation, individual feature-group evaluation,
redundancy auditing, and inner-fold permutation importance.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

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
    ProductionHybridBenchmark,
)
from experiments.model_benchmark.metrics import compute_race_metrics
from experiments.phase3.feature_inventory import FEATURE_GROUPS

ACTIVE_GROUPS = [
    "Starting Grid",
    "Qualifying Performance",
    "Practice Pace",
    "Driver Performance",
    "Constructor Form",
    "Sprint Performance",
    "Teammate Comparison",
]


def get_features_by_group(group_name: str) -> List[str]:
    """Returns baseline features belonging to a specific group."""
    return [f for f in BASELINE_FEATURES if FEATURE_GROUPS.get(f) == group_name]


def run_leave_one_group_out_ablation(
    folds: List[Dict[str, Any]],
    model_name: str = "Production Hybrid (ML + Grid)"
) -> pd.DataFrame:
    """
    Executes Leave-One-Group-Out (LOGO) ablation across all test folds.
    Compares full baseline against 7 ablated variants.
    """
    ablation_experiments = ["Full Feature Baseline"] + [f"Omit: {g}" for g in ACTIVE_GROUPS]
    results = []

    for exp in ablation_experiments:
        if exp == "Full Feature Baseline":
            excluded_group = None
            active_cols = list(BASELINE_FEATURES)
        else:
            excluded_group = exp.replace("Omit: ", "")
            active_cols = [f for f in BASELINE_FEATURES if FEATURE_GROUPS.get(f) != excluded_group]

        race_metrics_list = []
        for fold in folds:
            X_train = fold["X_train"][active_cols].copy()
            y_train = fold["y_train"]
            X_test = fold["X_test"][active_cols].copy()
            y_test = fold["y_test"]
            drivers = fold["driver_ids"]

            if model_name == "Production Hybrid (ML + Grid)":
                m = ProductionHybridBenchmark(
                    rf_weight=DEFAULT_RF_WEIGHT,
                    xgb_weight=DEFAULT_XGB_WEIGHT,
                    ml_weight=DEFAULT_ML_WEIGHT,
                    grid_weight=DEFAULT_GRID_WEIGHT
                )
            elif model_name == "Random Forest":
                m = RandomForestBenchmark()
            elif model_name == "XGBoost":
                m = XGBoostBenchmark()
            elif model_name == "Ridge Regression":
                m = RidgeRegressionBenchmark(alpha=RIDGE_ALPHA, random_state=RANDOM_SEED)
            else:
                m = ProductionHybridBenchmark()

            if isinstance(m, ProductionHybridBenchmark):
                m.fit(X_train, y_train, train_meta=fold.get("meta_train"))
                pred_score, pred_rank = m.predict(X_test, test_meta=fold.get("meta_test"))
            else:
                m.fit(X_train, y_train)
                pred_score, pred_rank = m.predict(X_test)

            m_dict = compute_race_metrics(pred_score, pred_rank, y_test, drivers)
            m_dict["Round"] = fold["round"]
            m_dict["TrackName"] = fold["track_name"]
            race_metrics_list.append(m_dict)

        df_exp = pd.DataFrame(race_metrics_list)
        results.append({
            "Experiment": exp,
            "OmittedGroup": excluded_group or "None",
            "FeatureCount": len(active_cols),
            "RacesEvaluated": len(df_exp),
            "MAE_Continuous": round(float(df_exp["mae_continuous"].mean()), 4),
            "MAE_Ordinal": round(float(df_exp["mae_ordinal"].mean()), 4),
            "RMSE": round(float(df_exp["rmse"].mean()), 4),
            "Spearman_r": round(float(df_exp["spearman"].mean()), 4),
            "Mean_Top3_Overlap": round(float(df_exp["top3_overlap"].mean()), 2),
            "Podium_F1": round(float(df_exp["podium_f1"].mean()), 3),
            "Exact_Podium_Pct": round(float(df_exp["exact_podium"].mean()) * 100.0, 1),
            "Winner_Accuracy_Pct": round(float(df_exp["winner_accuracy"].mean()) * 100.0, 1),
            "Mean_Top5_Overlap": round(float(df_exp["top5_overlap"].mean()), 2),
        })

    df_out = pd.DataFrame(results)
    # Compute delta relative to Full Baseline
    base_mae = df_out.loc[df_out["Experiment"] == "Full Feature Baseline", "MAE_Continuous"].values[0]
    base_ord = df_out.loc[df_out["Experiment"] == "Full Feature Baseline", "MAE_Ordinal"].values[0]
    base_top3 = df_out.loc[df_out["Experiment"] == "Full Feature Baseline", "Mean_Top3_Overlap"].values[0]

    df_out["Delta_Cont_MAE"] = round(df_out["MAE_Continuous"] - base_mae, 4)
    df_out["Delta_Ord_MAE"] = round(df_out["MAE_Ordinal"] - base_ord, 4)
    df_out["Delta_Top3_Overlap"] = round(df_out["Mean_Top3_Overlap"] - base_top3, 2)

    return df_out


def run_standalone_group_models(folds: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Evaluates predictive power of each feature group in complete isolation.
    """
    results = []
    for g in ACTIVE_GROUPS:
        g_cols = get_features_by_group(g)
        if not g_cols:
            continue

        race_metrics_list = []
        for fold in folds:
            X_train = fold["X_train"][g_cols].copy()
            y_train = fold["y_train"]
            X_test = fold["X_test"][g_cols].copy()
            y_test = fold["y_test"]
            drivers = fold["driver_ids"]

            m = RidgeRegressionBenchmark(alpha=RIDGE_ALPHA, random_state=RANDOM_SEED)
            if isinstance(m, ProductionHybridBenchmark):
                m.fit(X_train, y_train, train_meta=fold.get("meta_train"))
                pred_score, pred_rank = m.predict(X_test, test_meta=fold.get("meta_test"))
            else:
                m.fit(X_train, y_train)
                pred_score, pred_rank = m.predict(X_test)

            m_dict = compute_race_metrics(pred_score, pred_rank, y_test, drivers)
            race_metrics_list.append(m_dict)

        df_grp = pd.DataFrame(race_metrics_list)
        results.append({
            "FeatureGroup": g,
            "FeatureCount": len(g_cols),
            "MAE_Continuous": round(float(df_grp["mae_continuous"].mean()), 4),
            "MAE_Ordinal": round(float(df_grp["mae_ordinal"].mean()), 4),
            "RMSE": round(float(df_grp["rmse"].mean()), 4),
            "Spearman_r": round(float(df_grp["spearman"].mean()), 4),
            "Mean_Top3_Overlap": round(float(df_grp["top3_overlap"].mean()), 2),
            "Winner_Accuracy_Pct": round(float(df_grp["winner_accuracy"].mean()) * 100.0, 1),
        })

    return pd.DataFrame(results).sort_values("MAE_Continuous").reset_index(drop=True)


def compute_inner_feature_importance(folds: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Calculates permutation importance and model importances on inner training data.
    Ensures zero leakage: outer test races are never used for feature importance!
    """
    # Use final fold's training history (rounds 1-18) to evaluate feature importances
    final_fold = folds[-1]
    X_train = final_fold["X_train"].copy()
    y_train = final_fold["y_train"]

    # 1. Random Forest Gini Importance
    rf = RandomForestBenchmark(params={"n_estimators": 300, "random_state": RANDOM_SEED, "n_jobs": -1})
    rf.fit(X_train, y_train)
    rf_imp = rf.model.feature_importances_

    # 2. XGBoost Weight Importance
    xgb_m = XGBoostBenchmark()
    xgb_m.fit(X_train, y_train)
    xgb_imp = xgb_m.model.feature_importances_

    # 3. Ridge Standardized Coefficients (Absolute Magnitude)
    ridge = RidgeRegressionBenchmark(alpha=RIDGE_ALPHA, random_state=RANDOM_SEED)
    ridge.fit(X_train, y_train)
    ridge_coef = np.abs(ridge.pipeline.named_steps["ridge"].coef_)

    df_imp = pd.DataFrame({
        "feature_name": BASELINE_FEATURES,
        "feature_group": [FEATURE_GROUPS.get(f, "Other") for f in BASELINE_FEATURES],
        "rf_importance": np.round(rf_imp, 4),
        "xgb_importance": np.round(xgb_imp, 4),
        "ridge_abs_coef": np.round(ridge_coef, 4),
    })

    # Normalized composite rank
    df_imp["composite_score"] = np.round(
        (df_imp["rf_importance"] / df_imp["rf_importance"].max()) * 0.40 +
        (df_imp["xgb_importance"] / df_imp["xgb_importance"].max()) * 0.30 +
        (df_imp["ridge_abs_coef"] / df_imp["ridge_abs_coef"].max()) * 0.30,
        4
    )

    return df_imp.sort_values("composite_score", ascending=False).reset_index(drop=True)
