"""
Master Benchmark CLI Runner for F1 Model Comparison.
Coordinates chronological walk-forward validation across Track A (Historical) and Track B (2026),
evaluating all six prediction approaches under identical conditions.
"""

from datetime import datetime
import json
import time
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from experiments.model_benchmark.config import (
    RANDOM_SEED,
    BASELINE_FEATURES,
    BENCHMARK_STAGE,
    DEFAULT_RF_WEIGHT,
    DEFAULT_XGB_WEIGHT,
    DEFAULT_ML_WEIGHT,
    DEFAULT_GRID_WEIGHT,
    RIDGE_ALPHA,
)
from experiments.model_benchmark.models import (
    StartingGridBaseline,
    RandomForestBenchmark,
    XGBoostBenchmark,
    RidgeRegressionBenchmark,
    MLOnlyEnsembleBenchmark,
    ProductionHybridBenchmark,
)
from experiments.model_benchmark.metrics import (
    compute_race_metrics,
    compute_probability_metrics,
    compute_paired_comparison,
)
from experiments.model_benchmark.walk_forward import WalkForwardEngine
from experiments.model_benchmark.reporting import BenchmarkReporter
from src.utils.logging_utils import get_logger

logger = get_logger("benchmark_runner")


def run_benchmark(run_track_a: bool = True, run_track_b: bool = True) -> Dict[str, Any]:
    """
    Executes end-to-end benchmarking across all available test folds.
    """
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("STARTING APEXGRID MODEL BENCHMARK EXPERIMENT (PHASE 2)")
    logger.info(f"Stage: {BENCHMARK_STAGE} | Random Seed: {RANDOM_SEED}")
    logger.info("=" * 80)

    engine = WalkForwardEngine()
    reporter = BenchmarkReporter()

    all_per_race_metrics = []
    all_predictions = []
    prob_eval_rows = []

    # -------------------------------------------------------------------------
    # TRACK B: 2026 Championship Season
    # -------------------------------------------------------------------------
    folds_2026 = engine.build_2026_folds() if run_track_b else []
    logger.info(f"Loaded {len(folds_2026)} test folds for 2026 Season.")

    for fold in folds_2026:
        rnd = fold["round"]
        t_name = fold["track_name"]
        logger.info(f"[Track B - 2026] Evaluating Round {rnd}: {t_name} (N={fold['test_sample_count']} drivers)")

        X_train = fold["X_train"]
        y_train = fold["y_train"]
        X_test = fold["X_test"]
        y_test = fold["y_test"]
        drivers = fold["driver_ids"]
        meta_test = fold["meta_test"]

        # Instantiate fresh models for each fold
        models = [
            StartingGridBaseline(),
            RandomForestBenchmark(),
            XGBoostBenchmark(),
            RidgeRegressionBenchmark(alpha=RIDGE_ALPHA, random_state=RANDOM_SEED),
            MLOnlyEnsembleBenchmark(rf_weight=DEFAULT_RF_WEIGHT, xgb_weight=DEFAULT_XGB_WEIGHT),
            ProductionHybridBenchmark(
                rf_weight=DEFAULT_RF_WEIGHT,
                xgb_weight=DEFAULT_XGB_WEIGHT,
                ml_weight=DEFAULT_ML_WEIGHT,
                grid_weight=DEFAULT_GRID_WEIGHT
            ),
        ]

        # Fit and predict for each model
        for m in models:
            m.fit(X_train, y_train)
            pred_score, pred_rank = m.predict(X_test)

            metrics = compute_race_metrics(
                pred_scores=pred_score,
                pred_ranks=pred_rank,
                actual_finishes=y_test,
                driver_ids=drivers
            )

            metrics.update({
                "Track": "Track B (2026 Season)",
                "Round": rnd,
                "TrackName": t_name,
                "CircuitType": fold["circuit_type"],
                "Model": m.name,
            })
            all_per_race_metrics.append(metrics)

            # Store driver predictions
            for d_idx, drv in enumerate(drivers):
                all_predictions.append({
                    "Track": "Track B (2026 Season)",
                    "Round": rnd,
                    "TrackName": t_name,
                    "Driver": drv,
                    "ActualFinish": y_test[d_idx],
                    "Model": m.name,
                    "PredictedScore": round(float(pred_score[d_idx]), 3),
                    "PredictedRank": int(pred_rank[d_idx])
                })

            # Monte Carlo probability metrics for Model 6
            if isinstance(m, ProductionHybridBenchmark):
                sim_df = m.simulate_probabilities(X_test, pred_score, test_meta=meta_test)
                p_metrics = compute_probability_metrics(sim_df, y_test, drivers)
                p_metrics.update({
                    "Track": "Track B (2026 Season)",
                    "Round": rnd,
                    "TrackName": t_name,
                    "Model": m.name
                })
                prob_eval_rows.append(p_metrics)

    # -------------------------------------------------------------------------
    # TRACK A: Historical Circuit Races (Baku & Sao Paulo)
    # -------------------------------------------------------------------------
    folds_hist = engine.build_historical_folds() if run_track_a else []
    logger.info(f"Loaded {len(folds_hist)} test folds for Historical Tracks.")

    for fold in folds_hist:
        rnd = fold["round"]
        t_name = fold["track_name"]
        logger.info(f"[Track A - Historical] Evaluating Event {rnd}: {t_name} (N={fold['test_sample_count']} drivers)")

        X_train = fold["X_train"]
        y_train = fold["y_train"]
        X_test = fold["X_test"]
        y_test = fold["y_test"]
        drivers = fold["driver_ids"]
        meta_test = fold["meta_test"]

        models = [
            StartingGridBaseline(),
            RandomForestBenchmark(),
            XGBoostBenchmark(),
            RidgeRegressionBenchmark(alpha=RIDGE_ALPHA, random_state=RANDOM_SEED),
            MLOnlyEnsembleBenchmark(rf_weight=DEFAULT_RF_WEIGHT, xgb_weight=DEFAULT_XGB_WEIGHT),
            ProductionHybridBenchmark(
                rf_weight=DEFAULT_RF_WEIGHT,
                xgb_weight=DEFAULT_XGB_WEIGHT,
                ml_weight=DEFAULT_ML_WEIGHT,
                grid_weight=DEFAULT_GRID_WEIGHT
            ),
        ]

        for m in models:
            m.fit(X_train, y_train)
            pred_score, pred_rank = m.predict(X_test)

            metrics = compute_race_metrics(
                pred_scores=pred_score,
                pred_ranks=pred_rank,
                actual_finishes=y_test,
                driver_ids=drivers
            )

            metrics.update({
                "Track": "Track A (Historical Tracks)",
                "Round": rnd,
                "TrackName": t_name,
                "CircuitType": fold["circuit_type"],
                "Model": m.name,
            })
            all_per_race_metrics.append(metrics)

            for d_idx, drv in enumerate(drivers):
                all_predictions.append({
                    "Track": "Track A (Historical Tracks)",
                    "Round": rnd,
                    "TrackName": t_name,
                    "Driver": drv,
                    "ActualFinish": y_test[d_idx],
                    "Model": m.name,
                    "PredictedScore": round(float(pred_score[d_idx]), 3),
                    "PredictedRank": int(pred_rank[d_idx])
                })

    # Compile dataframes
    df_metrics = pd.DataFrame(all_per_race_metrics)
    df_preds = pd.DataFrame(all_predictions)
    df_prob = pd.DataFrame(prob_eval_rows) if prob_eval_rows else pd.DataFrame()

    df_2026 = df_metrics[df_metrics["Track"] == "Track B (2026 Season)"].copy()
    df_hist = df_metrics[df_metrics["Track"] == "Track A (Historical Tracks)"].copy()

    # Model comparison aggregate table
    comp_rows = []
    for track_label, sub_df in [("Track B (2026 Season)", df_2026), ("Track A (Historical Tracks)", df_hist)]:
        if sub_df.empty:
            continue
        for m_name, grp in sub_df.groupby("Model"):
            comp_rows.append({
                "Track": track_label,
                "Model": m_name,
                "RacesEvaluated": len(grp),
                "MAE_Continuous": round(float(grp["mae_continuous"].mean()), 4),
                "MAE_Ordinal": round(float(grp["mae_ordinal"].mean()), 4),
                "RMSE": round(float(grp["rmse"].mean()), 4),
                "Spearman_r": round(float(grp["spearman"].mean()), 4),
                "Exact_Podium_Pct": round(float(grp["exact_podium"].mean()) * 100.0, 1),
                "Mean_Top3_Overlap": round(float(grp["top3_overlap"].mean()), 2),
                "Podium_F1": round(float(grp["podium_f1"].mean()), 3),
                "Winner_Accuracy_Pct": round(float(grp["winner_accuracy"].mean()) * 100.0, 1),
                "Mean_Top5_Overlap": round(float(grp["top5_overlap"].mean()), 2),
                "Points_Accuracy_Pct": round(float(grp["points_accuracy"].mean()) * 100.0, 1)
            })

    df_comp = pd.DataFrame(comp_rows)
    # Order models logically
    model_order = [
        "Starting Grid Baseline",
        "Random Forest",
        "XGBoost",
        "Ridge Regression",
        "RF + XGBoost Ensemble (ML-only)",
        "Production Hybrid (ML + Grid)"
    ]
    if not df_comp.empty:
        df_comp["ModelOrder"] = df_comp["Model"].map(lambda x: model_order.index(x) if x in model_order else 99)
        df_comp = df_comp.sort_values(["Track", "ModelOrder"]).drop(columns=["ModelOrder"]).reset_index(drop=True)

    # Circuit analysis breakdown
    circ_rows = []
    if not df_2026.empty:
        for c_type, c_grp in df_2026.groupby("CircuitType"):
            for m_name, m_grp in c_grp.groupby("Model"):
                circ_rows.append({
                    "CircuitType": c_type.capitalize(),
                    "Model": m_name,
                    "RacesCount": len(m_grp),
                    "MAE_Continuous": round(float(m_grp["mae_continuous"].mean()), 4),
                    "MAE_Ordinal": round(float(m_grp["mae_ordinal"].mean()), 4),
                    "Spearman_r": round(float(m_grp["spearman"].mean()), 4),
                    "Mean_Top3_Overlap": round(float(m_grp["top3_overlap"].mean()), 2),
                    "Exact_Podium_Pct": round(float(m_grp["exact_podium"].mean()) * 100.0, 1),
                })
    df_circ = pd.DataFrame(circ_rows)
    if not df_circ.empty:
        df_circ["ModelOrder"] = df_circ["Model"].map(lambda x: model_order.index(x) if x in model_order else 99)
        df_circ = df_circ.sort_values(["CircuitType", "ModelOrder"]).drop(columns=["ModelOrder"]).reset_index(drop=True)

    # Paired statistical comparisons (for 2026 Season)
    paired_stats = {}
    if not df_2026.empty:
        grid_maes = df_2026[df_2026["Model"] == "Starting Grid Baseline"].sort_values("Round")["mae_ordinal"].tolist()
        hybrid_maes = df_2026[df_2026["Model"] == "Production Hybrid (ML + Grid)"].sort_values("Round")["mae_ordinal"].tolist()

        for m_name in model_order:
            m_maes = df_2026[df_2026["Model"] == m_name].sort_values("Round")["mae_ordinal"].tolist()
            if len(m_maes) == len(grid_maes):
                paired_stats[f"{m_name}_vs_Grid"] = compute_paired_comparison(m_maes, grid_maes)
                paired_stats[f"{m_name}_vs_Hybrid"] = compute_paired_comparison(m_maes, hybrid_maes)

    # Metadata
    elapsed = time.time() - start_time
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "benchmark_stage": BENCHMARK_STAGE,
        "random_seed": RANDOM_SEED,
        "feature_count": len(BASELINE_FEATURES),
        "features": BASELINE_FEATURES,
        "ridge_alpha": RIDGE_ALPHA,
        "rf_weight": DEFAULT_RF_WEIGHT,
        "xgb_weight": DEFAULT_XGB_WEIGHT,
        "ml_weight": DEFAULT_ML_WEIGHT,
        "grid_weight": DEFAULT_GRID_WEIGHT,
        "evaluated_2026_rounds": [f["round"] for f in folds_2026],
        "evaluated_historical_events": [f["round"] for f in folds_hist],
        "total_evaluated_races": len(folds_2026) + len(folds_hist),
        "execution_duration_sec": round(elapsed, 2),
        "paired_statistical_tests": paired_stats
    }

    # Export all artifacts
    paths = reporter.export_results(
        historical_metrics_df=df_hist,
        season_2026_metrics_df=df_2026,
        predictions_df=df_preds,
        model_comparison_df=df_comp,
        circuit_analysis_df=df_circ,
        probability_df=df_prob,
        metadata=metadata
    )

    logger.info("=" * 80)
    logger.info(f"BENCHMARK COMPLETED IN {elapsed:.2f} SECONDS")
    logger.info(f"Summary Report: {paths['benchmark_summary']}")
    logger.info("=" * 80)

    return {
        "paths": paths,
        "metadata": metadata,
        "comparison_table": df_comp,
        "circuit_table": df_circ,
        "paired_stats": paired_stats
    }


if __name__ == "__main__":
    run_benchmark()
