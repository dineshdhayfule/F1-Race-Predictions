"""
Comprehensive Evaluation Metrics for F1 Model Benchmarking.
Evaluates continuous error, ordinal rank accuracy, podium capture, and probability calibration.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon, ttest_rel


def compute_race_metrics(
    pred_scores: np.ndarray,
    pred_ranks: np.ndarray,
    actual_finishes: np.ndarray,
    driver_ids: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Computes deterministic position, podium, top-5, and points metrics for a single race.

    Parameters
    ----------
    pred_scores : np.ndarray
        Continuous predicted score / finishing expectation.
    pred_ranks : np.ndarray
        Discrete predicted ordinal finish rank (1, 2, ..., N).
    actual_finishes : np.ndarray
        Ground-truth official race finishing classification (1, 2, ..., N).
    driver_ids : list of str, optional
        Driver abbreviations for set-based intersection.
    """
    # Filter valid actual finishes (ignore drivers who did not start / unclassified if any)
    valid_mask = ~np.isnan(actual_finishes) & ~np.isnan(pred_scores)
    if not np.any(valid_mask):
        return {}

    y_true = actual_finishes[valid_mask]
    y_score = pred_scores[valid_mask]
    y_rank = pred_ranks[valid_mask]
    n_drivers = len(y_true)

    # 1. Position Prediction Metrics
    mae_continuous = float(np.mean(np.abs(y_score - y_true)))
    mae_ordinal = float(np.mean(np.abs(y_rank - y_true)))
    rmse = float(np.sqrt(np.mean((y_score - y_true) ** 2)))

    try:
        spearman_corr = float(spearmanr(y_score, y_true).statistic)
        if np.isnan(spearman_corr):
            spearman_corr = 0.0
    except Exception:
        spearman_corr = 0.0

    # Sort drivers by actual finish and predicted rank
    order_act = np.argsort(y_true)
    order_pred = np.argsort(y_rank)

    # 2. Winner Metrics
    winner_act = order_act[0]
    winner_pred = order_pred[0]
    winner_accuracy = 1.0 if winner_act == winner_pred else 0.0

    # 3. Podium Metrics
    act_podium_indices = set(order_act[:3])
    pred_podium_indices = set(order_pred[:3])

    # Exact 1-2-3 podium order match
    exact_podium = 1.0 if (
        len(order_act) >= 3 and len(order_pred) >= 3 and
        order_act[0] == order_pred[0] and
        order_act[1] == order_pred[1] and
        order_act[2] == order_pred[2]
    ) else 0.0

    # Top-3 set overlap (K in {0, 1, 2, 3})
    top3_overlap = float(len(act_podium_indices & pred_podium_indices))
    podium_precision = top3_overlap / 3.0
    podium_recall = top3_overlap / 3.0
    podium_f1 = top3_overlap / 3.0

    # 4. Top-5 and Points Metrics
    act_top5_indices = set(order_act[:min(5, n_drivers)])
    pred_top5_indices = set(order_pred[:min(5, n_drivers)])
    top5_overlap = float(len(act_top5_indices & pred_top5_indices))

    act_points_indices = set(order_act[:min(10, n_drivers)])
    pred_points_indices = set(order_pred[:min(10, n_drivers)])
    points_accuracy = float(len(act_points_indices & pred_points_indices)) / float(min(10, n_drivers))

    return {
        "mae_continuous": round(mae_continuous, 4),
        "mae_ordinal": round(mae_ordinal, 4),
        "rmse": round(rmse, 4),
        "spearman": round(spearman_corr, 4),
        "winner_accuracy": round(winner_accuracy, 4),
        "exact_podium": round(exact_podium, 4),
        "top3_overlap": round(top3_overlap, 2),
        "podium_precision": round(podium_precision, 4),
        "podium_recall": round(podium_recall, 4),
        "podium_f1": round(podium_f1, 4),
        "top5_overlap": round(top5_overlap, 2),
        "points_accuracy": round(points_accuracy, 4),
        "driver_count": n_drivers
    }


def compute_probability_metrics(
    sim_df: pd.DataFrame,
    actual_finishes: np.ndarray,
    driver_ids: List[str]
) -> Dict[str, float]:
    """
    Computes Brier scores and skill scores for probabilistic predictions.
    """
    df_eval = pd.DataFrame({
        "Abbreviation": driver_ids,
        "ActualFinish": actual_finishes
    })
    merged = sim_df.merge(df_eval, on="Abbreviation", how="inner")
    if merged.empty:
        return {}

    n = len(merged)
    # Win probability Brier score
    p_win = merged["WinProbability"].values / 100.0
    y_win = (merged["ActualFinish"].values == 1).astype(float)
    brier_win = float(np.mean((p_win - y_win) ** 2))
    naive_win_brier = float(np.mean(((1.0 / n) - y_win) ** 2))
    bss_win = 1.0 - (brier_win / naive_win_brier) if naive_win_brier > 0 else 0.0

    # Top-3 probability Brier score
    p_top3 = merged["Top3Probability"].values / 100.0
    y_top3 = (merged["ActualFinish"].values <= 3).astype(float)
    brier_top3 = float(np.mean((p_top3 - y_top3) ** 2))
    naive_top3_brier = float(np.mean(((3.0 / n) - y_top3) ** 2))
    bss_top3 = 1.0 - (brier_top3 / naive_top3_brier) if naive_top3_brier > 0 else 0.0

    # Top-5 probability Brier score
    p_top5 = merged.get("Top5Probability", pd.Series(np.zeros(n))).values / 100.0
    y_top5 = (merged["ActualFinish"].values <= 5).astype(float)
    brier_top5 = float(np.mean((p_top5 - y_top5) ** 2))

    return {
        "brier_win": round(brier_win, 4),
        "bss_win": round(bss_win, 4),
        "brier_top3": round(brier_top3, 4),
        "bss_top3": round(bss_top3, 4),
        "brier_top5": round(brier_top5, 4),
    }


def compute_paired_comparison(
    model_errors: List[float],
    baseline_errors: List[float]
) -> Dict[str, Any]:
    """
    Computes paired difference statistics, effect size (Cohen's d), and significance p-value.
    """
    diffs = np.array(model_errors) - np.array(baseline_errors)
    n = len(diffs)
    if n == 0:
        return {}

    mean_diff = float(np.mean(diffs))
    median_diff = float(np.median(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0

    cohens_d = (mean_diff / std_diff) if std_diff > 1e-6 else 0.0

    p_value = 1.0
    test_type = "none"
    if n >= 5:
        try:
            # Check if all diffs are zero
            if np.all(diffs == 0):
                p_value = 1.0
                test_type = "exact_zero"
            else:
                stat, p_val = wilcoxon(diffs, zero_method="pratt")
                p_value = float(p_val)
                test_type = "wilcoxon"
        except Exception:
            try:
                stat, p_val = ttest_rel(model_errors, baseline_errors)
                p_value = float(p_val)
                test_type = "paired_t"
            except Exception:
                p_value = 1.0

    return {
        "n_races": n,
        "mean_diff": round(mean_diff, 4),
        "median_diff": round(median_diff, 4),
        "std_diff": round(std_diff, 4),
        "cohens_d": round(cohens_d, 4),
        "p_value": round(p_value, 4),
        "test_type": test_type,
        "statistically_significant": bool(p_value < 0.05)
    }
