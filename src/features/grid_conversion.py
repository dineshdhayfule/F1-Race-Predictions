"""
Grid-to-Race Conversion Feature Engineering Module
==================================================
Models empirical translation of starting grid positions into race finishes
using grid buckets and empirical Bayes shrinkage.

Enforces:
1. Strict chronological cutoff: Round r only sees rounds < r.
2. DNF coordination: Computes classified/running conversion to prevent
   double-counting stochastic incident attrition in Monte Carlo.
3. Bucket grouping: P1-2 (Front Row), P3-5 (Top 5), P6-10 (Points),
   P11-16 (Midfield), P17-22 (Backmarkers) to eliminate small-sample noise.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from src.features.season_features import is_dnf


GRID_BUCKETS = {
    "front_row": (1, 2),
    "top_5": (3, 5),
    "points_zone": (6, 10),
    "midfield": (11, 16),
    "backmarkers": (17, 22),
}


def get_grid_bucket(grid_pos: float) -> str:
    """Classifies starting grid into standard competitive tiers."""
    pos = int(round(float(grid_pos)))
    if pos <= 2:
        return "front_row"
    elif pos <= 5:
        return "top_5"
    elif pos <= 10:
        return "points_zone"
    elif pos <= 16:
        return "midfield"
    else:
        return "backmarkers"


def compute_grid_conversion_map(
    historical_races_df: pd.DataFrame,
    expected_grid_size: int = 22,
    shrinkage_m: float = 5.0
) -> Dict[int, Dict[str, float]]:
    """
    Computes expected_grid_finish and grid_retention_prob for each grid position 1..N.
    Uses running/classified race results to prevent double-counting DNF shock.
    Shrinks bucket statistics smoothly toward prior grid position.

    Parameters
    ----------
    historical_races_df : pd.DataFrame
        Historical races strictly preceding the target round (Round < target_round).
    expected_grid_size : int
        Number of grid positions (default 22).
    shrinkage_m : float
        Empirical Bayes shrinkage weight towards prior (default 5.0).

    Returns
    -------
    dict
        Mapping of grid_pos (1..22) -> {
            'expected_grid_finish': float,
            'grid_retention_prob': float
        }
    """
    conversion_map: Dict[int, Dict[str, float]] = {}

    # If no prior historical races, return neutral identity prior
    if historical_races_df is None or historical_races_df.empty or "Position" not in historical_races_df.columns:
        for g in range(1, expected_grid_size + 1):
            prior_retention = 0.50 + 0.25 * ((g - 1) / max(expected_grid_size - 1, 1))
            conversion_map[g] = {
                "expected_grid_finish": float(g),
                "grid_retention_prob": round(float(prior_retention), 4),
            }
        return conversion_map

    df = historical_races_df.copy()
    if "is_dnf" not in df.columns:
        if "Status" in df.columns:
            df["is_dnf"] = df["Status"].apply(is_dnf)
        else:
            df["is_dnf"] = False

    df["GridPosition"] = pd.to_numeric(df["GridPosition"], errors="coerce")
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    valid_df = df.dropna(subset=["GridPosition", "Position"]).copy()

    # Precompute bucket metrics
    bucket_stats: Dict[str, Dict[str, float]] = {}
    for b_name, (g_min, g_max) in GRID_BUCKETS.items():
        b_df = valid_df[(valid_df["GridPosition"] >= g_min) & (valid_df["GridPosition"] <= g_max)]
        if b_df.empty:
            bucket_stats[b_name] = {
                "running_mean": (g_min + g_max) / 2.0,
                "retention_rate": 0.60,
                "count": 0.0,
            }
        else:
            # Running/classified finish (coordinating with Monte Carlo DNF modeling)
            running_finishes = b_df.loc[~b_df["is_dnf"], "Position"]
            if running_finishes.empty:
                running_mean = float(b_df["Position"].mean())
            else:
                running_mean = float(running_finishes.mean())

            # Retention rate: probability of finishing <= starting grid
            retention_rate = float((b_df["Position"] <= b_df["GridPosition"]).mean())
            bucket_stats[b_name] = {
                "running_mean": running_mean,
                "retention_rate": retention_rate,
                "count": float(len(b_df)),
            }

    # Generate smoothed stats for each grid position 1..N
    for g in range(1, expected_grid_size + 1):
        b_name = get_grid_bucket(g)
        b_stat = bucket_stats[b_name]
        n_count = b_stat["count"]

        # Prior for expected finish is the starting grid position itself
        prior_finish = float(g)
        # Prior for retention is smooth progression from ~0.50 (pole) to ~0.75 (rear)
        prior_retention = 0.50 + 0.25 * ((g - 1) / max(expected_grid_size - 1, 1))

        # Empirical Bayes shrinkage: (n * empirical + m * prior) / (n + m)
        denom = n_count + shrinkage_m
        if denom > 0:
            exp_finish = (n_count * b_stat["running_mean"] + shrinkage_m * prior_finish) / denom
            ret_prob = (n_count * b_stat["retention_rate"] + shrinkage_m * prior_retention) / denom
        else:
            exp_finish = prior_finish
            ret_prob = prior_retention

        conversion_map[g] = {
            "expected_grid_finish": round(float(exp_finish), 3),
            "grid_retention_prob": round(float(ret_prob), 4),
        }

    return conversion_map


def get_conversion_features_for_grid(
    grid_pos: float,
    conversion_map: Dict[int, Dict[str, float]],
    expected_grid_size: int = 22
) -> Dict[str, float]:
    """Helper to safely fetch or interpolate conversion features for a driver."""
    pos = int(np.clip(round(float(grid_pos)), 1, expected_grid_size))
    if pos in conversion_map:
        return conversion_map[pos]
    return {
        "expected_grid_finish": float(grid_pos),
        "grid_retention_prob": 0.50,
    }
