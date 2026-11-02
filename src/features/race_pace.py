"""
Race Pace Intelligence Feature Module
======================================
Extracts representative multi-lap long-run stints from practice session telemetry,
filters out in/out/abnormal laps, estimates fuel-corrected pace, and calculates
long-run race pace features for Formula 1 race prediction.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

from src.utils.driver_mapping import normalize_driver, normalize_driver_series
from src.utils.timing import timedelta_to_seconds
from src.utils.logging_utils import get_logger

logger = get_logger("race_pace")


def extract_stints_from_laps(
    laps_df: pd.DataFrame,
    min_consecutive_laps: int = 4,
    max_lap_time_multiplier: float = 1.12,
    fuel_correction_per_lap: float = 0.035,
) -> List[Dict[str, Any]]:
    """
    Extracts valid multi-lap stints from session lap data for a driver or session.
    Filters out:
    - In-laps (PitInTime or flag)
    - Out-laps (PitOutTime or flag)
    - Deleted laps (Deleted == True)
    - Unrepresentative or anomalous laps (> max_lap_time_multiplier * median stint time)

    Parameters
    ----------
    laps_df : pd.DataFrame
        DataFrame of session laps containing at minimum:
        ['Driver' or 'Abbreviation', 'LapTime' or 'LapTimeSeconds', 'Stint', 'Compound']
    min_consecutive_laps : int
        Minimum clean laps required for a stint to qualify as a representative long-run.
    max_lap_time_multiplier : float
        Laps exceeding this multiple of the stint median are filtered out as anomalous.
    fuel_correction_per_lap : float
        Estimated seconds gained per lap as fuel is burned off during a stint.

    Returns
    -------
    list of dict
        List of extracted valid stint dictionaries.
    """
    if laps_df is None or laps_df.empty:
        return []

    df = laps_df.copy()

    # Standardize driver abbreviation
    driver_col = "Abbreviation" if "Abbreviation" in df.columns else ("Driver" if "Driver" in df.columns else None)
    if driver_col is None:
        return []
    df["Abbreviation"] = normalize_driver_series(df[driver_col])

    # Standardize lap time to seconds
    if "LapTimeSeconds" not in df.columns:
        if "LapTime" in df.columns:
            df["LapTimeSeconds"] = df["LapTime"].apply(timedelta_to_seconds)
        elif "LapSeconds" in df.columns:
            df["LapTimeSeconds"] = pd.to_numeric(df["LapSeconds"], errors="coerce")
        else:
            return []
    else:
        df["LapTimeSeconds"] = pd.to_numeric(df["LapTimeSeconds"], errors="coerce")

    # Exclude non-positive or unrealistically short lap times (< 50 seconds for F1 circuits)
    df = df[df["LapTimeSeconds"] > 50.0].copy()
    if df.empty:
        return []

    # Exclude deleted laps
    if "Deleted" in df.columns:
        df = df[df["Deleted"] != True]

    # Stint grouping column
    if "Stint" not in df.columns:
        df["Stint"] = 1

    if "Compound" not in df.columns:
        df["Compound"] = "UNKNOWN"

    stints = []

    # Group by driver, stint number, and compound
    for (drv, stint_no, compound), group in df.groupby(["Abbreviation", "Stint", "Compound"]):
        # Sort chronologically by lap number
        lap_col = "LapNumber" if "LapNumber" in group.columns else None
        if lap_col:
            group = group.sort_values(lap_col).reset_index(drop=True)
        else:
            group = group.reset_index(drop=True)

        # Exclude pit-in and pit-out laps if indicators are present
        clean_mask = pd.Series(True, index=group.index)
        if "PitOutTime" in group.columns:
            clean_mask &= group["PitOutTime"].isna()
        if "PitInTime" in group.columns:
            clean_mask &= group["PitInTime"].isna()

        clean_laps = group[clean_mask].copy()
        if len(clean_laps) < min_consecutive_laps:
            continue

        # Filter anomalous laps using median threshold
        med_lap = clean_laps["LapTimeSeconds"].median()
        valid_laps = clean_laps[clean_laps["LapTimeSeconds"] <= med_lap * max_lap_time_multiplier].copy()

        if len(valid_laps) < min_consecutive_laps:
            continue

        raw_times = valid_laps["LapTimeSeconds"].values
        n_laps = len(raw_times)

        # Estimated fuel-load correction: pace improves by ~0.035s per lap as fuel burns
        # Measured pace is uncorrected; fuel-adjusted pace normalizes across stint progression
        lap_indices = np.arange(n_laps)
        fuel_adj_times = raw_times + (fuel_correction_per_lap * lap_indices)

        # Robust linear degradation slope (seconds per lap increase)
        if n_laps >= 4:
            poly = np.polyfit(lap_indices, raw_times, 1)
            deg_slope = float(poly[0])
        else:
            deg_slope = 0.0

        stint_info = {
            "driver": drv,
            "stint_number": int(stint_no) if pd.notna(stint_no) else 1,
            "compound": str(compound).upper(),
            "lap_count": n_laps,
            "raw_lap_times": raw_times.tolist(),
            "fuel_adj_times": fuel_adj_times.tolist(),
            "mean_lap": float(np.mean(raw_times)),
            "median_lap": float(np.median(raw_times)),
            "best_lap": float(np.min(raw_times)),
            "std_lap": float(np.std(raw_times)),
            "fuel_adj_mean_lap": float(np.mean(fuel_adj_times)),
            "deg_slope": round(deg_slope, 4),
        }
        stints.append(stint_info)

    return stints


def compute_driver_race_pace(
    stints: List[Dict[str, Any]],
    drivers: List[str],
    reference_compound: str = "MEDIUM",
) -> pd.DataFrame:
    """
    Aggregates driver-level long-run race pace metrics across extracted stints.

    Parameters
    ----------
    stints : list of dict
        Extracted valid stint dictionaries.
    drivers : list of str
        Target list of driver abbreviations.
    reference_compound : str
        Default reference compound for baseline comparisons.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by driver with calculated race pace statistics.
    """
    records = []
    stints_by_driver: Dict[str, List[Dict[str, Any]]] = {d: [] for d in drivers}
    for s in stints:
        drv = s["driver"]
        if drv in stints_by_driver:
            stints_by_driver[drv].append(s)

    for drv in drivers:
        d_stints = stints_by_driver.get(drv, [])
        if not d_stints:
            records.append({
                "Abbreviation": drv,
                "has_long_run": False,
                "best_stint_mean": np.nan,
                "best_stint_median": np.nan,
                "fuel_adj_pace": np.nan,
                "race_pace_consistency": np.nan,
                "race_pace_sample_count": 0,
                "long_run_stint_length": 0,
                "primary_compound": "NONE",
                "deg_slope": np.nan,
            })
            continue

        # Sort stints by lap count descending, then mean lap time ascending
        # Prioritize stints on race-relevant compounds with longest valid runs
        d_stints.sort(key=lambda x: (-x["lap_count"], x["fuel_adj_mean_lap"]))
        best_stint = d_stints[0]

        total_laps = sum(s["lap_count"] for s in d_stints)
        max_stint_len = max(s["lap_count"] for s in d_stints)
        # Pooled standard deviation of lap times across long runs
        all_laps = [lap for s in d_stints for lap in s["raw_lap_times"]]
        consistency = float(np.std(all_laps)) if len(all_laps) > 1 else best_stint["std_lap"]

        records.append({
            "Abbreviation": drv,
            "has_long_run": True,
            "best_stint_mean": round(best_stint["mean_lap"], 3),
            "best_stint_median": round(best_stint["median_lap"], 3),
            "fuel_adj_pace": round(best_stint["fuel_adj_mean_lap"], 3),
            "race_pace_consistency": round(consistency, 3),
            "race_pace_sample_count": int(total_laps),
            "long_run_stint_length": int(max_stint_len),
            "primary_compound": best_stint["compound"],
            "deg_slope": round(best_stint["deg_slope"], 4),
        })

    df = pd.DataFrame(records)
    return df


def build_race_pace_features(
    session_dfs: Dict[str, Optional[pd.DataFrame]],
    drivers: List[str],
    quali_positions: Optional[Dict[str, float]] = None,
    expected_grid_size: int = 22,
) -> pd.DataFrame:
    """
    Unified entry point for race pace feature extraction across available practice sessions.
    Respects stage cutoffs by operating only on sessions passed in session_dfs.

    Parameters
    ----------
    session_dfs : dict
        Mapping of session names ('fp1', 'fp2', 'fp3') to session DataFrames.
    drivers : list of str
        List of driver abbreviations.
    quali_positions : dict, optional
        Mapping of driver -> qualifying rank (to calculate race_pace_vs_quali_delta).
    expected_grid_size : int
        Championship grid size (default 22).

    Returns
    -------
    pd.DataFrame
        DataFrame containing standard engineered race pace features:
        - long_run_pace_gap_pct
        - long_run_pace_rank
        - race_pace_consistency
        - race_pace_sample_count
        - long_run_stint_length
        - race_pace_vs_quali_delta
        - race_pace_confidence
    """
    master = pd.DataFrame({"Abbreviation": list(drivers)})

    # Pool laps across all available practice sessions (FP1, FP2, FP3)
    all_laps_list = []
    # Normalize dictionary keys to lowercase for robust lookup
    clean_sessions = {str(k).lower(): v for k, v in session_dfs.items()} if session_dfs else {}
    # Prioritize FP2 (the canonical long-run practice session) followed by FP1, then FP3
    for s_name in ("fp2", "fp1", "fp3"):
        s_df = clean_sessions.get(s_name)
        if s_df is not None and not s_df.empty and ("LapTime" in s_df.columns or "LapTimeSeconds" in s_df.columns):
            all_laps_list.append(s_df)

    if not all_laps_list:
        # Graceful neutral defaults when practice data is not available
        master["long_run_pace_gap_pct"] = np.nan
        master["long_run_pace_rank"] = float(expected_grid_size) / 2.0
        master["race_pace_consistency"] = np.nan
        master["race_pace_sample_count"] = 0
        master["long_run_stint_length"] = 0
        master["race_pace_vs_quali_delta"] = 0.0
        master["race_pace_confidence"] = 0.0
        return master

    pooled_laps = pd.concat(all_laps_list, ignore_index=True)
    stints = extract_stints_from_laps(pooled_laps, min_consecutive_laps=4)
    driver_pace = compute_driver_race_pace(stints, drivers)

    merged = master.merge(driver_pace, on="Abbreviation", how="left")

    # Determine field benchmark pace (fastest observed long run)
    valid_paces = merged["fuel_adj_pace"].dropna()
    if not valid_paces.empty:
        pole_race_pace = valid_paces.min()
        merged["long_run_pace_gap_pct"] = (
            (merged["fuel_adj_pace"] - pole_race_pace) / pole_race_pace * 100.0
        ).round(3)
        # Compute ordinal rank of long-run pace (1 = fastest)
        merged["long_run_pace_rank"] = merged["fuel_adj_pace"].rank(method="min", ascending=True)
        # Fill missing drivers with default midfield/tail rank
        merged["long_run_pace_rank"] = merged["long_run_pace_rank"].fillna(float(expected_grid_size))
    else:
        merged["long_run_pace_gap_pct"] = np.nan
        merged["long_run_pace_rank"] = float(expected_grid_size) / 2.0

    # Confidence based on valid sample count: 12+ laps = high confidence (1.0)
    merged["race_pace_confidence"] = (merged["race_pace_sample_count"] / 12.0).clip(0.0, 1.0).round(2)

    # Race pace vs qualifying position delta
    # Positive delta means driver has stronger race pace than single-lap qualifying (a "Sunday car")
    if quali_positions:
        q_series = merged["Abbreviation"].map(quali_positions).fillna(float(expected_grid_size))
        merged["race_pace_vs_quali_delta"] = (q_series - merged["long_run_pace_rank"]).round(1)
    else:
        merged["race_pace_vs_quali_delta"] = 0.0

    output_cols = [
        "Abbreviation",
        "long_run_pace_gap_pct",
        "long_run_pace_rank",
        "race_pace_consistency",
        "race_pace_sample_count",
        "long_run_stint_length",
        "race_pace_vs_quali_delta",
        "race_pace_confidence",
    ]
    return merged[output_cols].copy()

