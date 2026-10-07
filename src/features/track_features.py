"""
Track-Specific Historical Feature Engineering Module
====================================================
Computes circuit-specific driver and team history (e.g. Baku 2021-2025).
Conditioned on historical_data_available configuration in tracks.json.
If circuit has no relevant recent history (e.g. Sepang hiatus), returns neutral defaults.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from src.utils.driver_mapping import normalize_driver_series


def build_track_history_features(
    historical_race_df: Optional[pd.DataFrame],
    drivers: List[str],
    default_pos: float = 12.0
) -> pd.DataFrame:
    """
    Computes historical circuit driver performance metrics.

    Parameters
    ----------
    historical_race_df : pd.DataFrame, optional
        Multi-year historical race results for this specific circuit (e.g. Baku 2021-2025).
    drivers : list of str
        Current driver lineup to compute features for.
    default_pos : float
        Default position for drivers with no previous circuit appearances.

    Returns
    -------
    pd.DataFrame
        DataFrame with circuit-specific performance indicators for each driver.
    """
    master = pd.DataFrame({"Abbreviation": list(drivers)})

    if historical_race_df is None or historical_race_df.empty:
        master["track_avg_finish"] = default_pos
        master["track_avg_grid"] = default_pos
        master["track_avg_pos_gained"] = 0.0
        master["track_races_completed"] = 0
        return master

    df = historical_race_df.copy()
    if "Abbreviation" not in df.columns:
        if "Driver" in df.columns:
            df["Abbreviation"] = df["Driver"]
        else:
            master["track_avg_finish"] = default_pos
            master["track_avg_grid"] = default_pos
            master["track_avg_pos_gained"] = 0.0
            master["track_races_completed"] = 0
            return master

    df["Abbreviation"] = normalize_driver_series(df["Abbreviation"])
    pos_col = "Position" if "Position" in df.columns else "ActualFinish"
    grid_col = "GridPosition" if "GridPosition" in df.columns else "StartingGrid"

    df["Position"] = pd.to_numeric(df[pos_col], errors="coerce")
    df["GridPosition"] = pd.to_numeric(df[grid_col], errors="coerce")
    df["pos_gained"] = df["GridPosition"] - df["Position"]

    agg = df.groupby("Abbreviation").agg(
        track_avg_finish=("Position", "mean"),
        track_avg_grid=("GridPosition", "mean"),
        track_avg_pos_gained=("pos_gained", "mean"),
        track_races_completed=("Position", "count")
    ).reset_index()

    merged = master.merge(agg, on="Abbreviation", how="left")
    merged["track_avg_finish"] = merged["track_avg_finish"].fillna(default_pos)
    merged["track_avg_grid"] = merged["track_avg_grid"].fillna(default_pos)
    merged["track_avg_pos_gained"] = merged["track_avg_pos_gained"].fillna(0.0)
    merged["track_races_completed"] = merged["track_races_completed"].fillna(0).astype(int)

    return merged
