"""
Teammate-Relative Feature Engineering Module
============================================
Calculates intra-team comparative metrics:
1. teammate_quali_gap_pct: Percentage lap time delta to teammate in qualifying.
   Negative = faster than teammate; Positive = slower than teammate.
   Evaluates to NaN if qualifying times are missing, invalid, or teammate is missing.
2. teammate_grid_delta: Starting grid difference to teammate (ActualStartingGrid_driver - ActualStartingGrid_teammate).
   Negative = starting ahead of teammate; Positive = starting behind teammate.
   Evaluates to NaN if grid is missing or teammate is missing.
3. driver_teammate_h2h_ratio: Empirical Bayes shrunk head-to-head finish win-rate in prior races (< r).
   Shrunk with prior M=3 (prior=0.50) and bounded to [0.15, 0.85].
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from src.utils.driver_mapping import normalize_driver_series
from src.utils.timing import timedelta_to_seconds


TERMINAL_STATUSES = {
    "Retired", "Collision", "Accident", "Engine", "Gearbox",
    "Brakes", "Suspension", "Did not start", "Withdrew"
}


def compute_teammate_quali_gap(
    quali_df: Optional[pd.DataFrame],
    drivers: List[str]
) -> pd.DataFrame:
    """
    Computes percentage lap time difference to teammate in qualifying:
    teammate_quali_gap_pct = (time_driver - time_teammate) / time_teammate * 100.
    Negative = faster than teammate. Positive = slower.

    STRICT MISSINGNESS RULE:
    If either driver's qualifying time is missing, invalid, or teammate cannot be
    identified, teammate_quali_gap_pct must be NaN. No synthetic offsets.
    """
    master = pd.DataFrame({"Abbreviation": list(drivers)})
    if quali_df is None or quali_df.empty:
        master["teammate_quali_gap_pct"] = np.nan
        return master

    q = quali_df.copy()
    if "Abbreviation" not in q.columns:
        if "Driver" in q.columns:
            q["Abbreviation"] = q["Driver"]
        else:
            master["teammate_quali_gap_pct"] = np.nan
            return master

    q["Abbreviation"] = normalize_driver_series(q["Abbreviation"])

    team_col = "TeamName" if "TeamName" in q.columns else ("Team" if "Team" in q.columns else None)
    if team_col is None:
        master["teammate_quali_gap_pct"] = np.nan
        return master

    # Extract best valid lap time in seconds
    for col in ("Q1", "Q2", "Q3"):
        if col in q.columns:
            q[f"{col}_s"] = q[col].apply(timedelta_to_seconds)
        else:
            q[f"{col}_s"] = np.nan

    q["best_lap"] = q[["Q1_s", "Q2_s", "Q3_s"]].apply(
        lambda row: min([x for x in row if pd.notna(x) and x > 50], default=np.nan),
        axis=1
    )

    gaps = {}
    for team, group in q.groupby(team_col):
        drvs = group["Abbreviation"].tolist()
        if len(drvs) == 2:
            d1, d2 = drvs[0], drvs[1]
            t1 = group.loc[group["Abbreviation"] == d1, "best_lap"].values[0]
            t2 = group.loc[group["Abbreviation"] == d2, "best_lap"].values[0]

            if pd.notna(t1) and pd.notna(t2) and t1 > 0 and t2 > 0:
                gaps[d1] = (t1 - t2) / t2 * 100.0
                gaps[d2] = (t2 - t1) / t1 * 100.0
            else:
                gaps[d1] = np.nan
                gaps[d2] = np.nan
        else:
            for d in drvs:
                gaps[d] = np.nan

    res = pd.DataFrame(list(gaps.items()), columns=["Abbreviation", "teammate_quali_gap_pct"])
    merged = master.merge(res, on="Abbreviation", how="left")
    return merged


def compute_teammate_grid_delta(
    grid_df: Optional[pd.DataFrame],
    drivers: List[str],
    grid_col: str = "ActualStartingGrid"
) -> pd.DataFrame:
    """
    Computes starting grid delta to teammate:
    teammate_grid_delta = ActualStartingGrid_driver - ActualStartingGrid_teammate.
    Negative = starting ahead of teammate. Positive = starting behind teammate.

    If starting grid is unheld, missing, or teammate cannot be identified, returns NaN.
    """
    master = pd.DataFrame({"Abbreviation": list(drivers)})
    if grid_df is None or grid_df.empty:
        master["teammate_grid_delta"] = np.nan
        return master

    gdf = grid_df.copy()
    if "Abbreviation" not in gdf.columns:
        if "Driver" in gdf.columns:
            gdf["Abbreviation"] = gdf["Driver"]
        else:
            master["teammate_grid_delta"] = np.nan
            return master

    gdf["Abbreviation"] = normalize_driver_series(gdf["Abbreviation"])

    team_col = "TeamName" if "TeamName" in gdf.columns else ("Team" if "Team" in gdf.columns else None)
    col = grid_col if grid_col in gdf.columns else ("GridPosition" if "GridPosition" in gdf.columns else None)

    if team_col is None or col is None:
        master["teammate_grid_delta"] = np.nan
        return master

    deltas = {}
    for team, group in gdf.groupby(team_col):
        drvs = group["Abbreviation"].tolist()
        if len(drvs) == 2:
            d1, d2 = drvs[0], drvs[1]
            g1 = pd.to_numeric(group.loc[group["Abbreviation"] == d1, col].values[0], errors="coerce")
            g2 = pd.to_numeric(group.loc[group["Abbreviation"] == d2, col].values[0], errors="coerce")

            if pd.notna(g1) and pd.notna(g2):
                deltas[d1] = float(g1 - g2)
                deltas[d2] = float(g2 - g1)
            else:
                deltas[d1] = np.nan
                deltas[d2] = np.nan
        else:
            for d in drvs:
                deltas[d] = np.nan

    res = pd.DataFrame(list(deltas.items()), columns=["Abbreviation", "teammate_grid_delta"])
    merged = master.merge(res, on="Abbreviation", how="left")
    return merged


def compute_rolling_teammate_h2h(
    prior_races_df: pd.DataFrame,
    current_lineup_df: pd.DataFrame,
    prior_weight: float = 3.0,
    lower_bound: float = 0.15,
    upper_bound: float = 0.85
) -> Dict[str, float]:
    """
    Computes Empirical Bayes shrunk head-to-head win-rate against current teammate across prior rounds.
    Strict anti-leakage guarantee: relies exclusively on completed prior_races_df (Round < current_round).

    shrunk_h2h = (wins + prior_weight * 0.50) / (comparable_races + prior_weight)
    Clamped to [lower_bound, upper_bound].
    """
    if current_lineup_df is None or current_lineup_df.empty:
        return {}

    cldf = current_lineup_df.copy()
    if "Abbreviation" not in cldf.columns and "Driver" in cldf.columns:
        cldf["Abbreviation"] = cldf["Driver"]

    team_col = "TeamName" if "TeamName" in cldf.columns else ("Team" if "Team" in cldf.columns else None)
    if team_col is None or "Abbreviation" not in cldf.columns:
        return {drv: 0.50 for drv in cldf.get("Abbreviation", [])}

    cldf["Abbreviation"] = normalize_driver_series(cldf["Abbreviation"])

    # Find teammate mapping for the current lineup
    teammate_map = {}
    for team, group in cldf.groupby(team_col):
        drvs = group["Abbreviation"].unique().tolist()
        if len(drvs) == 2:
            teammate_map[drvs[0]] = drvs[1]
            teammate_map[drvs[1]] = drvs[0]

    h2h_results = {}
    all_drivers = cldf["Abbreviation"].unique().tolist()

    if prior_races_df is None or prior_races_df.empty:
        return {drv: 0.50 for drv in all_drivers}

    pdf = prior_races_df.copy()
    pdf["Abbreviation"] = normalize_driver_series(pdf["Abbreviation"])
    p_team_col = "TeamName" if "TeamName" in pdf.columns else ("Team" if "Team" in pdf.columns else None)

    for drv in all_drivers:
        teammate = teammate_map.get(drv)
        if not teammate:
            h2h_results[drv] = 0.50
            continue

        drv_races = pdf[pdf["Abbreviation"] == drv]
        tm_races = pdf[pdf["Abbreviation"] == teammate]

        if drv_races.empty or tm_races.empty:
            h2h_results[drv] = 0.50
            continue

        # Common rounds on the same team
        merged = drv_races.merge(tm_races, on="Round", suffixes=("_drv", "_tm"))
        if p_team_col:
            # Ensure both raced for the same team in that prior round
            t_col_drv = f"{p_team_col}_drv"
            t_col_tm = f"{p_team_col}_tm"
            if t_col_drv in merged.columns and t_col_tm in merged.columns:
                merged = merged[merged[t_col_drv] == merged[t_col_tm]]

        if merged.empty:
            h2h_results[drv] = 0.50
            continue

        wins = 0.0
        valid_races = 0

        for _, row in merged.iterrows():
            pos_drv = pd.to_numeric(row.get("Position_drv"), errors="coerce")
            pos_tm = pd.to_numeric(row.get("Position_tm"), errors="coerce")

            status_drv = str(row.get("Status_drv", ""))
            status_tm = str(row.get("Status_tm", ""))

            dnf_drv = status_drv in TERMINAL_STATUSES
            dnf_tm = status_tm in TERMINAL_STATUSES

            # Both DNF: neutral, not counted
            if dnf_drv and dnf_tm:
                continue

            if not dnf_drv and dnf_tm:
                wins += 1.0
                valid_races += 1
            elif dnf_drv and not dnf_tm:
                wins += 0.0
                valid_races += 1
            elif pd.notna(pos_drv) and pd.notna(pos_tm):
                if pos_drv < pos_tm:
                    wins += 1.0
                elif pos_drv > pos_tm:
                    wins += 0.0
                else:
                    wins += 0.5
                valid_races += 1

        shrunk_rate = (wins + prior_weight * 0.50) / float(valid_races + prior_weight)
        bounded_rate = float(np.clip(shrunk_rate, lower_bound, upper_bound))
        h2h_results[drv] = round(bounded_rate, 4)

    return h2h_results
