"""
Weekend Session Feature Engineering Module
==========================================
Computes practice (FP1, FP2, FP3) and qualifying performance metrics for a Grand Prix weekend.
Robust across all prediction stages (pre_fp1, after_fp1, after_fp2, after_fp3, after_qualifying).
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.utils.timing import timedelta_to_seconds
from src.utils.driver_mapping import normalize_driver_series


def build_practice_features(
    session_dfs: Dict[str, Optional[pd.DataFrame]],
    drivers: List[str]
) -> pd.DataFrame:
    """
    Computes aggregated practice session features for the provided drivers.

    Parameters
    ----------
    session_dfs : dict
        Mapping of session names ('fp1', 'fp2', 'fp3') to their respective DataFrames.
    drivers : list of str
        List of driver abbreviations on the weekend entry list.

    Returns
    -------
    pd.DataFrame
        DataFrame with practice timing, lap counts, rank, and percentage gaps.
    """
    output = pd.DataFrame({"Abbreviation": list(drivers)})

    for session in ("fp1", "fp2", "fp3"):
        df = session_dfs.get(session)
        if df is None or df.empty:
            output[f"{session}_best_lap"] = np.nan
            output[f"{session}_gap_pct"] = np.nan
            output[f"{session}_laps"] = 0
            continue

        df = df.copy()
        if "Abbreviation" not in df.columns:
            if "Driver" in df.columns:
                df["Abbreviation"] = df["Driver"]
            else:
                output[f"{session}_best_lap"] = np.nan
                output[f"{session}_gap_pct"] = np.nan
                output[f"{session}_laps"] = 0
                continue

        df["Abbreviation"] = normalize_driver_series(df["Abbreviation"])

        time_col = None
        for col in ["LapTimeSeconds", "LapSeconds", "LapTime"]:
            if col in df.columns:
                time_col = col
                break

        if time_col is None:
            output[f"{session}_best_lap"] = np.nan
            output[f"{session}_gap_pct"] = np.nan
            output[f"{session}_laps"] = 0
            continue

        if time_col == "LapTime":
            df["LapTimeSeconds"] = df["LapTime"].apply(timedelta_to_seconds)
        else:
            df["LapTimeSeconds"] = pd.to_numeric(df[time_col], errors="coerce")

        if "Deleted" in df.columns:
            df = df[df["Deleted"] != True]

        valid = df[df["LapTimeSeconds"] > 50].copy()
        if valid.empty:
            output[f"{session}_best_lap"] = np.nan
            output[f"{session}_gap_pct"] = np.nan
            output[f"{session}_laps"] = 0
            continue

        bests = valid.groupby("Abbreviation")["LapTimeSeconds"].min()
        counts = valid.groupby("Abbreviation")["LapTimeSeconds"].count()
        session_best = bests.min()

        output[f"{session}_best_lap"] = output["Abbreviation"].map(bests)
        output[f"{session}_laps"] = output["Abbreviation"].map(counts).fillna(0).astype(int)
        output[f"{session}_gap_pct"] = (
            (output[f"{session}_best_lap"] - session_best) / session_best * 100.0
            if pd.notna(session_best) and session_best > 0 else np.nan
        )

    all_laps = output[["fp1_best_lap", "fp2_best_lap", "fp3_best_lap"]]
    output["best_practice_lap"] = all_laps.min(axis=1)
    output["avg_practice_lap"] = all_laps.mean(axis=1)

    overall_best = output["best_practice_lap"].min()
    if pd.notna(overall_best) and overall_best > 0:
        output["practice_gap_pct"] = (
            (output["best_practice_lap"] - overall_best) / overall_best * 100.0
        )
    else:
        output["practice_gap_pct"] = np.nan

    output["practice_rank"] = (
        output["best_practice_lap"].rank(method="min", ascending=True)
    ).fillna(float(len(drivers)))

    output["practice_laps"] = output["fp1_laps"] + output["fp2_laps"] + output["fp3_laps"]
    output["practice_laps_norm"] = output["practice_laps"] / 70.0
    output["practice_sessions"] = output[["fp1_best_lap", "fp2_best_lap", "fp3_best_lap"]].notna().sum(axis=1)

    return output


def build_qualifying_features(
    quali_df: Optional[pd.DataFrame],
    drivers: List[str]
) -> pd.DataFrame:
    """
    Computes qualifying features for the given drivers.
    If qualifying is not available yet (e.g. stage earlier than after_qualifying),
    returns default neutral features with quali_no_time = 1.

    Parameters
    ----------
    quali_df : pd.DataFrame, optional
        Qualifying results with Q1, Q2, Q3 and Position columns.
    drivers : list of str
        List of driver abbreviations.

    Returns
    -------
    pd.DataFrame
        DataFrame with QualifyingPosition, gaps, and segment performance.
    """
    if quali_df is None or quali_df.empty:
        return pd.DataFrame({
            "Abbreviation": list(drivers),
            "QualifyingPosition": np.nan,
            "quali_pos_norm": np.nan,
            "quali_gap_pct": np.nan,
            "quali_no_time": 1,
            "q1_gap_pct": np.nan,
            "q2_gap_pct": np.nan,
            "q3_gap_pct": np.nan,
        })

    q = quali_df.copy()
    if "Abbreviation" not in q.columns:
        if "Driver" in q.columns:
            q["Abbreviation"] = q["Driver"]
        else:
            raise ValueError("Qualifying data missing driver abbreviation column.")

    q["Abbreviation"] = normalize_driver_series(q["Abbreviation"])
    pos_col = "Position" if "Position" in q.columns else "QualifyingPosition"
    q["Position"] = pd.to_numeric(q[pos_col], errors="coerce")

    for col in ("Q1", "Q2", "Q3"):
        if col in q.columns:
            q[f"{col}_s"] = q[col].apply(timedelta_to_seconds)
        else:
            q[f"{col}_s"] = np.nan

    q["best_quali_lap"] = q[["Q1_s", "Q2_s", "Q3_s"]].apply(
        lambda row: min([x for x in row if pd.notna(x) and x > 50], default=np.nan),
        axis=1,
    )

    pole = q["best_quali_lap"].min()
    q1_pole = q["Q1_s"].min()
    q2_pole = q["Q2_s"].min()
    q3_pole = q["Q3_s"].min()

    q["QualifyingPosition"] = q["Position"]
    n_drivers = len(drivers) if len(drivers) > 1 else 22
    q["quali_pos_norm"] = (q["QualifyingPosition"] - 1.0) / float(n_drivers - 1)
    q["quali_gap_pct"] = (
        (q["best_quali_lap"] - pole) / pole * 100.0
        if pd.notna(pole) and pole > 0 else np.nan
    )
    q["quali_no_time"] = q["best_quali_lap"].isna().astype(int)

    for src, dst, pole_time in (
        ("Q1_s", "q1_gap_pct", q1_pole),
        ("Q2_s", "q2_gap_pct", q2_pole),
        ("Q3_s", "q3_gap_pct", q3_pole),
    ):
        q[dst] = (
            (q[src] - pole_time) / pole_time * 100.0
            if pd.notna(pole_time) and pole_time > 0 else np.nan
        )

    cols = [
        "Abbreviation", "QualifyingPosition", "quali_pos_norm",
        "quali_gap_pct", "quali_no_time",
        "q1_gap_pct", "q2_gap_pct", "q3_gap_pct"
    ]
    res = q[cols].drop_duplicates("Abbreviation")

    # Merge with entry drivers to guarantee all drivers are represented
    master = pd.DataFrame({"Abbreviation": list(drivers)})
    merged = master.merge(res, on="Abbreviation", how="left")
    merged["quali_no_time"] = merged["quali_no_time"].fillna(1).astype(int)
    return merged


def build_sprint_qualifying_features(
    sprint_quali_df: Optional[pd.DataFrame],
    drivers: List[str]
) -> pd.DataFrame:
    """
    Computes sprint qualifying finishing position feature.
    Experimental Sprint features: supported and leakage-free, but Phase 6E showed only marginal/mixed predictive improvement.
    Returns NaN for all drivers if sprint qualifying session is unavailable
    (e.g. normal weekends, or stages prior to sprint qualifying).
    """
    master = pd.DataFrame({"Abbreviation": list(drivers)})
    if sprint_quali_df is None or sprint_quali_df.empty:
        master["sprint_quali_pos"] = np.nan
        return master

    sq = sprint_quali_df.copy()
    if "Abbreviation" not in sq.columns:
        if "Driver" in sq.columns:
            sq["Abbreviation"] = sq["Driver"]
        else:
            master["sprint_quali_pos"] = np.nan
            return master

    sq["Abbreviation"] = normalize_driver_series(sq["Abbreviation"])
    pos_col = "Position" if "Position" in sq.columns else ("QualifyingPosition" if "QualifyingPosition" in sq.columns else None)
    if pos_col and pos_col in sq.columns:
        sq["sprint_quali_pos"] = pd.to_numeric(sq[pos_col], errors="coerce")
    else:
        sq["sprint_quali_pos"] = np.nan

    res = sq[["Abbreviation", "sprint_quali_pos"]].drop_duplicates("Abbreviation")
    return master.merge(res, on="Abbreviation", how="left")


def build_sprint_race_features(
    sprint_race_df: Optional[pd.DataFrame],
    drivers: List[str]
) -> pd.DataFrame:
    """
    Computes sprint race finishing position and position change delta features.
    sprint_pos_delta = SprintStartingGrid - SprintFinish (positive = gained positions).
    Experimental Sprint features: supported and leakage-free, but Phase 6E showed only marginal/mixed predictive improvement.
    Returns NaN for all drivers if sprint race is unavailable
    (e.g. normal weekends, or stages prior to sprint race).
    """
    master = pd.DataFrame({"Abbreviation": list(drivers)})
    if sprint_race_df is None or sprint_race_df.empty:
        master["sprint_finish_pos"] = np.nan
        master["sprint_pos_delta"] = np.nan
        return master

    sr = sprint_race_df.copy()
    if "Abbreviation" not in sr.columns:
        if "Driver" in sr.columns:
            sr["Abbreviation"] = sr["Driver"]
        else:
            master["sprint_finish_pos"] = np.nan
            master["sprint_pos_delta"] = np.nan
            return master

    sr["Abbreviation"] = normalize_driver_series(sr["Abbreviation"])
    pos_col = "Position" if "Position" in sr.columns else ("FinishPosition" if "FinishPosition" in sr.columns else None)
    grid_col = "GridPosition" if "GridPosition" in sr.columns else ("StartingGrid" if "StartingGrid" in sr.columns else ("Grid" if "Grid" in sr.columns else None))

    if pos_col and pos_col in sr.columns:
        sr["sprint_finish_pos"] = pd.to_numeric(sr[pos_col], errors="coerce")
    else:
        sr["sprint_finish_pos"] = np.nan

    if grid_col and grid_col in sr.columns and pos_col and pos_col in sr.columns:
        grid_vals = pd.to_numeric(sr[grid_col], errors="coerce")
        sr["sprint_pos_delta"] = grid_vals - sr["sprint_finish_pos"]
    else:
        sr["sprint_pos_delta"] = np.nan

    res = sr[["Abbreviation", "sprint_finish_pos", "sprint_pos_delta"]].drop_duplicates("Abbreviation")
    return master.merge(res, on="Abbreviation", how="left")
