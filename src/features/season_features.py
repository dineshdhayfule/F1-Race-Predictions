"""
Season Form and Historical Trajectory Feature Engineering
==========================================================
Computes driver and team rolling performance features strictly up to Round < target_round.
Zero forward leakage: target round race data is never included in form statistics.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


def is_dnf(status: Any) -> bool:
    """Classifies finishing status string into DNF (True) or classified finish (False)."""
    if pd.isna(status):
        return False
    text = str(status).lower()
    terms = [
        "retired", "collision", "accident", "engine", "gearbox",
        "hydraulics", "electrical", "brakes", "suspension",
        "power unit", "puncture", "water pressure", "oil pressure",
        "wheel", "damage", "spins off", "dnf"
    ]
    return any(term in text for term in terms)


def ewma(values: list, alpha: float = 0.3) -> float:
    """Exponentially weighted moving average with newest values having highest weight."""
    s = pd.Series(values).dropna()
    if s.empty:
        return np.nan
    return float(s.ewm(alpha=alpha, adjust=False).mean().iloc[-1])


def build_season_form(
    target_round: int,
    races_df: pd.DataFrame,
    qualis_df: Optional[pd.DataFrame] = None,
    alpha: float = 0.3,
    default_pos: float = 12.0
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Computes driver and team form metrics strictly from completed rounds where Round < target_round.

    Parameters
    ----------
    target_round : int
        Current round being predicted.
    races_df : pd.DataFrame
        Historical season race results (must contain 'Round', 'Abbreviation', 'Position', 'GridPosition', 'Points').
    qualis_df : pd.DataFrame, optional
        Historical season qualifying results.
    alpha : float
        Decay parameter for EWMA form calculation.
    default_pos : float
        Default position for drivers/teams with no prior completed races (midfield = 12.0).

    Returns
    -------
    tuple of (driver_stats, team_stats)
    """
    # Strict chronological cutoff: Round < target_round
    if "Round" not in races_df.columns:
        raise ValueError("races_df must contain 'Round' column.")

    prior = races_df[races_df["Round"] < target_round].copy()
    prior_q = (
        qualis_df[qualis_df["Round"] < target_round].copy()
        if (qualis_df is not None and not qualis_df.empty and "Round" in qualis_df.columns)
        else pd.DataFrame()
    )

    if "is_dnf" not in prior.columns and "Status" in prior.columns:
        prior["is_dnf"] = prior["Status"].apply(is_dnf)

    driver_stats: Dict[str, Dict[str, Any]] = {}
    team_stats: Dict[str, Dict[str, Any]] = {}

    if not prior.empty:
        for drv, grp in prior.groupby("Abbreviation"):
            grp = grp.sort_values("Round")
            qgrp = (
                prior_q[prior_q["Abbreviation"] == drv].sort_values("Round")
                if not prior_q.empty
                else pd.DataFrame()
            )

            finishes = pd.to_numeric(grp["Position"], errors="coerce").dropna().tolist()
            grids = pd.to_numeric(grp["GridPosition"], errors="coerce").dropna().tolist()
            points = pd.to_numeric(grp.get("Points", 0), errors="coerce").fillna(0).tolist()
            qvals = (
                pd.to_numeric(qgrp["Position"], errors="coerce").dropna().tolist()
                if not qgrp.empty and "Position" in qgrp.columns
                else []
            )

            driver_stats[str(drv)] = {
                "avg_finish": float(np.mean(finishes)) if finishes else default_pos,
                "avg_grid": float(np.mean(grids)) if grids else default_pos,
                "total_points": float(sum(points)),
                "races_completed": int(len(grp)),
                "recent_finish": float(finishes[-1]) if finishes else default_pos,
                "recent_grid": float(grids[-1]) if grids else default_pos,
                "recent_points": float(points[-1]) if points else 0.0,
                "driver_finish_ewma": float(ewma(finishes, alpha=alpha)) if finishes else default_pos,
                "driver_quali_ewma": float(ewma(qvals, alpha=alpha)) if qvals else (
                    float(ewma(grids, alpha=alpha)) if grids else default_pos
                ),
                "driver_dnf_rate": float(grp["is_dnf"].mean()) if "is_dnf" in grp else 0.0,
            }

        team_col = "TeamName" if "TeamName" in prior.columns else "Team"
        if team_col in prior.columns:
            for team, grp in prior.groupby(team_col):
                grp = grp.sort_values("Round")
                finishes = pd.to_numeric(grp["Position"], errors="coerce").dropna().tolist()
                points = pd.to_numeric(grp.get("Points", 0), errors="coerce").fillna(0).tolist()
                rounds = grp["Round"].nunique()

                team_stats[str(team)] = {
                    "team_avg_finish": float(np.mean(finishes)) if finishes else default_pos,
                    "team_finish_ewma": float(ewma(finishes, alpha=alpha)) if finishes else default_pos,
                    "team_points": float(sum(points)),
                    "team_points_rate": float(sum(points) / max(rounds * 2, 1)),
                }

    return driver_stats, team_stats
