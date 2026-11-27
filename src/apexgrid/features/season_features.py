"""
Season Form and Historical Trajectory Feature Engineering
==========================================================
Computes driver and team rolling performance features strictly up to Round < target_round.
Zero forward leakage: target round race data is never included in form statistics.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


def classify_race_status(status: Any) -> str:
    """
    Standardized classification of F1 race finishing status.
    
    Returns one of:
      - 'FINISHED': Completed race as a classified/running finisher ('Finished', '+1 Lap', '+2 Laps', 'Lapped', etc.)
      - 'RETIRED': Retired / DNF during the race (mechanical failure, accident, collision, damage, etc.)
      - 'DNS': Did not start / withdrew prior to start ('Did not start', 'DNS', 'Withdrew', 'Withdrawn')
      - 'DSQ': Disqualified / excluded ('Disqualified', 'DSQ', 'Excluded')
      - 'NC': Not classified ('Not classified', 'NC')
      - 'UNKNOWN': Unrecognized or missing status
    """
    if pd.isna(status):
        return "UNKNOWN"
    text = str(status).strip()
    if not text:
        return "UNKNOWN"
    t_lower = text.lower()

    # Finished / running statuses (e.g. 'Finished', '+1 Lap', '+2 Laps', 'Lapped')
    if t_lower == "finished" or "lap" in t_lower:
        return "FINISHED"

    # DNS / Withdrew
    if any(k in t_lower for k in ["did not start", "dns", "withdrew", "withdrawn"]):
        return "DNS"

    # DSQ / Excluded
    if any(k in t_lower for k in ["disqualified", "dsq", "excluded"]):
        return "DSQ"

    # NC (Not classified)
    if t_lower in ["nc", "not classified"]:
        return "NC"

    # In FastF1/Ergast, specific terminal mechanical/accident failure strings represent retirements:
    retirement_terms = [
        "retired", "collision", "accident", "engine", "gearbox",
        "hydraulics", "electrical", "brakes", "suspension",
        "power unit", "puncture", "water pressure", "oil pressure",
        "wheel", "damage", "spins off", "dnf", "turbo", "vibrations",
        "overheating", "mechanical", "technical", "out of fuel", "clutch",
        "exhaust", "throttle", "steering", "radiator", "battery"
    ]
    if any(term in t_lower for term in retirement_terms):
        return "RETIRED"

    return "UNKNOWN"


def is_classified_finish(status: Any) -> bool:
    """Classifies finishing status string into classified/running finish (True) or non-finish (False)."""
    return classify_race_status(status) == "FINISHED"


def is_retired(status: Any) -> bool:
    """Classifies finishing status string into in-race mechanical/incident retirement (True) or False."""
    return classify_race_status(status) == "RETIRED"


def is_dns(status: Any) -> bool:
    """Classifies finishing status string into did-not-start / withdrawn before start (True) or False."""
    return classify_race_status(status) == "DNS"


def is_dsq(status: Any) -> bool:
    """Classifies finishing status string into disqualified / excluded (True) or False."""
    return classify_race_status(status) == "DSQ"


def is_nc(status: Any) -> bool:
    """Classifies finishing status string into not-classified finish (True) or False."""
    return classify_race_status(status) == "NC"


def is_dnf(
    status: Any,
    include_dns: bool = False,
    include_dsq: bool = False,
    include_nc: bool = False
) -> bool:
    """
    Classifies race finishing status into DNF (mechanical/incident retirement).

    By default:
      - RETIRED (engine, accident, collision, gearbox, etc.): True
      - FINISHED ('Finished', '+1 Lap', etc.): False
      - DNS ('Did not start', 'DNS'): False (tracked separately; True if include_dns=True)
      - DSQ ('Disqualified', 'DSQ'): False (tracked separately; True if include_dsq=True)
      - NC ('Not classified', 'NC'): False (tracked separately; True if include_nc=True)
      - UNKNOWN (missing, NaN, unparsed): False (never counted as DNF)

    Parameters
    ----------
    status : Any
        Finishing status string or value from race results.
    include_dns : bool, optional
        Whether to count Did Not Start as DNF (default False).
    include_dsq : bool, optional
        Whether to count Disqualification as DNF (default False).
    include_nc : bool, optional
        Whether to count Not Classified as DNF (default False).

    Returns
    -------
    bool
        True if the status is classified as a retirement/DNF, False otherwise.
    """
    if pd.isna(status):
        return False
    cat = classify_race_status(status)
    if cat == "RETIRED":
        return True
    if cat == "DNS" and include_dns:
        return True
    if cat == "DSQ" and include_dsq:
        return True
    if cat == "NC" and include_nc:
        return True
    return False


def ewma(values: list, alpha: float = 0.3) -> float:
    """Exponentially weighted moving average with newest values having highest weight."""
    s = pd.Series(values).dropna()
    if s.empty:
        return np.nan
    return float(s.ewm(alpha=alpha, adjust=False).mean().iloc[-1])


def compute_season_driver_stats(
    prior_races_df: pd.DataFrame,
    prior_qualis_df: Optional[pd.DataFrame] = None,
    alpha: float = 0.3,
    default_pos: float = 12.0
) -> Dict[str, Dict[str, Any]]:
    """
    Computes rolling driver form and status-specific outcome rates up to target round.

    Status Metric Definitions & Statistical Denominators:
      - total_races (races_completed): Total scheduled rounds entered by driver (len(grp)).
      - valid_entries: Total rounds with known, valid classification (total_races - n_unknown).
        UNKNOWN statuses are excluded from valid outcome denominators to avoid dilution bias.
      - valid_starts: Total rounds where the driver took the start (valid_entries - n_dns).
      - driver_dns_rate: Non-start rate per valid race entry (n_dns / valid_entries).
      - driver_dnf_rate: Mechanical/accident retirement rate per actual race start (n_retired / valid_starts).
      - driver_dsq_rate: Disqualification rate per valid race entry (n_dsq / valid_entries).
      - driver_nc_rate: Unclassified finish rate (<90% distance) per actual race start (n_nc / valid_starts).

    Parameters
    ----------
    prior_races_df : pd.DataFrame
        Historical races strictly before target round (must contain 'Abbreviation', 'Position', 'GridPosition').
    prior_qualis_df : pd.DataFrame, optional
        Historical qualifying strictly before target round.
    alpha : float, optional
        EWMA decay parameter (default 0.3).
    default_pos : float, optional
        Default position for drivers with missing data (default 12.0).

    Returns
    -------
    dict
        Mapping of driver abbreviation to dictionary of performance and outcome metrics.
    """
    driver_stats: Dict[str, Dict[str, Any]] = {}
    if prior_races_df is None or prior_races_df.empty:
        return driver_stats

    prior = prior_races_df.copy()
    prior_q = (
        prior_qualis_df.copy()
        if (prior_qualis_df is not None and not prior_qualis_df.empty)
        else pd.DataFrame()
    )

    has_status = "Status" in prior.columns

    for drv, grp in prior.groupby("Abbreviation"):
        grp = grp.sort_values("Round") if "Round" in grp.columns else grp
        qgrp = (
            prior_q[prior_q["Abbreviation"] == drv].sort_values("Round")
            if (not prior_q.empty and "Round" in prior_q.columns and "Abbreviation" in prior_q.columns)
            else pd.DataFrame()
        )

        finishes = pd.to_numeric(grp["Position"], errors="coerce").dropna().tolist()
        grids = pd.to_numeric(grp["GridPosition"], errors="coerce").dropna().tolist()
        points = (
            pd.to_numeric(grp["Points"], errors="coerce").fillna(0).tolist()
            if "Points" in grp.columns
            else [0.0] * len(grp)
        )
        qvals = (
            pd.to_numeric(qgrp["Position"], errors="coerce").dropna().tolist()
            if not qgrp.empty and "Position" in qgrp.columns
            else []
        )

        total_races = len(grp)

        if has_status:
            statuses = grp["Status"].apply(classify_race_status)
            n_retired = int((statuses == "RETIRED").sum())
            n_dns = int((statuses == "DNS").sum())
            n_dsq = int((statuses == "DSQ").sum())
            n_nc = int((statuses == "NC").sum())
            n_unknown = int((statuses == "UNKNOWN").sum())

            # Valid outcome denominators excluding unknown/missing records
            valid_entries = total_races - n_unknown
            valid_starts = max(valid_entries - n_dns, 0)

            # Statistical rates per appropriate population at risk
            dns_rate = float(n_dns / valid_entries) if valid_entries > 0 else 0.0
            dnf_rate = float(n_retired / valid_starts) if valid_starts > 0 else 0.0
            dsq_rate = float(n_dsq / valid_entries) if valid_entries > 0 else 0.0
            nc_rate = float(n_nc / valid_starts) if valid_starts > 0 else 0.0
        elif "is_dnf" in grp.columns:
            n_retired = int(grp["is_dnf"].sum())
            n_dns = 0
            n_dsq = 0
            n_nc = 0
            dnf_rate = float(n_retired / total_races) if total_races > 0 else 0.0
            dns_rate = 0.0
            dsq_rate = 0.0
            nc_rate = 0.0
        else:
            dnf_rate = 0.0
            dns_rate = 0.0
            dsq_rate = 0.0
            nc_rate = 0.0

        driver_stats[str(drv)] = {
            "avg_finish": float(np.mean(finishes)) if finishes else default_pos,
            "avg_grid": float(np.mean(grids)) if grids else default_pos,
            "total_points": float(sum(points)),
            "races_completed": int(total_races),
            "recent_finish": float(finishes[-1]) if finishes else default_pos,
            "recent_grid": float(grids[-1]) if grids else default_pos,
            "recent_points": float(points[-1]) if points else 0.0,
            "driver_finish_ewma": float(ewma(finishes, alpha=alpha)) if finishes else default_pos,
            "driver_quali_ewma": float(ewma(qvals, alpha=alpha)) if qvals else (
                float(ewma(grids, alpha=alpha)) if grids else default_pos
            ),
            "driver_dnf_rate": dnf_rate,
            "driver_dns_rate": dns_rate,
            "driver_dsq_rate": dsq_rate,
            "driver_nc_rate": nc_rate,
        }

    return driver_stats


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

    driver_stats = compute_season_driver_stats(
        prior_races_df=prior,
        prior_qualis_df=prior_q,
        alpha=alpha,
        default_pos=default_pos
    )

    team_stats: Dict[str, Dict[str, Any]] = {}
    if not prior.empty:
        team_col = "TeamName" if "TeamName" in prior.columns else "Team"
        if team_col in prior.columns:
            for team, grp in prior.groupby(team_col):
                grp = grp.sort_values("Round")
                finishes = pd.to_numeric(grp["Position"], errors="coerce").dropna().tolist()
                points = (
            pd.to_numeric(grp["Points"], errors="coerce").fillna(0).tolist()
            if "Points" in grp.columns
            else [0.0] * len(grp)
        )
                rounds = grp["Round"].nunique()

                team_stats[str(team)] = {
                    "team_avg_finish": float(np.mean(finishes)) if finishes else default_pos,
                    "team_finish_ewma": float(ewma(finishes, alpha=alpha)) if finishes else default_pos,
                    "team_points": float(sum(points)),
                    "team_points_rate": float(sum(points) / max(rounds * 2, 1)),
                }

    return driver_stats, team_stats
