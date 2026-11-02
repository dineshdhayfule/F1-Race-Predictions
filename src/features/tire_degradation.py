"""
Tire Degradation Modeling Module
================================
Extracts compound-specific lap-time degradation rates from practice and historical stint telemetry.
Provides robust slope estimation, teammate borrowing for small samples, temperature sensitivity,
and circuit-level prior fallbacks.
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd

from src.utils.driver_mapping import normalize_driver, normalize_driver_series
from src.utils.timing import timedelta_to_seconds
from src.utils.logging_utils import get_logger

logger = get_logger("tire_degradation")

# Default config path
DEFAULT_STRATEGY_CFG_PATH = Path(__file__).resolve().parents[2] / "config" / "strategy_config.json"


def load_strategy_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads strategy and tyre configuration file."""
    p = config_path or DEFAULT_STRATEGY_CFG_PATH
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def estimate_stint_degradation_slope(
    laps_series: np.ndarray,
    tyre_age_series: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Estimates the degradation slope (seconds gained per lap of tyre age)
    using robust linear regression with outlier clipping.

    Returns
    -------
    tuple of (slope, r_squared)
    """
    if len(laps_series) < 3:
        return 0.05, 0.0

    n = len(laps_series)
    x = tyre_age_series if tyre_age_series is not None and len(tyre_age_series) == n else np.arange(n)
    y = np.array(laps_series, dtype=float)

    # Filter any non-finite entries
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return 0.05, 0.0
    x, y = x[valid], y[valid]

    # Robust regression: clip extreme outlier residuals
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.RankWarning if hasattr(np, "exceptions") else UserWarning)
        p_initial = np.polyfit(x, y, 1)
        res = y - (p_initial[0] * x + p_initial[1])
        mad = np.median(np.abs(res - np.median(res)))
        threshold = max(2.5 * 1.4826 * mad, 0.5)

        clean_mask = np.abs(res) <= threshold
        if clean_mask.sum() >= 3:
            p_clean = np.polyfit(x[clean_mask], y[clean_mask], 1)
            slope = float(p_clean[0])
            # Compute R-squared on clean subset
            y_hat = p_clean[0] * x[clean_mask] + p_clean[1]
            ss_res = np.sum((y[clean_mask] - y_hat) ** 2)
            ss_tot = np.sum((y[clean_mask] - np.mean(y[clean_mask])) ** 2)
            r2 = float(1.0 - (ss_res / (ss_tot + 1e-8))) if ss_tot > 1e-6 else 0.0
        else:
            slope = float(p_initial[0])
            r2 = 0.0

    # Ensure realistic non-negative degradation bounds [0.005, 0.40] seconds/lap
    slope = float(np.clip(slope, 0.005, 0.40))
    r2 = float(np.clip(r2, 0.0, 1.0))
    return round(slope, 4), round(r2, 3)


class TireDegradationModel:
    """Models compound-specific tyre degradation across drivers, teams, and circuits."""

    def __init__(self, config_path: Optional[Path] = None):
        self.cfg = load_strategy_config(config_path)
        self.compounds_cfg = self.cfg.get("compounds", {})
        self.circuit_priors = self.cfg.get("circuit_priors", {})

    def get_circuit_prior(self, circuit_key: Optional[str]) -> Dict[str, Any]:
        """Retrieves circuit-specific baseline tyre degradation multiplier."""
        ck = str(circuit_key).lower().strip() if circuit_key else "default"
        return self.circuit_priors.get(ck, self.circuit_priors.get("default", {
            "deg_multiplier": 1.0,
            "pit_loss_seconds": 22.0
        }))

    def extract_driver_compound_degradation(
        self,
        laps_df: pd.DataFrame,
        drivers: List[str],
        circuit_key: Optional[str] = None,
        track_temperature: Optional[float] = None,
        team_map: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Extracts driver and compound-specific degradation rates from valid practice stints.
        Applies teammate pooling for small samples and circuit prior fallbacks when telemetry is missing.

        Parameters
        ----------
        laps_df : pd.DataFrame
            Session lap telemetry.
        drivers : list of str
            List of driver abbreviations.
        circuit_key : str, optional
            Circuit key (e.g. 'mexico', 'austin', 'sepang').
        track_temperature : float, optional
            Track temperature in Celsius.
        team_map : dict, optional
            Mapping of driver -> team name for teammate pooling.

        Returns
        -------
        pd.DataFrame
            Features:
            - soft_tyre_deg_rate
            - medium_tyre_deg_rate
            - hard_tyre_deg_rate
            - estimated_tyre_deg_rate
            - compound_pace_delta
            - tyre_deg_confidence
        """
        circuit_info = self.get_circuit_prior(circuit_key)
        deg_mult = circuit_info.get("deg_multiplier", 1.0)

        # Temperature sensitivity adjustment (+1.5% deg per degree above 30 C track temp)
        temp_scale = 1.0
        if track_temperature is not None and pd.notna(track_temperature):
            delta_t = float(track_temperature) - 30.0
            temp_scale = 1.0 + (delta_t * 0.015)
            temp_scale = float(np.clip(temp_scale, 0.70, 1.45))

        # Base prior rates per compound
        base_soft = self.compounds_cfg.get("SOFT", {}).get("base_deg_rate_sec_per_lap", 0.110) * deg_mult * temp_scale
        base_med = self.compounds_cfg.get("MEDIUM", {}).get("base_deg_rate_sec_per_lap", 0.065) * deg_mult * temp_scale
        base_hard = self.compounds_cfg.get("HARD", {}).get("base_deg_rate_sec_per_lap", 0.038) * deg_mult * temp_scale

        results = []

        if laps_df is None or laps_df.empty:
            # Complete prior fallback
            for drv in drivers:
                results.append({
                    "Abbreviation": drv,
                    "soft_tyre_deg_rate": round(base_soft, 4),
                    "medium_tyre_deg_rate": round(base_med, 4),
                    "hard_tyre_deg_rate": round(base_hard, 4),
                    "estimated_tyre_deg_rate": round(base_med, 4),
                    "compound_pace_delta": 0.65,
                    "tyre_deg_confidence": 0.20,
                })
            return pd.DataFrame(results)

        df = laps_df.copy()
        driver_col = "Abbreviation" if "Abbreviation" in df.columns else ("Driver" if "Driver" in df.columns else None)
        if driver_col:
            df["Abbreviation"] = normalize_driver_series(df[driver_col])

        if "LapTimeSeconds" not in df.columns and "LapTime" in df.columns:
            df["LapTimeSeconds"] = df["LapTime"].apply(timedelta_to_seconds)

        df = df[pd.to_numeric(df.get("LapTimeSeconds"), errors="coerce") > 50.0].copy()

        # Build driver stint degradation measurements
        driver_comp_deg: Dict[str, Dict[str, List[Tuple[float, int]]]] = {
            d: {"SOFT": [], "MEDIUM": [], "HARD": []} for d in drivers
        }

        if not df.empty and "Stint" in df.columns and "Compound" in df.columns:
            for (drv, stint_no, comp), group in df.groupby(["Abbreviation", "Stint", "Compound"]):
                if drv not in driver_comp_deg:
                    continue
                comp_clean = str(comp).upper().strip()
                if comp_clean not in ["SOFT", "MEDIUM", "HARD"]:
                    continue

                times = group["LapTimeSeconds"].dropna().values
                if len(times) >= 4:
                    tyre_age = group["TyreLife"].values if "TyreLife" in group.columns else None
                    slope, r2 = estimate_stint_degradation_slope(times, tyre_age)
                    driver_comp_deg[drv][comp_clean].append((slope, len(times)))

        # Also aggregate team-level degradation for teammate pooling
        team_comp_deg: Dict[str, Dict[str, List[Tuple[float, int]]]] = {}
        if team_map:
            for drv, comp_dict in driver_comp_deg.items():
                team = team_map.get(drv, "Unknown")
                if team not in team_comp_deg:
                    team_comp_deg[team] = {"SOFT": [], "MEDIUM": [], "HARD": []}
                for c, runs in comp_dict.items():
                    team_comp_deg[team][c].extend(runs)

        for drv in drivers:
            team = team_map.get(drv, "Unknown") if team_map else "Unknown"
            comp_rates = {}
            total_stint_laps = 0

            for comp, base_prior in [("SOFT", base_soft), ("MEDIUM", base_med), ("HARD", base_hard)]:
                runs = driver_comp_deg[drv][comp]
                if runs:
                    # Weighted average across driver's stints
                    driver_laps = sum(r[1] for r in runs)
                    driver_rate = sum(r[0] * r[1] for r in runs) / driver_laps
                    # Blend with prior (shrinkage towards circuit prior for small sample)
                    weight_data = min(driver_laps / 12.0, 0.85)
                    final_rate = weight_data * driver_rate + (1.0 - weight_data) * base_prior
                    total_stint_laps += driver_laps
                elif team in team_comp_deg and team_comp_deg[team][comp]:
                    # Borrow teammate rate
                    t_runs = team_comp_deg[team][comp]
                    t_laps = sum(r[1] for r in t_runs)
                    t_rate = sum(r[0] * r[1] for r in t_runs) / t_laps
                    final_rate = 0.50 * t_rate + 0.50 * base_prior
                    total_stint_laps += int(t_laps * 0.5)
                else:
                    final_rate = base_prior

                comp_rates[comp] = round(float(final_rate), 4)

            # Confidence based on total observed long-run laps
            confidence = round(float(np.clip(total_stint_laps / 18.0, 0.20, 1.0)), 2)

            # Overall estimated degradation rate is weighted average across standard race compounds
            overall_deg = (
                0.20 * comp_rates["SOFT"] + 0.50 * comp_rates["MEDIUM"] + 0.30 * comp_rates["HARD"]
            )

            # Measured compound pace delta between Soft and Medium
            pace_delta = round(float(comp_rates["SOFT"] - comp_rates["MEDIUM"] + 0.65), 3)

            results.append({
                "Abbreviation": drv,
                "soft_tyre_deg_rate": comp_rates["SOFT"],
                "medium_tyre_deg_rate": comp_rates["MEDIUM"],
                "hard_tyre_deg_rate": comp_rates["HARD"],
                "estimated_tyre_deg_rate": round(overall_deg, 4),
                "compound_pace_delta": pace_delta,
                "tyre_deg_confidence": confidence,
            })

        return pd.DataFrame(results)


def build_tire_degradation_features(
    session_dfs: Dict[str, Optional[pd.DataFrame]],
    drivers: List[str],
    circuit_key: Optional[str] = None,
    track_temperature: Optional[float] = None,
    team_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Unified entry point for tyre degradation feature calculation.
    """
    model = TireDegradationModel()
    all_laps_list = []
    for s_name in ("fp2", "fp1", "fp3"):
        df = session_dfs.get(s_name)
        if df is not None and not df.empty and ("LapTime" in df.columns or "LapTimeSeconds" in df.columns):
            all_laps_list.append(df)

    pooled_laps = pd.concat(all_laps_list, ignore_index=True) if all_laps_list else pd.DataFrame()

    return model.extract_driver_compound_degradation(
        laps_df=pooled_laps,
        drivers=drivers,
        circuit_key=circuit_key,
        track_temperature=track_temperature,
        team_map=team_map,
    )

