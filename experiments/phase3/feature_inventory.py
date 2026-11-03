"""
Phase 3 Feature Inventory and Taxonomy Generator.
Catalogs all 37 baseline and 25 enhanced features, categorizing by functional group,
temporal origin, missingness, stage availability, and leakage risk.
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from experiments.model_benchmark.config import (
    ROOT_DIR,
    BASELINE_FEATURES,
    BENCHMARK_STAGE,
)
from src.features.build_features import STAGE_UNAVAILABLE_FEATURES

FEATURE_GROUPS = {
    # 1. Starting Grid Features
    "GridPosition": "Starting Grid",
    "grid_norm": "Starting Grid",
    "expected_grid_finish": "Starting Grid",
    "grid_retention_prob": "Starting Grid",

    # 2. Qualifying Performance
    "QualifyingPosition": "Qualifying Performance",
    "quali_pos_norm": "Qualifying Performance",
    "quali_gap_pct": "Qualifying Performance",
    "quali_no_time": "Qualifying Performance",
    "q1_gap_pct": "Qualifying Performance",
    "q2_gap_pct": "Qualifying Performance",
    "q3_gap_pct": "Qualifying Performance",

    # 3. Practice Pace
    "practice_gap_pct": "Practice Pace",
    "practice_rank": "Practice Pace",
    "practice_laps_norm": "Practice Pace",
    "fp1_gap_pct": "Practice Pace",
    "fp2_gap_pct": "Practice Pace",
    "fp3_gap_pct": "Practice Pace",

    # 4. Driver Performance & Form
    "driver_finish_ewma": "Driver Performance",
    "driver_quali_ewma": "Driver Performance",
    "driver_dnf_rate": "Driver Performance",
    "recent_finish": "Driver Performance",
    "recent_grid": "Driver Performance",
    "recent_points": "Driver Performance",
    "avg_finish": "Driver Performance",
    "avg_grid": "Driver Performance",
    "total_points": "Driver Performance",
    "races_completed": "Driver Performance",

    # 5. Constructor / Team Form
    "team_finish_ewma": "Constructor Form",
    "team_points_rate": "Constructor Form",
    "team_avg_finish": "Constructor Form",
    "team_points": "Constructor Form",

    # 6. Sprint Performance
    "sprint_quali_pos": "Sprint Performance",
    "sprint_finish_pos": "Sprint Performance",
    "sprint_pos_delta": "Sprint Performance",

    # 7. Teammate Comparison
    "teammate_quali_gap_pct": "Teammate Comparison",
    "teammate_grid_delta": "Teammate Comparison",
    "driver_teammate_h2h_ratio": "Teammate Comparison",

    # 8. Enhanced: Race Pace Intelligence
    "long_run_pace_gap_pct": "Race Pace Intelligence",
    "long_run_pace_rank": "Race Pace Intelligence",
    "race_pace_consistency": "Race Pace Intelligence",
    "race_pace_sample_count": "Race Pace Intelligence",
    "long_run_stint_length": "Race Pace Intelligence",
    "race_pace_vs_quali_delta": "Race Pace Intelligence",
    "race_pace_confidence": "Race Pace Intelligence",

    # 9. Enhanced: Tire Degradation
    "soft_tyre_deg_rate": "Tire Degradation",
    "medium_tyre_deg_rate": "Tire Degradation",
    "hard_tyre_deg_rate": "Tire Degradation",
    "estimated_tyre_deg_rate": "Tire Degradation",
    "compound_pace_delta": "Tire Degradation",
    "tyre_deg_confidence": "Tire Degradation",

    # 10. Enhanced: Pit Strategy
    "optimal_strategy_stops": "Pit Strategy",
    "strategy_time_loss_sec": "Pit Strategy",

    # 11. Enhanced: Race Control Hazard
    "p_safety_car": "Race Control",
    "p_vsc": "Race Control",
    "p_red_flag": "Race Control",

    # 12. Enhanced: Weather & Track Conditions
    "air_temperature": "Weather & Conditions",
    "track_temperature": "Weather & Conditions",
    "humidity": "Weather & Conditions",
    "wind_speed": "Weather & Conditions",
    "rain_probability": "Weather & Conditions",
    "wet_conditions_indicator": "Weather & Conditions",
    "weather_confidence": "Weather & Conditions",
}

ENHANCED_FEATURES = [
    "long_run_pace_gap_pct", "long_run_pace_rank", "race_pace_consistency",
    "race_pace_sample_count", "long_run_stint_length", "race_pace_vs_quali_delta",
    "race_pace_confidence", "soft_tyre_deg_rate", "medium_tyre_deg_rate",
    "hard_tyre_deg_rate", "estimated_tyre_deg_rate", "compound_pace_delta",
    "tyre_deg_confidence", "optimal_strategy_stops", "strategy_time_loss_sec",
    "p_safety_car", "p_vsc", "p_red_flag", "air_temperature", "track_temperature",
    "humidity", "wind_speed", "rain_probability", "wet_conditions_indicator",
    "weather_confidence"
]


def build_feature_inventory(
    sample_df_2026: pd.DataFrame,
    sample_df_hist: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes empirical missingness, variance, temporal origin, and leakage risk for all features.
    """
    all_features = BASELINE_FEATURES + ENHANCED_FEATURES
    rows = []

    for feat in all_features:
        group = FEATURE_GROUPS.get(feat, "Other")
        status = "Active Baseline" if feat in BASELINE_FEATURES else "Enhanced Capability"

        # Missingness in 2026
        if feat in sample_df_2026.columns:
            miss_2026 = round(float(sample_df_2026[feat].isna().mean()) * 100.0, 1)
            vals = sample_df_2026[feat].dropna()
            is_const = bool(vals.nunique() <= 1) if not vals.empty else True
            dtype = str(sample_df_2026[feat].dtype)
        else:
            miss_2026 = 100.0
            is_const = True
            dtype = "float64"

        # Missingness in historical tracks
        if feat in sample_df_hist.columns:
            miss_hist = round(float(sample_df_hist[feat].isna().mean()) * 100.0, 1)
        else:
            miss_hist = 100.0

        # Temporal Origin
        if group in ["Driver Performance", "Constructor Form", "Race Control"]:
            origin = "Historical Trajectory (< r)"
            leakage_risk = "Low (Strictly filtered Round < r)"
        elif group in ["Starting Grid", "Qualifying Performance", "Practice Pace", "Sprint Performance", "Teammate Comparison"]:
            origin = "Target Weekend Sessions"
            leakage_risk = "High if unmasked; Zero under stage mask"
        elif group in ["Tire Degradation", "Race Pace Intelligence", "Weather & Conditions", "Pit Strategy"]:
            origin = "Weekend Telemetry & Track Priors"
            leakage_risk = "Medium (Requires strict FP-only boundary)"
        else:
            origin = "Engineered"
            leakage_risk = "Low"

        # Stage Availability
        unavail = [stg for stg, cols in STAGE_UNAVAILABLE_FEATURES.items() if feat in cols]
        if not unavail:
            stage_avail = "All Stages"
        elif "after_qualifying" not in unavail:
            stage_avail = "Available at after_qualifying"
        else:
            stage_avail = f"Unavailable until {', '.join(unavail)}"

        source = (
            "Official Starting Grid" if group == "Starting Grid" else
            "Qualifying Classification" if group == "Qualifying Performance" else
            "FP1/FP2/FP3 FastF1 Telemetry" if group == "Practice Pace" else
            "Season Historical CSVs" if group in ["Driver Performance", "Constructor Form"] else
            "Sprint Shootout & Race" if group == "Sprint Performance" else
            "Intra-Team Delta Math" if group == "Teammate Comparison" else
            "Physics & Weather Engines"
        )

        rows.append({
            "feature_name": feat,
            "feature_group": group,
            "production_status": status,
            "data_type": dtype,
            "source": source,
            "prediction_stage_availability": stage_avail,
            "missing_pct_2026": miss_2026,
            "missing_pct_historical": miss_hist,
            "is_constant_2026": is_const,
            "temporal_origin": origin,
            "leakage_risk": leakage_risk
        })

    return pd.DataFrame(rows)
