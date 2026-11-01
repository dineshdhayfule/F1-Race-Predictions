"""
Unified Feature Engineering Pipeline
====================================
Orchestrates training and target feature matrix construction for any F1 circuit.
Enforces zero data leakage: target race result is strictly excluded from target features.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

from src.features.grid_features import apply_grid_penalties, resolve_fia_starting_grid
from src.features.grid_conversion import compute_grid_conversion_map, get_conversion_features_for_grid
from src.features.season_features import build_season_form
from src.features.weekend_features import (
    build_practice_features,
    build_qualifying_features,
    build_sprint_qualifying_features,
    build_sprint_race_features,
)
from src.features.teammate_features import (
    compute_teammate_quali_gap,
    compute_teammate_grid_delta,
    compute_rolling_teammate_h2h,
)
from src.utils.driver_mapping import normalize_driver_series
from src.utils.logging_utils import get_logger

logger = get_logger("build_features")

STAGE_UNAVAILABLE_FEATURES: Dict[str, List[str]] = {
    "pre_fp1": [
        # Practice
        "fp1_best_lap", "fp2_best_lap", "fp3_best_lap", "best_practice_lap",
        "avg_practice_lap", "practice_gap_pct", "practice_rank", "practice_laps",
        "practice_laps_norm", "practice_sessions", "fp1_gap_pct", "fp2_gap_pct", "fp3_gap_pct",
        # Sprint
        "sprint_quali_pos", "sprint_finish_pos", "sprint_pos_delta",
        # Qualifying
        "QualifyingPosition", "quali_pos_norm", "quali_gap_pct", "quali_no_time",
        "q1_gap_pct", "q2_gap_pct", "q3_gap_pct", "teammate_quali_gap_pct",
        # Grid
        "GridPosition", "ActualStartingGrid", "simulation_start_rank", "is_pit_lane_start",
        "StartingGridStatus", "GridPenalty", "grid_penalty_places", "grid_vs_quali_delta",
        "grid_norm", "expected_grid_finish", "grid_retention_prob", "teammate_grid_delta",
    ],
    "after_fp1": [
        # FP2, FP3
        "fp2_best_lap", "fp3_best_lap", "fp2_gap_pct", "fp3_gap_pct",
        # Sprint
        "sprint_quali_pos", "sprint_finish_pos", "sprint_pos_delta",
        # Qualifying
        "QualifyingPosition", "quali_pos_norm", "quali_gap_pct", "quali_no_time",
        "q1_gap_pct", "q2_gap_pct", "q3_gap_pct", "teammate_quali_gap_pct",
        # Grid
        "GridPosition", "ActualStartingGrid", "simulation_start_rank", "is_pit_lane_start",
        "StartingGridStatus", "GridPenalty", "grid_penalty_places", "grid_vs_quali_delta",
        "grid_norm", "expected_grid_finish", "grid_retention_prob", "teammate_grid_delta",
    ],
    "after_fp2": [
        # FP3
        "fp3_best_lap", "fp3_gap_pct",
        # Sprint
        "sprint_quali_pos", "sprint_finish_pos", "sprint_pos_delta",
        # Qualifying
        "QualifyingPosition", "quali_pos_norm", "quali_gap_pct", "quali_no_time",
        "q1_gap_pct", "q2_gap_pct", "q3_gap_pct", "teammate_quali_gap_pct",
        # Grid
        "GridPosition", "ActualStartingGrid", "simulation_start_rank", "is_pit_lane_start",
        "StartingGridStatus", "GridPenalty", "grid_penalty_places", "grid_vs_quali_delta",
        "grid_norm", "expected_grid_finish", "grid_retention_prob", "teammate_grid_delta",
    ],
    "after_fp3": [
        # Sprint
        "sprint_quali_pos", "sprint_finish_pos", "sprint_pos_delta",
        # Qualifying
        "QualifyingPosition", "quali_pos_norm", "quali_gap_pct", "quali_no_time",
        "q1_gap_pct", "q2_gap_pct", "q3_gap_pct", "teammate_quali_gap_pct",
        # Grid
        "GridPosition", "ActualStartingGrid", "simulation_start_rank", "is_pit_lane_start",
        "StartingGridStatus", "GridPenalty", "grid_penalty_places", "grid_vs_quali_delta",
        "grid_norm", "expected_grid_finish", "grid_retention_prob", "teammate_grid_delta",
    ],
    "after_qualifying": [],
    "before_race": [],
    "post_race": [],
}


def apply_stage_feature_mask(df: pd.DataFrame, stage: Optional[str]) -> pd.DataFrame:
    """
    Masks features that are not yet available at the given prediction stage cutoff.
    Prevents temporal data leakage into early-stage models.
    """
    if df is None or df.empty or not stage:
        return df

    stage_clean = str(stage).lower().strip()
    masked_cols = STAGE_UNAVAILABLE_FEATURES.get(stage_clean, [])
    for col in masked_cols:
        if col in df.columns:
            df[col] = np.nan
    return df


class FeatureEngineer:
    """Unified feature engineering engine for Formula 1 race prediction."""

    def __init__(self, ewma_alpha: float = 0.3, expected_grid_size: int = 22):
        self.ewma_alpha = ewma_alpha
        self.expected_grid_size = expected_grid_size

    def build_training_dataset(
        self,
        races_df: pd.DataFrame,
        qualis_df: pd.DataFrame,
        practice_sessions_by_round: Dict[int, Dict[str, pd.DataFrame]],
        train_rounds: List[int],
        stage: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Builds the training dataset across completed historical rounds.
        For each round r in train_rounds:
          - Uses only rounds < r for rolling driver/team form.
          - Uses round r's practice and qualifying sessions.
          - Target is round r's ActualFinish.

        Parameters
        ----------
        races_df : pd.DataFrame
            All completed season race results.
        qualis_df : pd.DataFrame
            All completed season qualifying results.
        practice_sessions_by_round : dict
            Mapping of round_number -> {'fp1': df, 'fp2': df, 'fp3': df}.
        train_rounds : list of int
            List of completed round numbers to include in training.

        Returns
        -------
        pd.DataFrame
            Complete training feature matrix with 'ActualFinish'.
        """
        rows = []
        logger.info(f"Building training features across {len(train_rounds)} rounds: {train_rounds}")

        for r in train_rounds:
            race = races_df[races_df["Round"] == r].copy()
            if race.empty:
                logger.warning(f"Round {r} has no race data; skipping.")
                continue

            race["Abbreviation"] = normalize_driver_series(race["Abbreviation"])
            drivers = race["Abbreviation"].dropna().unique().tolist()

            weekend_sessions = practice_sessions_by_round.get(r, {})
            practice = build_practice_features(weekend_sessions, drivers)

            round_quali = qualis_df[qualis_df["Round"] == r].copy() if not qualis_df.empty else None
            quali = build_qualifying_features(round_quali, drivers)

            # Sprint features (if sprint sessions were held and available prior to cutoff)
            sq_df = weekend_sessions.get("sprint_qualifying") if weekend_sessions else None
            sr_df = weekend_sessions.get("sprint_race") if weekend_sessions else None
            sq_feats = build_sprint_qualifying_features(sq_df, drivers)
            sr_feats = build_sprint_race_features(sr_df, drivers)

            # Strict chronological cutoff: form and grid conversion computed ONLY on rounds < r
            dform, tform = build_season_form(r, races_df, qualis_df, alpha=self.ewma_alpha)
            prior_races = races_df[races_df["Round"] < r]
            grid_conv_map = compute_grid_conversion_map(prior_races, expected_grid_size=self.expected_grid_size)

            # Teammate features
            tm_quali_feats = compute_teammate_quali_gap(round_quali, drivers)
            tm_grid_feats = compute_teammate_grid_delta(race, drivers, grid_col="GridPosition")
            tm_h2h_map = compute_rolling_teammate_h2h(prior_races, race[["Abbreviation", "TeamName"]])

            for _, rr in race.iterrows():
                drv = rr["Abbreviation"]
                team = str(rr.get("TeamName", rr.get("Team", "Unknown")))

                p = practice[practice["Abbreviation"] == drv]
                q = quali[quali["Abbreviation"] == drv]
                sq_row = sq_feats[sq_feats["Abbreviation"] == drv]
                sr_row = sr_feats[sr_feats["Abbreviation"] == drv]

                p_dict = p.iloc[0].to_dict() if not p.empty else {}
                q_dict = q.iloc[0].to_dict() if not q.empty else {}
                sq_dict = sq_row.iloc[0].to_dict() if not sq_row.empty else {}
                sr_dict = sr_row.iloc[0].to_dict() if not sr_row.empty else {}

                tm_q_row = tm_quali_feats[tm_quali_feats["Abbreviation"] == drv]
                tm_g_row = tm_grid_feats[tm_grid_feats["Abbreviation"] == drv]
                tm_q_gap = tm_q_row["teammate_quali_gap_pct"].iloc[0] if not tm_q_row.empty else np.nan
                tm_g_delta = tm_g_row["teammate_grid_delta"].iloc[0] if not tm_g_row.empty else np.nan
                tm_h2h = tm_h2h_map.get(drv, 0.50)

                raw_grid = pd.to_numeric(rr.get("GridPosition"), errors="coerce")
                is_pit = 1.0 if (pd.notna(raw_grid) and raw_grid <= 0) or bool(rr.get("is_pit_lane", False)) or bool(rr.get("is_pit_lane_start", False)) else 0.0

                if is_pit == 1.0:
                    actual_grid = np.nan
                    sim_rank = float(self.expected_grid_size)
                    effective_grid = float(self.expected_grid_size)
                    grid_norm = 1.0
                elif pd.isna(raw_grid):
                    actual_grid = 12.0
                    sim_rank = 12.0
                    effective_grid = 12.0
                    grid_norm = (12.0 - 1.0) / float(self.expected_grid_size - 1)
                else:
                    actual_grid = float(raw_grid)
                    sim_rank = float(raw_grid)
                    effective_grid = float(raw_grid)
                    grid_norm = (float(raw_grid) - 1.0) / float(self.expected_grid_size - 1)

                dfv = dform.get(drv, {})
                tfv = tform.get(team, {})

                conv_feat = get_conversion_features_for_grid(effective_grid, grid_conv_map, expected_grid_size=self.expected_grid_size)

                finish = pd.to_numeric(rr.get("Position"), errors="coerce")
                q_pos_val = float(q_dict.get("QualifyingPosition", effective_grid))

                rows.append({
                    "Round": r,
                    "RaceName": str(rr.get("Race", rr.get("EventName", f"Round_{r}"))),
                    "Abbreviation": drv,
                    "TeamName": team,
                    "ActualFinish": finish,

                    "GridPosition": float(effective_grid),
                    "ActualStartingGrid": actual_grid,
                    "simulation_start_rank": float(sim_rank),
                    "is_pit_lane_start": float(is_pit),
                    "grid_penalty_places": 0.0,
                    "grid_vs_quali_delta": float(effective_grid - q_pos_val),
                    "grid_norm": grid_norm,
                    "expected_grid_finish": conv_feat["expected_grid_finish"],
                    "grid_retention_prob": conv_feat["grid_retention_prob"],

                    "teammate_quali_gap_pct": float(tm_q_gap) if pd.notna(tm_q_gap) else np.nan,
                    "teammate_grid_delta": float(tm_g_delta) if pd.notna(tm_g_delta) else np.nan,
                    "driver_teammate_h2h_ratio": float(tm_h2h),

                    # Experimental Sprint features: supported and leakage-free, but Phase 6E showed only marginal/mixed predictive improvement.
                    "sprint_quali_pos": float(sq_dict["sprint_quali_pos"]) if pd.notna(sq_dict.get("sprint_quali_pos")) else np.nan,
                    "sprint_finish_pos": float(sr_dict["sprint_finish_pos"]) if pd.notna(sr_dict.get("sprint_finish_pos")) else np.nan,
                    "sprint_pos_delta": float(sr_dict["sprint_pos_delta"]) if pd.notna(sr_dict.get("sprint_pos_delta")) else np.nan,

                    "fp1_best_lap": p_dict.get("fp1_best_lap", np.nan),
                    "fp2_best_lap": p_dict.get("fp2_best_lap", np.nan),
                    "fp3_best_lap": p_dict.get("fp3_best_lap", np.nan),
                    "best_practice_lap": p_dict.get("best_practice_lap", np.nan),
                    "avg_practice_lap": p_dict.get("avg_practice_lap", np.nan),
                    "practice_gap_pct": p_dict.get("practice_gap_pct", np.nan),
                    "practice_rank": p_dict.get("practice_rank", float(self.expected_grid_size)),
                    "practice_laps": p_dict.get("practice_laps", 0),
                    "practice_laps_norm": p_dict.get("practice_laps_norm", 0.0),
                    "practice_sessions": p_dict.get("practice_sessions", 0),
                    "fp1_gap_pct": p_dict.get("fp1_gap_pct", np.nan),
                    "fp2_gap_pct": p_dict.get("fp2_gap_pct", np.nan),
                    "fp3_gap_pct": p_dict.get("fp3_gap_pct", np.nan),

                    "QualifyingPosition": float(q_dict.get("QualifyingPosition", effective_grid)),
                    "quali_pos_norm": float(q_dict.get("quali_pos_norm", (effective_grid - 1.0) / float(self.expected_grid_size - 1))),
                    "quali_gap_pct": q_dict.get("quali_gap_pct", np.nan),
                    "quali_no_time": q_dict.get("quali_no_time", 1),
                    "q1_gap_pct": q_dict.get("q1_gap_pct", np.nan),
                    "q2_gap_pct": q_dict.get("q2_gap_pct", np.nan),
                    "q3_gap_pct": q_dict.get("q3_gap_pct", np.nan),

                    "avg_finish": dfv.get("avg_finish", 12.0),
                    "avg_grid": dfv.get("avg_grid", 12.0),
                    "total_points": dfv.get("total_points", 0.0),
                    "races_completed": dfv.get("races_completed", 0),
                    "recent_finish": dfv.get("recent_finish", 12.0),
                    "recent_grid": dfv.get("recent_grid", 12.0),
                    "recent_points": dfv.get("recent_points", 0.0),
                    "driver_finish_ewma": dfv.get("driver_finish_ewma", 12.0),
                    "driver_quali_ewma": dfv.get("driver_quali_ewma", 12.0),
                    "driver_dnf_rate": dfv.get("driver_dnf_rate", 0.0),

                    "team_avg_finish": tfv.get("team_avg_finish", 12.0),
                    "team_points": tfv.get("team_points", 0.0),
                    "team_finish_ewma": tfv.get("team_finish_ewma", 12.0),
                    "team_points_rate": tfv.get("team_points_rate", 0.0),
                })

        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("Training dataset generation produced 0 rows.")

        if stage is not None:
            df = apply_stage_feature_mask(df, stage)

        logger.info(f"Built training feature set: {len(df)} total driver-race samples.")
        return df

    def build_target_dataset(
        self,
        target_round: int,
        track_name: str,
        quali_df: Optional[pd.DataFrame],
        practice_sessions: Dict[str, Optional[pd.DataFrame]],
        season_races_df: pd.DataFrame,
        season_qualis_df: Optional[pd.DataFrame] = None,
        penalties: Optional[Dict[str, int]] = None,
        historical_track_df: Optional[pd.DataFrame] = None,
        starting_grid_df: Optional[pd.DataFrame] = None,
        stage: str = "after_qualifying",
    ) -> pd.DataFrame:
        """
        Builds the target feature dataset for the upcoming race.
        STRICT PRE-RACE GUARANTEES:
        - NEVER includes 'ActualFinish' or any post-race data.
        - Calculates effective starting grid applying known penalties or authoritative manifest.
        - Strictly evaluates rolling form on rounds < target_round.

        Parameters
        ----------
        target_round : int
            Round number of the target race (e.g. 16 for Sepang).
        track_name : str
            Display name of the track (e.g. 'Sepang').
        quali_df : pd.DataFrame, optional
            Qualifying session results.
        practice_sessions : dict
            Mapping of 'fp1', 'fp2', 'fp3' to practice DataFrames.
        season_races_df : pd.DataFrame
            Historical completed races of the current season.
        season_qualis_df : pd.DataFrame, optional
            Historical completed qualifying sessions of the season.
        penalties : dict, optional
            Driver grid penalties (e.g. {'HAD': 5, 'COL': 15, 'LIN': 30}).
        historical_track_df : pd.DataFrame, optional
            Multi-year historical data for this circuit if available.
        starting_grid_df : pd.DataFrame, optional
            Authoritative starting grid DataFrame (e.g. from StartingGridRetriever).
        stage : str
            Prediction stage cutoff.

        Returns
        -------
        pd.DataFrame
            Target features with exactly expected_grid_size rows and no ActualFinish column.
        """
        logger.info(f"Building target features for {track_name} (Round {target_round}, Stage: {stage})...")

        if quali_df is not None and not quali_df.empty:
            q_sess = quali_df.copy()
            if "Abbreviation" not in q_sess.columns and "Driver" in q_sess.columns:
                q_sess["Abbreviation"] = q_sess["Driver"]
            q_sess["Abbreviation"] = normalize_driver_series(q_sess["Abbreviation"])
        elif starting_grid_df is not None and not starting_grid_df.empty:
            q_sess = starting_grid_df.copy()
            if "Abbreviation" not in q_sess.columns and "Driver" in q_sess.columns:
                q_sess["Abbreviation"] = q_sess["Driver"]
            q_sess["Abbreviation"] = normalize_driver_series(q_sess["Abbreviation"])
        else:
            raise ValueError(f"Target qualifying or starting grid data is required for {track_name}.")

        q = q_sess.copy()
        pos_col = "Position" if "Position" in q.columns else "QualifyingPosition"
        q["Position"] = pd.to_numeric(q[pos_col], errors="coerce")
        q = q.dropna(subset=["Abbreviation", "Position"]).drop_duplicates("Abbreviation")

        if starting_grid_df is not None and not starting_grid_df.empty:
            q_grid = starting_grid_df.copy()
            if "Abbreviation" not in q_grid.columns and "Driver" in q_grid.columns:
                q_grid["Abbreviation"] = q_grid["Driver"]
            q_grid["Abbreviation"] = normalize_driver_series(q_grid["Abbreviation"])
        else:
            penalties = penalties or {}
            q_grid = resolve_fia_starting_grid(q, penalties=penalties, expected_grid_size=len(q))

        if len(q_grid) != self.expected_grid_size:
            logger.warning(
                f"Starting grid lineup has {len(q_grid)} drivers (expected {self.expected_grid_size})."
            )

        grid_lookup = {}
        for _, grow in q_grid.iterrows():
            d = grow["Abbreviation"]
            grid_lookup[d] = grow.to_dict()

        drivers = q_grid["Abbreviation"].tolist()
        team_col = "TeamName" if "TeamName" in q_grid.columns else ("TeamName" if "TeamName" in q_sess.columns else "Team")
        team_map = q_grid.set_index("Abbreviation")[team_col].to_dict() if team_col in q_grid.columns else (
            q_sess.set_index("Abbreviation")[team_col].to_dict() if team_col in q_sess.columns else {}
        )

        practice = build_practice_features(practice_sessions, drivers)
        quali_feats = build_qualifying_features(q_sess, drivers)

        # Sprint features (if sprint sessions were held and available prior to stage cutoff)
        sq_df = practice_sessions.get("sprint_qualifying") if practice_sessions else None
        sr_df = practice_sessions.get("sprint_race") if practice_sessions else None
        sq_feats = build_sprint_qualifying_features(sq_df, drivers)
        sr_feats = build_sprint_race_features(sr_df, drivers)

        # Strict chronological form and grid conversion up to Round < target_round
        dform, tform = build_season_form(
            target_round, season_races_df, season_qualis_df, alpha=self.ewma_alpha
        )
        prior_target_races = season_races_df[season_races_df["Round"] < target_round]
        target_conv_map = compute_grid_conversion_map(prior_target_races, expected_grid_size=self.expected_grid_size)

        # Teammate features
        tm_quali_feats = compute_teammate_quali_gap(q, drivers)
        grid_col_for_tm = "ActualStartingGrid" if "ActualStartingGrid" in q_grid.columns else "GridPosition"
        tm_grid_feats = compute_teammate_grid_delta(q_grid, drivers, grid_col=grid_col_for_tm)
        tm_h2h_map = compute_rolling_teammate_h2h(prior_target_races, q)

        rows = []
        for drv in drivers:
            p = practice[practice["Abbreviation"] == drv]
            qq = quali_feats[quali_feats["Abbreviation"] == drv]
            sq_row = sq_feats[sq_feats["Abbreviation"] == drv]
            sr_row = sr_feats[sr_feats["Abbreviation"] == drv]

            p_dict = p.iloc[0].to_dict() if not p.empty else {}
            q_dict = qq.iloc[0].to_dict() if not qq.empty else {}
            sq_dict = sq_row.iloc[0].to_dict() if not sq_row.empty else {}
            sr_dict = sr_row.iloc[0].to_dict() if not sr_row.empty else {}

            tm_q_row = tm_quali_feats[tm_quali_feats["Abbreviation"] == drv]
            tm_g_row = tm_grid_feats[tm_grid_feats["Abbreviation"] == drv]
            tm_q_gap = tm_q_row["teammate_quali_gap_pct"].iloc[0] if not tm_q_row.empty else np.nan
            tm_g_delta = tm_g_row["teammate_grid_delta"].iloc[0] if not tm_g_row.empty else np.nan
            tm_h2h = tm_h2h_map.get(drv, 0.50)

            team = str(team_map.get(drv, "Unknown"))
            dfv = dform.get(drv, {})
            tfv = tform.get(team, {})

            g_info = grid_lookup.get(drv, {})
            actual_grid = g_info.get("ActualStartingGrid")
            sim_rank = g_info.get("simulation_start_rank")
            is_pit = float(g_info.get("is_pit_lane_start", 0))
            status = str(g_info.get("StartingGridStatus", "CLASSIFIED"))
            pen_places = float(g_info.get("grid_penalty_places", g_info.get("GridPenalty", 0)))
            delta = float(g_info.get("grid_vs_quali_delta", 0))

            if is_pit == 1.0 or pd.isna(actual_grid):
                effective_grid = float(sim_rank) if pd.notna(sim_rank) else float(self.expected_grid_size)
                actual_grid_val = np.nan
            else:
                effective_grid = float(actual_grid)
                actual_grid_val = float(actual_grid)

            quali_pos = float(q_dict.get("QualifyingPosition", effective_grid))
            conv_feat = get_conversion_features_for_grid(effective_grid, target_conv_map, expected_grid_size=self.expected_grid_size)
            grid_norm_val = (effective_grid - 1.0) / float(self.expected_grid_size - 1)

            rows.append({
                "Round": target_round,
                "RaceName": track_name,
                "Abbreviation": drv,
                "TeamName": team,

                # STRICT PRE-RACE: ActualFinish is intentionally excluded!

                "GridPosition": effective_grid,
                "ActualStartingGrid": actual_grid_val,
                "simulation_start_rank": float(sim_rank) if pd.notna(sim_rank) else effective_grid,
                "is_pit_lane_start": is_pit,
                "StartingGridStatus": status,
                "GridPenalty": int(pen_places),
                "grid_penalty_places": pen_places,
                "grid_vs_quali_delta": delta,
                "grid_norm": grid_norm_val,
                "expected_grid_finish": conv_feat["expected_grid_finish"],
                "grid_retention_prob": conv_feat["grid_retention_prob"],

                "teammate_quali_gap_pct": float(tm_q_gap) if pd.notna(tm_q_gap) else np.nan,
                "teammate_grid_delta": float(tm_g_delta) if pd.notna(tm_g_delta) else np.nan,
                "driver_teammate_h2h_ratio": float(tm_h2h),

                # Experimental Sprint features: supported and leakage-free, but Phase 6E showed only marginal/mixed predictive improvement.
                "sprint_quali_pos": float(sq_dict["sprint_quali_pos"]) if pd.notna(sq_dict.get("sprint_quali_pos")) else np.nan,
                "sprint_finish_pos": float(sr_dict["sprint_finish_pos"]) if pd.notna(sr_dict.get("sprint_finish_pos")) else np.nan,
                "sprint_pos_delta": float(sr_dict["sprint_pos_delta"]) if pd.notna(sr_dict.get("sprint_pos_delta")) else np.nan,

                "fp1_best_lap": p_dict.get("fp1_best_lap", np.nan),
                "fp2_best_lap": p_dict.get("fp2_best_lap", np.nan),
                "fp3_best_lap": p_dict.get("fp3_best_lap", np.nan),
                "best_practice_lap": p_dict.get("best_practice_lap", np.nan),
                "avg_practice_lap": p_dict.get("avg_practice_lap", np.nan),
                "practice_gap_pct": p_dict.get("practice_gap_pct", np.nan),
                "practice_rank": p_dict.get("practice_rank", float(self.expected_grid_size)),
                "practice_laps": p_dict.get("practice_laps", 0),
                "practice_laps_norm": p_dict.get("practice_laps_norm", 0.0),
                "practice_sessions": p_dict.get("practice_sessions", 0),
                "fp1_gap_pct": p_dict.get("fp1_gap_pct", np.nan),
                "fp2_gap_pct": p_dict.get("fp2_gap_pct", np.nan),
                "fp3_gap_pct": p_dict.get("fp3_gap_pct", np.nan),

                "QualifyingPosition": quali_pos,
                "quali_pos_norm": (quali_pos - 1.0) / float(self.expected_grid_size - 1),
                "quali_gap_pct": q_dict.get("quali_gap_pct", np.nan),
                "quali_no_time": q_dict.get("quali_no_time", 1),
                "q1_gap_pct": q_dict.get("q1_gap_pct", np.nan),
                "q2_gap_pct": q_dict.get("q2_gap_pct", np.nan),
                "q3_gap_pct": q_dict.get("q3_gap_pct", np.nan),

                "avg_finish": dfv.get("avg_finish", 12.0),
                "avg_grid": dfv.get("avg_grid", 12.0),
                "total_points": dfv.get("total_points", 0.0),
                "races_completed": dfv.get("races_completed", 0),
                "recent_finish": dfv.get("recent_finish", 12.0),
                "recent_grid": dfv.get("recent_grid", 12.0),
                "recent_points": dfv.get("recent_points", 0.0),
                "driver_finish_ewma": dfv.get("driver_finish_ewma", 12.0),
                "driver_quali_ewma": dfv.get("driver_quali_ewma", 12.0),
                "driver_dnf_rate": dfv.get("driver_dnf_rate", 0.0),

                "team_avg_finish": tfv.get("team_avg_finish", 12.0),
                "team_points": tfv.get("team_points", 0.0),
                "team_finish_ewma": tfv.get("team_finish_ewma", 12.0),
                "team_points_rate": tfv.get("team_points_rate", 0.0),
            })

        target_df = pd.DataFrame(rows)

        if "ActualFinish" in target_df.columns:
            raise AssertionError("LEAKAGE DETECTED: ActualFinish found in target dataset!")

        if stage is not None:
            target_df = apply_stage_feature_mask(target_df, stage)

        if len(target_df) != self.expected_grid_size:
            logger.warning(
                f"Target feature count is {len(target_df)} (expected {self.expected_grid_size})."
            )

        logger.info(f"Target dataset built successfully for {track_name}: {len(target_df)} drivers.")
        return target_df
