"""
Unified Feature Engineering Pipeline
====================================
Orchestrates training and target feature matrix construction for any F1 circuit.
Enforces zero data leakage: target race result is strictly excluded from target features.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

from src.features.grid_features import apply_grid_penalties
from src.features.grid_conversion import compute_grid_conversion_map, get_conversion_features_for_grid
from src.features.season_features import build_season_form
from src.features.weekend_features import (
    build_practice_features,
    build_qualifying_features,
    build_sprint_qualifying_features,
    build_sprint_race_features,
)
from src.utils.driver_mapping import normalize_driver_series
from src.utils.logging_utils import get_logger

logger = get_logger("build_features")


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
        train_rounds: List[int]
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

                grid = pd.to_numeric(rr.get("GridPosition"), errors="coerce")
                if pd.isna(grid):
                    grid = 12.0

                dfv = dform.get(drv, {})
                tfv = tform.get(team, {})

                conv_feat = get_conversion_features_for_grid(grid, grid_conv_map, expected_grid_size=self.expected_grid_size)

                finish = pd.to_numeric(rr.get("Position"), errors="coerce")

                rows.append({
                    "Round": r,
                    "RaceName": str(rr.get("Race", rr.get("EventName", f"Round_{r}"))),
                    "Abbreviation": drv,
                    "TeamName": team,
                    "ActualFinish": finish,

                    "GridPosition": float(grid),
                    "grid_norm": (float(grid) - 1.0) / float(self.expected_grid_size - 1),
                    "expected_grid_finish": conv_feat["expected_grid_finish"],
                    "grid_retention_prob": conv_feat["grid_retention_prob"],

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

                    "QualifyingPosition": float(q_dict.get("QualifyingPosition", grid)),
                    "quali_pos_norm": float(q_dict.get("quali_pos_norm", (grid - 1.0) / float(self.expected_grid_size - 1))),
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
        historical_track_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Builds the target feature dataset for the upcoming race.
        STRICT PRE-RACE GUARANTEES:
        - NEVER includes 'ActualFinish' or any post-race data.
        - Calculates effective starting grid applying known penalties.
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

        Returns
        -------
        pd.DataFrame
            Target features with exactly expected_grid_size rows and no ActualFinish column.
        """
        logger.info(f"Building target features for {track_name} (Round {target_round})...")

        if quali_df is None or quali_df.empty:
            raise ValueError(f"Target qualifying data is required to construct starting grid for {track_name}.")

        q = quali_df.copy()
        if "Abbreviation" not in q.columns:
            if "Driver" in q.columns:
                q["Abbreviation"] = q["Driver"]
            else:
                raise ValueError("Qualifying data missing driver abbreviation column.")

        q["Abbreviation"] = normalize_driver_series(q["Abbreviation"])
        pos_col = "Position" if "Position" in q.columns else "QualifyingPosition"
        q["Position"] = pd.to_numeric(q[pos_col], errors="coerce")
        q = q.dropna(subset=["Abbreviation", "Position"]).drop_duplicates("Abbreviation")

        if len(q) != self.expected_grid_size:
            logger.warning(
                f"Qualifying lineup has {len(q)} drivers (expected {self.expected_grid_size})."
            )

        # Apply grid penalties to construct valid 1-N starting grid
        penalties = penalties or {}
        q_grid = apply_grid_penalties(q, penalties=penalties, expected_grid_size=len(q))
        grid_map = q_grid.set_index("Abbreviation")["ActualStartingGrid"].to_dict()

        drivers = q["Abbreviation"].tolist()
        team_col = "TeamName" if "TeamName" in q.columns else "Team"
        team_map = q.set_index("Abbreviation")[team_col].to_dict() if team_col in q.columns else {}

        practice = build_practice_features(practice_sessions, drivers)
        quali_feats = build_qualifying_features(q, drivers)

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

            team = str(team_map.get(drv, "Unknown"))
            dfv = dform.get(drv, {})
            tfv = tform.get(team, {})

            effective_grid = float(grid_map.get(drv, q_dict.get("QualifyingPosition", 12.0)))
            quali_pos = float(q_dict.get("QualifyingPosition", effective_grid))
            conv_feat = get_conversion_features_for_grid(effective_grid, target_conv_map, expected_grid_size=self.expected_grid_size)

            rows.append({
                "Round": target_round,
                "RaceName": track_name,
                "Abbreviation": drv,
                "TeamName": team,

                # STRICT PRE-RACE: ActualFinish is intentionally excluded!

                "GridPosition": effective_grid,
                "ActualStartingGrid": effective_grid,
                "grid_norm": (effective_grid - 1.0) / float(self.expected_grid_size - 1),
                "expected_grid_finish": conv_feat["expected_grid_finish"],
                "grid_retention_prob": conv_feat["grid_retention_prob"],

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

        if len(target_df) != self.expected_grid_size:
            logger.warning(
                f"Target feature count is {len(target_df)} (expected {self.expected_grid_size})."
            )

        logger.info(f"Target dataset built successfully for {track_name}: {len(target_df)} drivers.")
        return target_df
