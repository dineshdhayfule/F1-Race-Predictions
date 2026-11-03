"""
Chronological Walk-Forward Validation Engine for F1 Benchmarks.
Manages race-level folds, temporal data isolation, and feature matrix preparation
for both 2026 Championship (Track B) and Historical Circuit Races (Track A).
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Generator, Optional
import json
import numpy as np
import pandas as pd

from experiments.model_benchmark.config import (
    ROOT_DIR,
    DATA_DIR,
    BASELINE_FEATURES,
    BENCHMARK_STAGE,
    MIN_TRAIN_ROUNDS_2026,
    MIN_TRAIN_ROUNDS_HISTORICAL,
)
from src.data.data_loader import DataLoader
from src.data.data_validator import DataValidator
from src.features.build_features import FeatureEngineer, apply_stage_feature_mask
from src.utils.driver_mapping import filter_competing_drivers, normalize_driver_series
from src.utils.logging_utils import get_logger

logger = get_logger("benchmark_walk_forward")


class WalkForwardEngine:
    """
    Constructs chronological walk-forward folds without future leakage.
    Guarantees all drivers from the same race are evaluated together in test fold.
    """

    def __init__(self, root_dir: Optional[Path] = None):
        self.root_dir = root_dir or ROOT_DIR
        self.loader = DataLoader(self.root_dir)
        self.validator = DataValidator()
        self.feature_engineer = FeatureEngineer()
        self.tracks_config = self._load_tracks_config()

    def _load_tracks_config(self) -> Dict[str, Any]:
        cfg_path = self.root_dir / "config" / "tracks.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                return json.load(f).get("tracks", {})
        return {}

    def get_track_key_for_round_2026(self, round_num: int) -> str:
        for k, v in self.tracks_config.items():
            if v.get("round_2026") == round_num:
                return k
        return f"round_{round_num}"

    def build_2026_folds(
        self,
        min_train_rounds: int = MIN_TRAIN_ROUNDS_2026
    ) -> List[Dict[str, Any]]:
        """
        Builds chronological walk-forward evaluation folds across completed 2026 championship rounds.
        """
        season_dir = self.root_dir / "data" / "raw" / "season" / "2026"
        race_files = sorted(list(season_dir.glob("2026_round_*_race.csv")))

        # Extract round numbers
        available_rounds = []
        for p in race_files:
            name = p.stem
            # e.g. 2026_round_1_race or 2026_round_12_sprint_race
            if "_sprint_" in name:
                continue
            parts = name.split("_")
            try:
                rnd = int(parts[2])
                available_rounds.append(rnd)
            except (ValueError, IndexError):
                pass

        available_rounds = sorted(list(set(available_rounds)))
        test_rounds = [r for r in available_rounds if r > min_train_rounds]

        logger.info(f"Discovered {len(available_rounds)} 2026 rounds: {available_rounds}")
        logger.info(f"Configured {len(test_rounds)} test folds (Rounds {test_rounds})")

        # Load all race and qualifying data up to max round once
        all_races_df, all_qualis_df = self.loader.load_season_rounds(2026, available_rounds)
        all_practice_sessions = self.loader.load_all_practice_sessions(2026, available_rounds)

        folds = []
        for target_round in test_rounds:
            train_rounds = [r for r in available_rounds if r < target_round]
            track_key = self.get_track_key_for_round_2026(target_round)
            track_info = self.tracks_config.get(track_key, {})
            track_name = track_info.get("display_name", track_key.title())

            # 1. Build training features strictly using prior rounds
            train_features = self.feature_engineer.build_training_dataset(
                races_df=all_races_df,
                qualis_df=all_qualis_df,
                practice_sessions_by_round=all_practice_sessions,
                train_rounds=train_rounds
            )

            # 2. Build target test features
            target_race = all_races_df[all_races_df["Round"] == target_round].copy()
            target_quali = all_qualis_df[all_qualis_df["Round"] == target_round].copy()
            weekend_sessions = all_practice_sessions.get(target_round, {})
            penalties = track_info.get("known_penalties_2026", {})

            # Multi-source starting grid resolution
            starting_grid_df, grid_meta = self.loader.load_starting_grid(
                track_id=track_key,
                year=2026,
                round_num=target_round,
                stage=BENCHMARK_STAGE,
                quali_df=target_quali,
                penalties=penalties
            )

            test_features = self.feature_engineer.build_target_dataset(
                target_round=target_round,
                track_name=track_name,
                quali_df=target_quali,
                practice_sessions=weekend_sessions,
                season_races_df=all_races_df[all_races_df["Round"] < target_round],
                season_qualis_df=all_qualis_df[all_qualis_df["Round"] < target_round],
                penalties=penalties,
                starting_grid_df=starting_grid_df,
                stage=BENCHMARK_STAGE,
                track_key=track_key
            )

            # Strict leakage assertion
            audit = self.validator.audit_leakage(train_features, test_features, target_round=target_round)
            assert audit["is_clean"] is True, f"Leakage violation on Round {target_round}: {audit['violations']}"

            # Prepare X and y
            train_clean = train_features.dropna(subset=["ActualFinish"]).copy()
            train_clean = apply_stage_feature_mask(train_clean, BENCHMARK_STAGE)
            test_clean = apply_stage_feature_mask(test_features.copy(), BENCHMARK_STAGE)

            for col in BASELINE_FEATURES:
                if col not in train_clean.columns:
                    train_clean[col] = np.nan
                if col not in test_clean.columns:
                    test_clean[col] = np.nan

            # Match actual race finishing positions
            target_race["Abbreviation"] = normalize_driver_series(target_race["Abbreviation"])
            pos_col = "Position" if "Position" in target_race.columns else "ActualFinish"
            target_race["ActualFinish"] = pd.to_numeric(target_race[pos_col], errors="coerce")

            eval_df = test_clean.merge(
                target_race[["Abbreviation", "ActualFinish"]],
                on="Abbreviation",
                how="inner"
            ).sort_values("ActualFinish").reset_index(drop=True)

            X_train = train_clean[BASELINE_FEATURES].copy()
            y_train = pd.to_numeric(train_clean["ActualFinish"], errors="coerce").values
            X_test = eval_df[BASELINE_FEATURES].copy()
            y_test = eval_df["ActualFinish"].values

            folds.append({
                "track": "Track B (2026 Season)",
                "round": target_round,
                "track_key": track_key,
                "track_name": track_name,
                "circuit_type": track_info.get("circuit_type", "permanent"),
                "train_rounds": train_rounds,
                "train_sample_count": len(X_train),
                "test_sample_count": len(X_test),
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
                "meta_train": train_clean,
                "meta_test": eval_df,
                "driver_ids": eval_df["Abbreviation"].tolist()
            })

        return folds

    def build_historical_folds(
        self,
        min_train_races: int = MIN_TRAIN_ROUNDS_HISTORICAL
    ) -> List[Dict[str, Any]]:
        """
        Builds chronological walk-forward folds across available historical circuit datasets
        (Baku 2021-2025 and Sao Paulo 2022-2025).
        """
        historical_races = [
            {"year": 2021, "track_key": "baku", "name": "Baku City Circuit", "type": "street"},
            {"year": 2022, "track_key": "baku", "name": "Baku City Circuit", "type": "street"},
            {"year": 2022, "track_key": "sao_paulo", "name": "Autódromo José Carlos Pace", "type": "permanent"},
            {"year": 2023, "track_key": "baku", "name": "Baku City Circuit", "type": "street"},
            {"year": 2023, "track_key": "sao_paulo", "name": "Autódromo José Carlos Pace", "type": "permanent"},
            {"year": 2024, "track_key": "baku", "name": "Baku City Circuit", "type": "street"},
            {"year": 2024, "track_key": "sao_paulo", "name": "Autódromo José Carlos Pace", "type": "permanent"},
            {"year": 2025, "track_key": "baku", "name": "Baku City Circuit", "type": "street"},
            {"year": 2025, "track_key": "sao_paulo", "name": "Autódromo José Carlos Pace", "type": "permanent"},
        ]

        # Load all race files
        all_hist_dfs = []
        for idx, ev in enumerate(historical_races):
            yr = ev["year"]
            tk = ev["track_key"]
            p_race = self.root_dir / "data" / "raw" / "tracks" / tk / "history" / f"{tk}_{yr}_race.csv"
            p_quali = self.root_dir / "data" / "raw" / "tracks" / tk / "history" / f"{tk}_{yr}_qualifying.csv"
            if not p_race.exists():
                continue

            rdf = pd.read_csv(p_race)
            qdf = pd.read_csv(p_quali) if p_quali.exists() else pd.DataFrame()

            rdf["Abbreviation"] = normalize_driver_series(rdf.get("Abbreviation", rdf.get("Driver", "")))
            rdf["EventIndex"] = idx + 1
            rdf["Year"] = yr
            rdf["TrackKey"] = tk
            pos_col = "Position" if "Position" in rdf.columns else "ActualFinish"
            rdf["ActualFinish"] = pd.to_numeric(rdf[pos_col], errors="coerce")
            rdf["GridPosition"] = pd.to_numeric(rdf["GridPosition"], errors="coerce").fillna(12.0)
            rdf["QualifyingPosition"] = pd.to_numeric(rdf.get("QualifyingPosition", rdf["GridPosition"]), errors="coerce").fillna(12.0)

            # Compute normalized grid
            n_d = len(rdf)
            rdf["grid_norm"] = (rdf["GridPosition"] - 1.0) / float(max(n_d - 1, 1))
            rdf["expected_grid_finish"] = rdf["GridPosition"]
            rdf["grid_retention_prob"] = 0.65

            all_hist_dfs.append(rdf)

        if not all_hist_dfs:
            return []

        master_hist = pd.concat(all_hist_dfs, ignore_index=True)

        folds = []
        n_events = len(all_hist_dfs)
        test_indices = list(range(min_train_races + 1, n_events + 1))

        for t_idx in test_indices:
            train_df = master_hist[master_hist["EventIndex"] < t_idx].copy()
            test_df = master_hist[master_hist["EventIndex"] == t_idx].copy()

            if train_df.empty or test_df.empty:
                continue

            target_info = historical_races[t_idx - 1]

            # Compute rolling driver finish EWMA strictly on prior races
            driver_ewma = {}
            for drv, grp in train_df.groupby("Abbreviation"):
                finishes = pd.to_numeric(grp["ActualFinish"], errors="coerce").dropna().tolist()
                driver_ewma[drv] = float(np.mean(finishes[-3:])) if finishes else 12.0

            for df in (train_df, test_df):
                df["driver_finish_ewma"] = df["Abbreviation"].map(driver_ewma).fillna(12.0)
                df["driver_quali_ewma"] = df["GridPosition"]
                df["driver_dnf_rate"] = 0.158
                df["recent_finish"] = df["driver_finish_ewma"]
                df["recent_grid"] = df["GridPosition"]
                df["recent_points"] = 0.0
                df["avg_finish"] = df["driver_finish_ewma"]
                df["avg_grid"] = df["GridPosition"]
                df["total_points"] = 0.0
                df["races_completed"] = 3
                df["team_finish_ewma"] = 12.0
                df["team_points_rate"] = 0.0
                df["team_avg_finish"] = 12.0
                df["team_points"] = 0.0
                df["sprint_quali_pos"] = np.nan
                df["sprint_finish_pos"] = np.nan
                df["sprint_pos_delta"] = np.nan
                df["teammate_quali_gap_pct"] = np.nan
                df["teammate_grid_delta"] = np.nan
                df["driver_teammate_h2h_ratio"] = 0.50
                # Practice features are all NaN for historical track sets
                for col in [
                    "quali_pos_norm", "quali_gap_pct", "quali_no_time", "q1_gap_pct",
                    "q2_gap_pct", "q3_gap_pct", "practice_gap_pct", "practice_rank",
                    "practice_laps_norm", "fp1_gap_pct", "fp2_gap_pct", "fp3_gap_pct"
                ]:
                    if col not in df.columns:
                        df[col] = np.nan

            eval_df = test_df.dropna(subset=["ActualFinish"]).sort_values("ActualFinish").reset_index(drop=True)

            X_train = train_df[BASELINE_FEATURES].copy()
            y_train = train_df["ActualFinish"].values
            X_test = eval_df[BASELINE_FEATURES].copy()
            y_test = eval_df["ActualFinish"].values

            folds.append({
                "track": "Track A (Historical Tracks)",
                "round": target_info["year"],
                "track_key": target_info["track_key"],
                "track_name": f"{target_info['name']} ({target_info['year']})",
                "circuit_type": target_info["type"],
                "train_rounds": [h["year"] for h in historical_races[:t_idx - 1]],
                "train_sample_count": len(X_train),
                "test_sample_count": len(X_test),
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
                "meta_train": train_df,
                "meta_test": eval_df,
                "driver_ids": eval_df["Abbreviation"].tolist()
            })

        return folds
