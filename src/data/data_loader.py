"""
Unified Stage-Aware Data Loader.
Enforces information cutoffs based on prediction stage.
"""

from pathlib import Path
import json
from typing import Dict, Optional, Tuple
import pandas as pd

from src.utils.driver_mapping import normalize_driver


SESSION_NORMALIZATION_MAP = {
    "fp1": "fp1",
    "practice 1": "fp1",
    "free practice 1": "fp1",
    "fp2": "fp2",
    "practice 2": "fp2",
    "free practice 2": "fp2",
    "fp3": "fp3",
    "practice 3": "fp3",
    "free practice 3": "fp3",
    "sprint qualifying": "sprint_qualifying",
    "sprint shootout": "sprint_qualifying",
    "sprint_shootout": "sprint_qualifying",
    "sprint_qualifying": "sprint_qualifying",
    "sq": "sprint_qualifying",
    "sprint": "sprint_race",
    "sprint race": "sprint_race",
    "sprint_race": "sprint_race",
    "qualifying": "qualifying",
    "quali": "qualifying",
    "race": "race",
}


def normalize_session_name(session_name: str) -> str:
    """
    Normalizes raw session names (e.g. from FastF1, user inputs, or filenames)
    into standard canonical tokens:
      - 'fp1', 'fp2', 'fp3', 'sprint_qualifying', 'sprint_race', 'qualifying', 'race'
    """
    if not session_name:
        return ""
    clean = str(session_name).strip().lower().replace("-", " ")
    if clean in SESSION_NORMALIZATION_MAP:
        return SESSION_NORMALIZATION_MAP[clean]
    clean_spaces = clean.replace("_", " ")
    if clean_spaces in SESSION_NORMALIZATION_MAP:
        return SESSION_NORMALIZATION_MAP[clean_spaces]
    return clean.replace(" ", "_")


class DataLoader:
    """
    Loads raw and processed datasets while respecting strict stage boundaries.
    """

    STAGES = [
        "pre_fp1",
        "after_fp1",
        "after_sprint_qualifying",
        "after_sprint_race",
        "after_fp2",
        "after_fp3",
        "after_qualifying",
        "post_race",
    ]

    def __init__(self, root_dir: Optional[Path] = None):
        provided = Path(root_dir) if root_dir else Path(__file__).resolve().parents[2]
        if provided.name == "raw" and provided.parent.name == "data":
            self.root_dir = provided.parent.parent
            self.raw_data_dir = provided
            self.processed_data_dir = self.root_dir / "data" / "processed"
        elif provided.name == "data":
            self.root_dir = provided.parent
            self.raw_data_dir = provided / "raw"
            self.processed_data_dir = provided / "processed"
        else:
            self.root_dir = provided
            self.raw_data_dir = self.root_dir / "data" / "raw"
            self.processed_data_dir = self.root_dir / "data" / "processed"
        self.config_dir = self.root_dir / "config"
        self.tracks_config = self._load_tracks_config()

    def _load_tracks_config(self) -> Dict:
        config_path = self.config_dir / "tracks.json"
        if not config_path.exists():
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f).get("tracks", {})

    def get_track_info(self, track_id: str) -> Dict:
        track_key = track_id.lower().strip()
        if track_key not in self.tracks_config:
            raise KeyError(f"Track '{track_id}' not found in tracks.json. Available: {list(self.tracks_config.keys())}")
        return self.tracks_config[track_key]

    def is_sprint_weekend(self, track_id: str, year: int = 2026) -> bool:
        """
        Determines whether a track is configured as a Sprint weekend for the given year.
        Checks 'sprint_weekend_{year}' first, then falls back to 'sprint_weekend'.
        """
        try:
            info = self.get_track_info(track_id)
        except KeyError:
            return False

        year_key = f"sprint_weekend_{year}"
        if year_key in info:
            return bool(info[year_key])
        return bool(info.get("sprint_weekend", False))

    def load_season_races(self, year: int, max_round: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads all completed races and qualifying sessions for a given year up to max_round.
        """
        season_dir = self.raw_data_dir / "season" / str(year)
        races = []
        qualis = []

        if not season_dir.exists():
            # Fallback to processed season directory if raw not populated yet
            season_dir = self.processed_data_dir / "season" / str(year)

        if not season_dir.exists():
            return pd.DataFrame(), pd.DataFrame()

        # Find all round race/qualifying files
        race_files = sorted(season_dir.glob(f"{year}_round_*_race.csv"))
        for rf in race_files:
            try:
                r_num = int(rf.stem.split("_")[2])
            except (IndexError, ValueError):
                continue

            if max_round is not None and r_num > max_round:
                continue

            qf = season_dir / f"{year}_round_{r_num}_qualifying.csv"

            df_r = pd.read_csv(rf)
            df_r = normalize_driver(df_r)
            df_r["Round"] = r_num
            races.append(df_r)

            if qf.exists():
                df_q = pd.read_csv(qf)
                df_q = normalize_driver(df_q)
                df_q["Round"] = r_num
                qualis.append(df_q)

        race_df = pd.concat(races, ignore_index=True) if races else pd.DataFrame()
        quali_df = pd.concat(qualis, ignore_index=True) if qualis else pd.DataFrame()
        return race_df, quali_df

    def load_season_rounds(self, year: int, rounds: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads race and qualifying results for an explicit list of round numbers.
        """
        season_dir = self.raw_data_dir / "season" / str(year)
        if not season_dir.exists():
            season_dir = self.processed_data_dir

        races = []
        qualis = []
        for r_num in rounds:
            rf = season_dir / f"{year}_round_{r_num}_race.csv"
            qf = season_dir / f"{year}_round_{r_num}_qualifying.csv"

            if rf.exists():
                df_r = pd.read_csv(rf)
                df_r = normalize_driver(df_r)
                df_r["Round"] = r_num
                races.append(df_r)

            if qf.exists():
                df_q = pd.read_csv(qf)
                df_q = normalize_driver(df_q)
                df_q["Round"] = r_num
                qualis.append(df_q)

        race_df = pd.concat(races, ignore_index=True) if races else pd.DataFrame()
        quali_df = pd.concat(qualis, ignore_index=True) if qualis else pd.DataFrame()
        return race_df, quali_df

    def load_all_practice_sessions(self, year: int, rounds: list) -> Dict[int, Dict[str, Optional[pd.DataFrame]]]:
        """
        Loads fp1, fp2, and fp3 sessions for all requested completed rounds.
        """
        season_dir = self.raw_data_dir / "season" / str(year)
        if not season_dir.exists():
            season_dir = self.processed_data_dir

        sessions_by_round = {}
        for r_num in rounds:
            r_sessions = {}
            for s in ("fp1", "fp2", "fp3"):
                sf = season_dir / f"{year}_round_{r_num}_{s}.csv"
                if sf.exists():
                    df = pd.read_csv(sf)
                    r_sessions[s] = normalize_driver(df)
                else:
                    r_sessions[s] = None
            sessions_by_round[r_num] = r_sessions

        return sessions_by_round

    def load_target_qualifying(self, track_id: Optional[str] = None, year: int = 2026, track_key: Optional[str] = None, **kwargs) -> Optional[pd.DataFrame]:
        """
        Loads current target qualifying session results.
        """
        tid = track_id or track_key or kwargs.get("track") or ""
        track_k = str(tid).lower().strip()
        candidates = [
            self.raw_data_dir / "tracks" / track_k / str(year) / f"{year}_{track_k}_qualifying.csv",
            self.raw_data_dir / "tracks" / track_k / str(year) / "qualifying.csv",
            self.processed_data_dir / f"{year}_{track_k}_qualifying.csv",
        ]
        for p in candidates:
            if p.exists():
                df = pd.read_csv(p)
                return normalize_driver(df)
        return None

    def load_target_sprint_qualifying(self, track_id: Optional[str] = None, year: int = 2026, track_key: Optional[str] = None, **kwargs) -> Optional[pd.DataFrame]:
        """
        Loads current target sprint qualifying session results.
        """
        tid = track_id or track_key or kwargs.get("track") or ""
        track_k = str(tid).lower().strip()
        candidates = [
            self.raw_data_dir / "tracks" / track_k / str(year) / f"{year}_{track_k}_sprint_qualifying.csv",
            self.raw_data_dir / "tracks" / track_k / str(year) / f"{year}_{track_k}_sprint_shootout.csv",
            self.raw_data_dir / "tracks" / track_k / str(year) / "sprint_qualifying.csv",
            self.processed_data_dir / f"{year}_{track_k}_sprint_qualifying.csv",
            self.processed_data_dir / f"{year}_{track_k}_sprint_shootout.csv",
        ]
        for p in candidates:
            if p.exists():
                df = pd.read_csv(p)
                return normalize_driver(df)
        return None

    def load_target_sprint_race(self, track_id: Optional[str] = None, year: int = 2026, track_key: Optional[str] = None, **kwargs) -> Optional[pd.DataFrame]:
        """
        Loads current target sprint race results.
        """
        tid = track_id or track_key or kwargs.get("track") or ""
        track_k = str(tid).lower().strip()
        candidates = [
            self.raw_data_dir / "tracks" / track_k / str(year) / f"{year}_{track_k}_sprint_race.csv",
            self.raw_data_dir / "tracks" / track_k / str(year) / f"{year}_{track_k}_sprint.csv",
            self.raw_data_dir / "tracks" / track_k / str(year) / "sprint_race.csv",
            self.processed_data_dir / f"{year}_{track_k}_sprint_race.csv",
            self.processed_data_dir / f"{year}_{track_k}_sprint.csv",
        ]
        for p in candidates:
            if p.exists():
                df = pd.read_csv(p)
                return normalize_driver(df)
        return None

    def load_weekend_sessions(
        self,
        track_id: Optional[str] = None,
        year: int = 2026,
        stage: str = "after_qualifying",
        track_key: Optional[str] = None,
        is_sprint: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Loads weekend sessions up to the declared stage cutoff.
        Respects weekend format (normal vs sprint) and prevents future-session data leakage.
        """
        stage = stage.lower().strip()
        if stage not in self.STAGES:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of: {self.STAGES}")

        tid = track_id or track_key or kwargs.get("track") or ""
        track_k = str(tid).lower().strip()

        if is_sprint is None:
            is_sprint = self.is_sprint_weekend(track_k, year)

        track_dir = self.raw_data_dir / "tracks" / track_k / str(year)
        if not track_dir.exists():
            track_dir = self.processed_data_dir

        sessions = {
            "fp1": None,
            "fp2": None,
            "fp3": None,
            "sprint_qualifying": None,
            "sprint_race": None,
            "qualifying": None,
            "race": None,
        }

        # Stage cutoffs depending on weekend format
        if is_sprint:
            if stage in ["after_fp2", "after_fp3"]:
                raise ValueError(
                    f"Stage '{stage}' is invalid for Sprint weekend format. "
                    f"Valid Sprint stages are: ['pre_fp1', 'after_fp1', 'after_sprint_qualifying', 'after_sprint_race', 'after_qualifying', 'post_race']."
                )
            if stage == "pre_fp1":
                allowed_sessions = []
            elif stage == "after_fp1":
                allowed_sessions = ["fp1"]
            elif stage == "after_sprint_qualifying":
                allowed_sessions = ["fp1", "sprint_qualifying"]
            elif stage == "after_sprint_race":
                allowed_sessions = ["fp1", "sprint_qualifying", "sprint_race"]
            elif stage == "after_qualifying":
                allowed_sessions = ["fp1", "sprint_qualifying", "sprint_race", "qualifying"]
            elif stage == "post_race":
                allowed_sessions = ["fp1", "sprint_qualifying", "sprint_race", "qualifying", "race"]
            else:
                allowed_sessions = []
        else:
            if stage in ["after_sprint_qualifying", "after_sprint_race"]:
                raise ValueError(
                    f"Stage '{stage}' is invalid for Normal weekend format. "
                    f"Valid Normal stages are: ['pre_fp1', 'after_fp1', 'after_fp2', 'after_fp3', 'after_qualifying', 'post_race']."
                )
            if stage == "pre_fp1":
                allowed_sessions = []
            elif stage == "after_fp1":
                allowed_sessions = ["fp1"]
            elif stage == "after_fp2":
                allowed_sessions = ["fp1", "fp2"]
            elif stage == "after_fp3":
                allowed_sessions = ["fp1", "fp2", "fp3"]
            elif stage == "after_qualifying":
                allowed_sessions = ["fp1", "fp2", "fp3", "qualifying"]
            elif stage == "post_race":
                allowed_sessions = ["fp1", "fp2", "fp3", "qualifying", "race"]
            else:
                allowed_sessions = []

        for s in allowed_sessions:
            candidates = [
                track_dir / f"{s}.csv",
                track_dir / f"{year}_{track_k}_{s}.csv",
                self.processed_data_dir / f"{year}_{track_k}_{s}.csv",
                self.raw_data_dir / "tracks" / track_k / str(year) / f"{year}_{track_k}_{s}.csv",
                self.raw_data_dir / "tracks" / track_k / str(year) / f"{s}.csv",
            ]
            if s == "sprint_qualifying":
                candidates.extend([
                    track_dir / f"{year}_{track_k}_sprint_shootout.csv",
                    track_dir / "sprint_shootout.csv",
                    self.processed_data_dir / f"{year}_{track_k}_sprint_shootout.csv",
                ])
            elif s == "sprint_race":
                candidates.extend([
                    track_dir / f"{year}_{track_k}_sprint.csv",
                    track_dir / "sprint.csv",
                    self.processed_data_dir / f"{year}_{track_k}_sprint.csv",
                ])

            for p in candidates:
                if p.exists():
                    df = pd.read_csv(p)
                    sessions[s] = normalize_driver(df)
                    break

        return sessions

    def load_historical_track_data(self, track_id: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Loads historical track data if configured as available.
        """
        info = self.get_track_info(track_id)
        if not info.get("historical_data_available", False):
            return None, None

        track_key = track_id.lower()
        hist_dir = self.raw_data_dir / "tracks" / track_key / "history"
        if not hist_dir.exists():
            hist_dir = self.raw_data_dir / "tracks" / track_key / "historical"
        if not hist_dir.exists():
            hist_dir = self.processed_data_dir

        r_path = hist_dir / f"{track_key}_historical_races.csv"
        q_path = hist_dir / f"{track_key}_historical_qualifying.csv"

        df_r = pd.read_csv(r_path) if r_path.exists() else None
        df_q = pd.read_csv(q_path) if q_path.exists() else None
        return df_r, df_q

