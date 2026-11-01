"""
F1 Season Discovery Module
===========================
Discovers, validates, and categorizes championship seasons from actual project datasets.
Enforces FastF1 historical coverage boundaries (>= 2018 for detailed session telemetry vs
< 2018 for results-only data).
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger("season_discovery")

FASTF1_MIN_YEAR = 2018

SOURCE_FASTF1_DETAILED = "fastf1_detailed"
SOURCE_RESULTS_ONLY = "results_only"
SOURCE_INCOMPLETE = "incomplete"
SOURCE_NONE = "none"


class SeasonDiscovery:
    """
    Dynamically discovers and validates available championship seasons
    from disk without hardcoded ranges or theoretical API assumptions.
    """

    def __init__(self, root_dir: Optional[Path] = None):
        self.root_dir = Path(root_dir) if root_dir else Path(__file__).resolve().parents[2]
        self.raw_season_dir = self.root_dir / "data" / "raw" / "season"
        self.proc_season_dir = self.root_dir / "data" / "processed" / "season"

    def get_season_dir(self, year: int) -> Optional[Path]:
        """Returns the existing raw or processed directory for a season, or None."""
        raw_p = self.raw_season_dir / str(year)
        if raw_p.exists() and raw_p.is_dir():
            return raw_p
        proc_p = self.proc_season_dir / str(year)
        if proc_p.exists() and proc_p.is_dir():
            return proc_p
        return None

    def get_season_status(self, year: int, workflow: str = "prediction") -> Dict[str, Any]:
        """
        Inspects dataset integrity, data source, and round completeness for a specific season.
        """
        s_dir = self.get_season_dir(year)
        if s_dir is None:
            return {
                "year": year,
                "exists": False,
                "usable": False,
                "source_type": SOURCE_NONE,
                "completed_rounds": [],
                "total_files": 0,
                "valid_files": 0,
                "error_reason": f"No season directory found in {self.raw_season_dir} or {self.proc_season_dir}",
            }

        files = [p for p in s_dir.glob("*.csv") if p.is_file()]
        if not files:
            return {
                "year": year,
                "exists": True,
                "usable": False,
                "source_type": SOURCE_INCOMPLETE,
                "completed_rounds": [],
                "total_files": 0,
                "valid_files": 0,
                "error_reason": f"Season directory exists but contains zero CSV files: {s_dir}",
            }

        valid_files = [p for p in files if p.stat().st_size > 0]
        if not valid_files:
            return {
                "year": year,
                "exists": True,
                "usable": False,
                "source_type": SOURCE_INCOMPLETE,
                "completed_rounds": [],
                "total_files": len(files),
                "valid_files": 0,
                "error_reason": f"All {len(files)} CSV files in {s_dir} are 0 bytes (empty).",
            }

        # Find rounds with both valid race and qualifying results
        race_files = {
            int(p.stem.split("_")[2]): p
            for p in valid_files
            if "_race" in p.stem and len(p.stem.split("_")) >= 3 and p.stem.split("_")[2].isdigit()
        }
        quali_files = {
            int(p.stem.split("_")[2]): p
            for p in valid_files
            if "_qualifying" in p.stem and len(p.stem.split("_")) >= 3 and p.stem.split("_")[2].isdigit()
        }

        completed_rounds = sorted(list(set(race_files.keys()) & set(quali_files.keys())))

        # Validate that completed rounds have actual data rows
        verified_rounds = []
        for r in completed_rounds:
            rf = race_files[r]
            qf = quali_files[r]
            try:
                df_r = pd.read_csv(rf, nrows=2)
                df_q = pd.read_csv(qf, nrows=2)
                if len(df_r) > 0 and len(df_q) > 0:
                    verified_rounds.append(r)
            except Exception as e:
                logger.warning(f"Failed to read CSV for season {year} round {r}: {e}")

        # Classify data source type
        if year >= FASTF1_MIN_YEAR:
            source_type = SOURCE_FASTF1_DETAILED
        else:
            source_type = SOURCE_RESULTS_ONLY

        if not verified_rounds:
            return {
                "year": year,
                "exists": True,
                "usable": False,
                "source_type": SOURCE_INCOMPLETE if year >= FASTF1_MIN_YEAR else SOURCE_RESULTS_ONLY,
                "completed_rounds": [],
                "total_files": len(files),
                "valid_files": len(valid_files),
                "error_reason": f"No rounds in season {year} have both valid race and qualifying CSV files with data rows.",
            }

        # Check workflow compatibility
        if workflow == "prediction":
            # Prediction requires FastF1 detailed session data (>= 2018) and at least 1 completed round
            if year < FASTF1_MIN_YEAR:
                return {
                    "year": year,
                    "exists": True,
                    "usable": False,
                    "source_type": SOURCE_RESULTS_ONLY,
                    "completed_rounds": verified_rounds,
                    "total_files": len(files),
                    "valid_files": len(valid_files),
                    "error_reason": f"Season {year} is prior to FastF1 detailed coverage ({FASTF1_MIN_YEAR}). Results-only data is not usable for telemetry-based prediction.",
                }
            usable = len(verified_rounds) >= 1
            error_reason = None if usable else "Prediction workflow requires at least 1 completed round."
        else:
            usable = len(verified_rounds) >= 1
            error_reason = None if usable else "At least 1 completed round is required."

        return {
            "year": year,
            "exists": True,
            "usable": usable,
            "source_type": source_type,
            "completed_rounds": verified_rounds,
            "total_files": len(files),
            "valid_files": len(valid_files),
            "error_reason": error_reason,
        }

    def discover_seasons(self, workflow: str = "prediction") -> List[int]:
        """
        Discovers and validates all championship seasons available on disk
        that meet the criteria for the specified workflow.
        """
        candidate_years = set()
        for s_dir in [self.raw_season_dir, self.proc_season_dir]:
            if s_dir.exists() and s_dir.is_dir():
                for p in s_dir.iterdir():
                    if p.is_dir() and p.name.isdigit() and len(p.name) == 4:
                        candidate_years.add(int(p.name))

        usable_seasons = []
        for yr in sorted(candidate_years):
            status = self.get_season_status(yr, workflow=workflow)
            if status["usable"]:
                usable_seasons.append(yr)

        return sorted(usable_seasons)

    @staticmethod
    def is_contiguous(seasons: List[int]) -> bool:
        """Determines if a list of seasons forms an unbroken sequence."""
        if len(seasons) <= 1:
            return True
        s = sorted(seasons)
        return s == list(range(s[0], s[-1] + 1))

    @classmethod
    def format_available_seasons(cls, seasons: List[int]) -> str:
        """
        Formats a list of available seasons into a human-readable string:
          - [] -> "None available"
          - [2026] -> "2026"
          - [2022, 2023, 2024, 2025, 2026] -> "2022–2026"
          - [2018, 2022, 2026] -> "2018, 2022, 2026"
        """
        if not seasons:
            return "None available"
        s = sorted(seasons)
        if len(s) == 1:
            return str(s[0])
        if cls.is_contiguous(s):
            return f"{s[0]}–{s[-1]}"
        return ", ".join(str(yr) for yr in s)

    @classmethod
    def format_prompt_hint(cls, seasons: List[int]) -> str:
        """
        Formats the range hint for a prompt:
          - [] -> ""
          - [2026] -> ""  (clean single season, avoids '(2026-2026)')
          - [2018..2026] -> " (2018-2026)"
          - [2018, 2022, 2026] -> " (2018, 2022, 2026)"
        """
        if not seasons:
            return ""
        s = sorted(seasons)
        if len(s) == 1:
            return ""
        if cls.is_contiguous(s):
            return f" ({s[0]}-{s[-1]})"
        return f" ({', '.join(str(yr) for yr in s)})"

    @classmethod
    def get_default_season(cls, seasons: List[int], preferred: int = 2026) -> Optional[int]:
        """
        Selects the default season:
        Preserves preferred (2026) if available; otherwise returns the latest available.
        """
        if not seasons:
            return None
        if preferred in seasons:
            return preferred
        return max(seasons)

