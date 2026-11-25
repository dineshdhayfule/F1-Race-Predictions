"""
ApexGrid — Historical Session Results Service (Option 1)
=========================================================
Retrieves and structures classified race results, qualifying classifications,
sprint classifications, and practice session timings across Formula 1 history (1950–2026).
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import pandas as pd

from src.historical.api_client import HistoricalAPIClient
from src.historical.classification_service import HistoricalClassificationService
from src.historical.qualifying_service import HistoricalQualifyingService
from src.historical.models import RaceResultEntry, QualifyingResultEntry, SprintResultEntry
from src.data.data_loader import DataLoader
from src.utils.logging_utils import get_logger

logger = get_logger("historical_session_results")


def _format_lap_str(val: Any) -> Optional[str]:
    """Helper to convert raw lap time/timedelta strings to clean format (e.g. 1:28.723)."""
    if val is None or pd.isna(val):
        return None
    s = str(val).strip()
    if not s or s.lower() in ("nan", "nat", "none"):
        return None
    # Handle timedelta formats like '0 days 00:01:27.869000'
    if "days" in s:
        parts = s.split()
        if len(parts) >= 3:
            s = parts[-1]
    if s.startswith("00:"):
        s = s[3:]
    if s.startswith("0") and len(s) > 1 and s[1].isdigit():
        s = s[1:]
    if "." in s:
        base, frac = s.split(".", 1)
        if len(frac) > 3:
            frac = frac[:3]
        s = f"{base}.{frac}"
    return s


class SessionResultsService:
    """Service for retrieving official session finishing classifications and telemetry."""

    def __init__(
        self,
        api_client: Optional[HistoricalAPIClient] = None,
        data_loader: Optional[DataLoader] = None,
        root_dir: Optional[Path] = None,
    ):
        self.api_client = api_client or HistoricalAPIClient()
        self.classification_service = HistoricalClassificationService(api_client=self.api_client)
        self.qualifying_service = HistoricalQualifyingService(api_client=self.api_client)
        self.data_loader = data_loader or DataLoader()
        self.root_dir = Path(root_dir) if root_dir else Path(__file__).resolve().parents[3]

    def is_sprint_round(self, season: int, round_num: int) -> bool:
        """
        Determines dynamically whether a round was or is a Sprint weekend.
        Sprint weekends exist strictly from 2021 onwards.
        Checks local season ground truth files first, then DataLoader, then API.
        """
        if season < 2021:
            return False

        # 1. Check local ground-truth files
        season_dirs = [
            self.root_dir / "data" / "raw" / "season" / str(season),
            self.root_dir / "data" / "processed" / "season" / str(season),
        ]
        for sdir in season_dirs:
            if sdir.exists():
                for pattern in (
                    f"{season}_round_{round_num}_sprint_race.csv",
                    f"{season}_round_{round_num}_sprint_qualifying.csv",
                    f"{season}_round_{round_num}_sq.csv",
                ):
                    if (sdir / pattern).exists():
                        return True

        # 2. Check DataLoader historical sprint round detection
        try:
            if self.data_loader.is_historical_sprint_round(season, round_num):
                return True
        except Exception:
            pass

        # 3. Check API for 2021+ historical seasons
        endpoint = f"{season}/{round_num}/sprint"
        try:
            response = self.api_client.get_json(endpoint)
            races = response.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            if races and races[0].get("SprintResults"):
                return True
        except Exception:
            pass

        return False

    def get_race_results(self, season: int, round_num: int) -> List[RaceResultEntry]:
        """
        Retrieves official Grand Prix race finishing classification.
        Falls back to local season CSV files if provider API returns empty.
        """
        results = self.classification_service.get_race_classification(season, round_num)
        if results:
            return results

        # Fallback to local file (e.g. for simulated season 2026 or offline)
        local_file = self.root_dir / "data" / "raw" / "season" / str(season) / f"{season}_round_{round_num}_race.csv"
        if not local_file.exists():
            local_file = self.root_dir / "data" / "processed" / "season" / str(season) / f"{season}_round_{round_num}_race.csv"

        if local_file.exists():
            return self._parse_local_race_file(local_file)

        return []

    def get_qualifying_results(self, season: int, round_num: int) -> List[QualifyingResultEntry]:
        """
        Retrieves official qualifying results.
        Falls back to local season CSV files if provider API returns empty.
        """
        results = self.qualifying_service.get_round_qualifying(season, round_num)
        if results:
            return results

        local_file = self.root_dir / "data" / "raw" / "season" / str(season) / f"{season}_round_{round_num}_qualifying.csv"
        if not local_file.exists():
            local_file = self.root_dir / "data" / "processed" / "season" / str(season) / f"{season}_round_{round_num}_qualifying.csv"

        if local_file.exists():
            return self._parse_local_qualifying_file(local_file, is_sprint_quali=False)

        return []

    def get_sprint_results(self, season: int, round_num: int) -> List[SprintResultEntry]:
        """
        Retrieves official sprint session classification for sprint weekends (2021+).
        Returns empty list if no sprint was held.
        """
        if season < 2021:
            return []

        endpoint = f"{season}/{round_num}/sprint"
        try:
            response = self.api_client.get_json(endpoint)
            mrdata = response.get("MRData", {})
            race_table = mrdata.get("RaceTable", {})
            races = race_table.get("Races", [])
            if races:
                sprint_results_raw = races[0].get("SprintResults", [])
                if sprint_results_raw:
                    results: List[SprintResultEntry] = []
                    for item in sprint_results_raw:
                        pos_text = str(item.get("positionText", "")).strip()
                        try:
                            pos = int(item.get("position", 0))
                        except (ValueError, TypeError):
                            pos = None

                        drv = item.get("Driver", {})
                        driver_id = str(drv.get("driverId", "")).strip().lower()
                        given = drv.get("givenName", "")
                        family = drv.get("familyName", "")
                        driver_name = f"{given} {family}".strip() or driver_id

                        constructor = item.get("Constructor", {})
                        constructor_name = str(constructor.get("name", "")).strip() or "Unknown"

                        try:
                            grid = int(item.get("grid", 0))
                        except (ValueError, TypeError):
                            grid = None

                        try:
                            laps = int(item.get("laps", 0))
                        except (ValueError, TypeError):
                            laps = 0

                        try:
                            points = float(item.get("points", 0.0))
                        except (ValueError, TypeError):
                            points = 0.0

                        status = str(item.get("status", "Unknown"))
                        time_obj = item.get("Time", {})
                        time_str = time_obj.get("time") if isinstance(time_obj, dict) else None

                        results.append(
                            SprintResultEntry(
                                position=pos,
                                position_text=pos_text,
                                driver_id=driver_id,
                                driver_name=driver_name,
                                constructor_name=constructor_name,
                                grid_position=grid,
                                laps_completed=laps,
                                points=points,
                                status=status,
                                race_time_or_gap=time_str,
                            )
                        )
                    return results
        except Exception as e:
            logger.debug(f"API sprint retrieval note for {season} R{round_num}: {e}")

        # Fallback to local sprint race file
        local_file = self.root_dir / "data" / "raw" / "season" / str(season) / f"{season}_round_{round_num}_sprint_race.csv"
        if not local_file.exists():
            local_file = self.root_dir / "data" / "processed" / "season" / str(season) / f"{season}_round_{round_num}_sprint_race.csv"

        if local_file.exists():
            return self._parse_local_sprint_race_file(local_file)

        return []

    def get_sprint_qualifying_results(self, season: int, round_num: int) -> List[QualifyingResultEntry]:
        """
        Retrieves official sprint qualifying (SQ / Shootout) classification for sprint weekends.
        Checks local telemetry files first, then API.
        """
        if season < 2021:
            return []

        # Check local season directories
        candidate_names = (
            f"{season}_round_{round_num}_sprint_qualifying.csv",
            f"{season}_round_{round_num}_sprint_shootout.csv",
            f"{season}_round_{round_num}_sq.csv",
            f"{season}_round_{round_num}_sprint_quali.csv",
        )
        season_dirs = [
            self.root_dir / "data" / "raw" / "season" / str(season),
            self.root_dir / "data" / "processed" / "season" / str(season),
        ]
        for sdir in season_dirs:
            if sdir.exists():
                for cname in candidate_names:
                    target = sdir / cname
                    if target.exists():
                        return self._parse_local_qualifying_file(target, is_sprint_quali=True)

        return []

    def get_practice_classification(
        self,
        season: int,
        round_num: int,
        session_type: str = "fp1"
    ) -> List[Dict[str, Any]]:
        """
        Retrieves practice session classification for modern seasons (2018–2026).
        Returns list of driver lap classifications (fastest lap, lap count, delta).
        """
        stype = str(session_type).lower().strip()
        if season < 2018:
            return []

        df: Optional[pd.DataFrame] = None
        try:
            sessions_by_round = self.data_loader.load_all_practice_sessions(season, [round_num])
            round_sessions = sessions_by_round.get(round_num, {})
            df = round_sessions.get(stype)
        except Exception as e:
            logger.debug(f"DataLoader practice fetch note: {e}")

        # Check direct track folder if dataframe is empty
        if df is None or df.empty:
            local_season_file = self.root_dir / "data" / "raw" / "season" / str(season) / f"{season}_round_{round_num}_{stype}.csv"
            if local_season_file.exists():
                try:
                    df = pd.read_csv(local_season_file)
                except Exception:
                    df = None

        if df is None or df.empty:
            return []

        driver_col = "Driver" if "Driver" in df.columns else ("Abbreviation" if "Abbreviation" in df.columns else None)
        time_col = "LapTimeSeconds" if "LapTimeSeconds" in df.columns else ("LapTime" if "LapTime" in df.columns else None)
        if not driver_col or not time_col:
            return []

        valid_laps = df.dropna(subset=[driver_col, time_col]).copy()
        if valid_laps.empty:
            return []

        # If LapTime is timedelta or string, convert
        if not pd.api.types.is_numeric_dtype(valid_laps[time_col]):
            if pd.api.types.is_timedelta64_dtype(valid_laps[time_col]):
                valid_laps["sec"] = valid_laps[time_col].dt.total_seconds()
            else:
                # If string lap time e.g. "1:27.869"
                valid_laps["sec"] = valid_laps[time_col].apply(self._parse_str_to_seconds)
        else:
            valid_laps["sec"] = valid_laps[time_col]

        valid_laps = valid_laps.dropna(subset=["sec"])
        if valid_laps.empty:
            return []

        driver_best = valid_laps.groupby(driver_col).agg(
            best_lap=("sec", "min"),
            total_laps=("sec", "count"),
        ).reset_index()

        driver_best = driver_best.sort_values("best_lap").reset_index(drop=True)
        if driver_best.empty:
            return []
            
        p1_time = driver_best["best_lap"].iloc[0]

        rows = []
        for rank, row in driver_best.iterrows():
            b_time = row["best_lap"]
            gap = b_time - p1_time
            minutes = int(b_time // 60)
            seconds = b_time % 60
            time_fmt = f"{minutes}:{seconds:06.3f}" if minutes > 0 else f"{seconds:06.3f}"
            gap_fmt = f"+{gap:.3f}s" if rank > 0 else "P1"

            rows.append({
                "position": rank + 1,
                "driver": str(row[driver_col]),
                "best_lap_seconds": b_time,
                "best_lap_time": time_fmt,
                "gap": gap_fmt,
                "laps_completed": int(row["total_laps"]),
            })
        return rows

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    @staticmethod
    def _parse_str_to_seconds(val: Any) -> Optional[float]:
        """Parses lap time strings ('1:27.869' or '87.869') into seconds."""
        if val is None or pd.isna(val):
            return None
        s = str(val).strip()
        if not s or s.lower() in ("nan", "nat", "none"):
            return None
        try:
            if ":" in s:
                parts = s.split(":")
                mins = float(parts[0])
                secs = float(parts[1])
                return mins * 60.0 + secs
            return float(s)
        except Exception:
            return None

    def _parse_local_race_file(self, path: Path) -> List[RaceResultEntry]:
        """Parses a local race CSV into RaceResultEntry objects."""
        try:
            df = pd.read_csv(path)
            if "Position" in df.columns:
                df = df.sort_values("Position")
            entries: List[RaceResultEntry] = []
            for idx, row in df.iterrows():
                pos = int(row["Position"]) if pd.notna(row.get("Position")) else None
                pos_text = str(pos) if pos is not None else str(row.get("ClassifiedPosition") or "DNF")
                drv_id = str(row.get("DriverId") or row.get("Abbreviation") or "unknown").lower()
                drv_name = str(row.get("FullName") or row.get("BroadcastName") or drv_id)
                num = str(int(row["DriverNumber"])) if pd.notna(row.get("DriverNumber")) else None
                c_id = str(row.get("TeamId") or row.get("TeamName") or "unknown").lower()
                c_name = str(row.get("TeamName") or "Unknown")
                grid = int(row["GridPosition"]) if pd.notna(row.get("GridPosition")) else None
                raw_grid = str(grid) if grid is not None else ""
                gained = (grid - pos) if (grid is not None and pos is not None) else None
                t_str = str(row["Time"]) if pd.notna(row.get("Time")) else None
                laps = int(row["Laps"]) if pd.notna(row.get("Laps")) else 0
                pts = float(row["Points"]) if pd.notna(row.get("Points")) else 0.0
                status = str(row["Status"]) if pd.notna(row.get("Status")) else "Finished"

                entries.append(
                    RaceResultEntry(
                        position=pos,
                        position_text=pos_text,
                        driver_id=drv_id,
                        driver_name=drv_name,
                        driver_number=num,
                        constructor_id=c_id,
                        constructor_name=c_name,
                        grid_position=grid,
                        raw_grid_value=raw_grid,
                        positions_gained=gained,
                        race_time_or_gap=t_str,
                        laps_completed=laps,
                        points=pts,
                        status=status,
                    )
                )
            return entries
        except Exception as e:
            logger.warning(f"Error parsing local race CSV {path}: {e}")
            return []

    def _parse_local_sprint_race_file(self, path: Path) -> List[SprintResultEntry]:
        """Parses a local sprint race CSV into SprintResultEntry objects."""
        try:
            df = pd.read_csv(path)
            if "Position" in df.columns:
                df = df.sort_values("Position")
            entries: List[SprintResultEntry] = []
            for idx, row in df.iterrows():
                pos = int(row["Position"]) if pd.notna(row.get("Position")) else None
                pos_text = str(pos) if pos is not None else str(row.get("ClassifiedPosition") or "DNF")
                drv_id = str(row.get("DriverId") or row.get("Abbreviation") or "unknown").lower()
                drv_name = str(row.get("FullName") or row.get("BroadcastName") or drv_id)
                c_name = str(row.get("TeamName") or "Unknown")
                grid = int(row["GridPosition"]) if pd.notna(row.get("GridPosition")) else None
                laps = int(row["Laps"]) if pd.notna(row.get("Laps")) else 0
                pts = float(row["Points"]) if pd.notna(row.get("Points")) else 0.0
                status = str(row["Status"]) if pd.notna(row.get("Status")) else "Finished"
                t_str = str(row["Time"]) if pd.notna(row.get("Time")) else None

                entries.append(
                    SprintResultEntry(
                        position=pos,
                        position_text=pos_text,
                        driver_id=drv_id,
                        driver_name=drv_name,
                        constructor_name=c_name,
                        grid_position=grid,
                        laps_completed=laps,
                        points=pts,
                        status=status,
                        race_time_or_gap=t_str,
                    )
                )
            return entries
        except Exception as e:
            logger.warning(f"Error parsing local sprint race CSV {path}: {e}")
            return []

    def _parse_local_qualifying_file(self, path: Path, is_sprint_quali: bool = False) -> List[QualifyingResultEntry]:
        """Parses a local qualifying or sprint qualifying CSV into QualifyingResultEntry objects."""
        try:
            df = pd.read_csv(path)
            if "Position" in df.columns:
                df = df.sort_values("Position")
            entries: List[QualifyingResultEntry] = []
            fmt = "Sprint Qualifying (SQ)" if is_sprint_quali else "Knockout"
            code = "SQ" if is_sprint_quali else "Q"

            for idx, row in df.iterrows():
                pos = int(row["Position"]) if pd.notna(row.get("Position")) else idx + 1
                drv_id = str(row.get("DriverId") or row.get("Abbreviation") or "unknown").lower()
                drv_name = str(row.get("FullName") or row.get("BroadcastName") or drv_id)
                drv_code = str(row.get("Abbreviation") or "")
                c_id = str(row.get("TeamId") or row.get("TeamName") or "unknown").lower()
                c_name = str(row.get("TeamName") or "Unknown")

                q1 = _format_lap_str(row.get("Q1"))
                q2 = _format_lap_str(row.get("Q2"))
                q3 = _format_lap_str(row.get("Q3"))
                best_lap = q3 or q2 or q1 or _format_lap_str(row.get("Time"))

                entries.append(
                    QualifyingResultEntry(
                        position=pos,
                        driver_id=drv_id,
                        driver_name=drv_name,
                        driver_code=drv_code if drv_code else None,
                        constructor_id=c_id,
                        constructor_name=c_name,
                        q1=q1,
                        q2=q2,
                        q3=q3,
                        best_lap_time=best_lap,
                        gap_to_pole=None,  # Computed below
                        session_format=fmt,
                        session_code=code,
                    )
                )

            # Compute gap to pole
            if entries and entries[0].best_lap_time:
                p1_sec = self._parse_str_to_seconds(entries[0].best_lap_time)
                if p1_sec:
                    updated_entries = []
                    for e in entries:
                        if e.position == 1:
                            updated_entries.append(e)
                        else:
                            e_sec = self._parse_str_to_seconds(e.best_lap_time)
                            gap_str = f"+{(e_sec - p1_sec):.3f}" if (e_sec and e_sec >= p1_sec) else None
                            updated_entries.append(
                                QualifyingResultEntry(
                                    position=e.position,
                                    driver_id=e.driver_id,
                                    driver_name=e.driver_name,
                                    driver_code=e.driver_code,
                                    constructor_id=e.constructor_id,
                                    constructor_name=e.constructor_name,
                                    q1=e.q1,
                                    q2=e.q2,
                                    q3=e.q3,
                                    best_lap_time=e.best_lap_time,
                                    gap_to_pole=gap_str,
                                    session_format=e.session_format,
                                    session_code=e.session_code,
                                )
                            )
                    return updated_entries

            return entries
        except Exception as e:
            logger.warning(f"Error parsing local qualifying CSV {path}: {e}")
            return []
