"""
Authoritative F1 Calendar Service
=================================
Retrieves, normalizes, and caches Formula 1 championship schedules using FastF1.
Provides dynamic, calendar-driven season and Grand Prix resolution without
relying on hardcoded 2026 calendar assumptions or silent year substitution.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import fastf1
import pandas as pd

from src.utils.circuit_mapping import resolve_circuit_key, load_tracks_config
from src.utils.logging_utils import get_logger

logger = get_logger("calendar_service")

MIN_CHAMPIONSHIP_YEAR = 1950
MAX_CHAMPIONSHIP_YEAR = 2050


class CalendarRetrievalError(Exception):
    """Raised when FastF1 schedule retrieval fails due to network, API, or system errors."""
    pass


class CalendarEmptyError(CalendarRetrievalError):
    """Raised when FastF1 returns an empty schedule or no championship rounds."""
    pass


@dataclass
class CalendarEvent:
    """Normalized representation of a single Grand Prix event in an F1 championship season."""
    year: int
    round_number: int
    event_name: str
    official_event_name: str
    country: str
    location: str
    circuit_name: str
    circuit_key: Optional[str]
    event_date: Optional[str]
    event_format: str
    is_sprint: bool
    sessions: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CalendarService:
    """
    Centralized service for F1 championship calendar retrieval and normalization.
    Decoupled from CLI presentation and predictive pipeline internals.
    """

    def __init__(
        self,
        root_dir: Optional[Path] = None,
        enable_cache: bool = False,
        cache_dir: Optional[Path] = None,
    ):
        self.root_dir = Path(root_dir) if root_dir else Path(__file__).resolve().parents[2]
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir or (self.root_dir / "data" / "cache")
        self._memory_cache: Dict[int, List[CalendarEvent]] = {}
        self._fastf1_cache_initialized = False

    def _ensure_fastf1_cache(self) -> None:
        if self.enable_cache and not self._fastf1_cache_initialized:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                fastf1.Cache.enable_cache(str(self.cache_dir))
                self._fastf1_cache_initialized = True
            except Exception as e:
                logger.debug(f"FastF1 cache initialization notice: {e}")

    def clear_cache(self) -> None:
        """Clears in-memory cached calendars."""
        self._memory_cache.clear()

    def get_calendar(self, year: int, force_refresh: bool = False) -> List[CalendarEvent]:
        """
        Retrieves and normalizes the full championship calendar for a season.

        Parameters
        ----------
        year : int
            Championship year (e.g. 2024, 2026).
        force_refresh : bool, default False
            If True, bypasses memory cache and re-fetches from FastF1.

        Returns
        -------
        List[CalendarEvent]
            Chronological list of championship Grand Prix events (positive rounds).

        Raises
        ------
        ValueError
            If year is outside valid championship history (1950-2050).
        CalendarRetrievalError
            If FastF1 schedule retrieval fails.
        """
        if not isinstance(year, int):
            try:
                year = int(year)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid championship year: {year}. Year must be an integer.")

        if year < MIN_CHAMPIONSHIP_YEAR or year > MAX_CHAMPIONSHIP_YEAR:
            raise ValueError(
                f"Year {year} is out of valid championship range ({MIN_CHAMPIONSHIP_YEAR}–{MAX_CHAMPIONSHIP_YEAR})."
            )

        if not force_refresh and year in self._memory_cache:
            logger.debug(f"Using in-memory cached calendar for season {year}.")
            return self._memory_cache[year]

        self._ensure_fastf1_cache()
        logger.info(f"Retrieving FastF1 event schedule for season {year}...")
        try:
            raw_schedule = fastf1.get_event_schedule(year)
        except Exception as e:
            err_msg = f"Failed to retrieve FastF1 schedule for season {year}: {e}"
            logger.error(err_msg)
            raise CalendarRetrievalError(err_msg) from e

        if raw_schedule is None or (isinstance(raw_schedule, pd.DataFrame) and raw_schedule.empty):
            logger.warning(f"FastF1 returned empty schedule for season {year}.")
            self._memory_cache[year] = []
            return []

        events = self._normalize_schedule(year, raw_schedule)
        self._memory_cache[year] = events
        return events

    def get_event_by_round(self, year: int, round_num: int) -> Optional[CalendarEvent]:
        """Looks up a specific Grand Prix event by season year and round number."""
        calendar = self.get_calendar(year)
        for ev in calendar:
            if ev.round_number == round_num:
                return ev
        return None

    def get_event_by_circuit(self, year: int, circuit_identifier: str) -> Optional[CalendarEvent]:
        """
        Looks up a Grand Prix event in a season by circuit identifier,
        name pattern, display name, or alias.
        """
        canon_key = resolve_circuit_key(circuit_identifier)
        calendar = self.get_calendar(year)
        for ev in calendar:
            if ev.circuit_key and canon_key and ev.circuit_key == canon_key:
                return ev
            if canon_key and resolve_circuit_key(ev.event_name) == canon_key:
                return ev
            if circuit_identifier.lower() in ev.event_name.lower():
                return ev
            if circuit_identifier.lower() in ev.location.lower():
                return ev
        return None

    def _normalize_schedule(self, year: int, schedule: pd.DataFrame) -> List[CalendarEvent]:
        """
        Normalizes a raw FastF1 schedule DataFrame into a list of CalendarEvent objects.
        Filters out non-championship testing sessions (round <= 0).
        """
        if "RoundNumber" not in schedule.columns:
            logger.warning(f"Schedule for {year} does not contain 'RoundNumber' column.")
            return []

        # Filter strictly positive championship rounds
        valid_rounds_df = schedule.copy()
        valid_rounds_df["RoundNumber"] = pd.to_numeric(valid_rounds_df["RoundNumber"], errors="coerce")
        championship_df = valid_rounds_df[valid_rounds_df["RoundNumber"] > 0].copy()

        if championship_df.empty:
            logger.warning(f"No positive championship rounds found in schedule for season {year}.")
            return []

        championship_df = championship_df.sort_values(by="RoundNumber")

        tracks_cfg = load_tracks_config()
        normalized_events: List[CalendarEvent] = []

        for _, row in championship_df.iterrows():
            r_num = int(row["RoundNumber"])
            ev_name = str(row.get("EventName", f"Round {r_num}"))
            off_name = str(row.get("OfficialEventName", ev_name))
            country = str(row.get("Country", ""))
            location = str(row.get("Location", ""))
            fmt_raw = str(row.get("EventFormat", "")).strip().lower()

            # Event Date
            raw_date = row.get("EventDate")
            date_str = None
            if pd.notna(raw_date):
                if isinstance(raw_date, (datetime, date, pd.Timestamp)):
                    date_str = pd.Timestamp(raw_date).strftime("%Y-%m-%d")
                else:
                    date_str = str(raw_date).split(" ")[0]

            # Detect sessions (Session1 through Session5)
            sessions_dict = {}
            session_names = []
            for i in range(1, 6):
                s_col = f"Session{i}"
                if s_col in row and pd.notna(row[s_col]):
                    s_name = str(row[s_col]).strip()
                    if s_name:
                        session_names.append(s_name)
                        date_col = f"Session{i}DateUtc" if f"Session{i}DateUtc" in row else f"Session{i}Date"
                        s_date = str(row.get(date_col, "")) if pd.notna(row.get(date_col)) else ""
                        sessions_dict[s_name] = s_date

            # Sprint format detection
            is_sprint = (
                "sprint" in fmt_raw
                or any("sprint" in s.lower() for s in session_names)
            )

            # Resolve canonical circuit key
            circuit_key = self._resolve_event_circuit_key(
                event_name=ev_name,
                official_name=off_name,
                country=country,
                location=location,
                tracks_cfg=tracks_cfg,
            )

            # Determine circuit name
            circuit_name = ""
            if circuit_key and circuit_key in tracks_cfg:
                circuit_name = tracks_cfg[circuit_key].get("circuit_name", "")
            if not circuit_name:
                circuit_name = location or country or ev_name

            event = CalendarEvent(
                year=year,
                round_number=r_num,
                event_name=ev_name,
                official_event_name=off_name,
                country=country,
                location=location,
                circuit_name=circuit_name,
                circuit_key=circuit_key,
                event_date=date_str,
                event_format="sprint" if is_sprint else (fmt_raw or "conventional"),
                is_sprint=is_sprint,
                sessions=sessions_dict,
            )
            normalized_events.append(event)

        return normalized_events

    def _resolve_event_circuit_key(
        self,
        event_name: str,
        official_name: str,
        country: str,
        location: str,
        tracks_cfg: Dict[str, Any],
    ) -> Optional[str]:
        """
        Resolves the canonical circuit key for a schedule event using
        multiple heuristics (EventName, Location, Country, OfficialEventName).
        """
        # 1. Direct resolution via event name
        key = resolve_circuit_key(event_name)
        if key:
            return key

        # 2. Location
        if location:
            key = resolve_circuit_key(location)
            if key:
                return key

        # 3. Country
        if country:
            key = resolve_circuit_key(country)
            if key:
                return key

        # 4. Official event name
        if official_name:
            key = resolve_circuit_key(official_name)
            if key:
                return key

        # 5. Fuzzy match against tracks.json display_name and event_name_patterns
        search_corpus = f"{event_name} {official_name} {location} {country}".lower()
        for k, info in tracks_cfg.items():
            disp = str(info.get("display_name", "")).lower()
            if disp and disp in search_corpus:
                return k
            for pat in info.get("event_name_patterns", []):
                pat_str = str(pat).lower()
                if pat_str and pat_str in search_corpus:
                    return k

        return None
