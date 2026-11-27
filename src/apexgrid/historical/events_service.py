"""
ApexGrid — Historical Events Service
=====================================
Dedicated service for retrieving Formula 1 season calendars and event metadata (1950–2026).
Retrieves data dynamically via HistoricalAPIClient from Jolpica's Ergast-compatible API.
Enforces:
- Separation of completed events from upcoming/pending events.
- Clean handling of cancelled, rescheduled, or incomplete rounds.
- Missing provider fields represented as None (zero fabrication).
- Dynamic round counts without hardcoding.
- In-memory processing with zero filesystem persistence.
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

from apexgrid.historical.api_client import HistoricalAPIClient
from apexgrid.historical.models import HistoricalEvent, SeasonCalendar
from apexgrid.utils.logging_utils import get_logger

logger = get_logger("historical_events_service")


class HistoricalEventsService:
    """
    Fetch-only service for Formula 1 seasons, calendars, and Grand Prix events.
    All data is retrieved on-demand and retained strictly in process memory.
    """

    def __init__(self, api_client: Optional[HistoricalAPIClient] = None):
        self.api_client = api_client or HistoricalAPIClient()

    def get_season_calendar(
        self,
        year: int,
        ref_date: Optional[datetime] = None,
        bypass_cache: bool = False,
    ) -> SeasonCalendar:
        """
        Retrieves the complete event calendar for a given season.

        Args:
            year: Championship season year (1950–2026).
            ref_date: Reference datetime for evaluating completed vs upcoming events.
                      Defaults to current UTC time.
            bypass_cache: If True, forces fresh retrieval from API.

        Returns:
            SeasonCalendar: Structured calendar with event progression metadata.
        """
        now_utc = ref_date or datetime.now(timezone.utc)
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)

        # Jolpica endpoint: /{year}.json
        endpoint = f"{year}"
        try:
            response = self.api_client.get_json(endpoint, bypass_cache=bypass_cache)
        except Exception as e:
            logger.warning(f"Failed to retrieve season calendar for {year}: {e}")
            return SeasonCalendar(
                year=year,
                total_rounds=0,
                completed_rounds=0,
                upcoming_rounds=0,
                is_in_progress=False,
                events=[],
                status_display=f"Season {year} (Data Retrieval Error: {e})",
            )

        mrdata = response.get("MRData", {})
        race_table = mrdata.get("RaceTable", {})
        raw_races = race_table.get("Races", [])

        if not raw_races:
            return SeasonCalendar(
                year=year,
                total_rounds=0,
                completed_rounds=0,
                upcoming_rounds=0,
                is_in_progress=False,
                events=[],
                status_display=f"Season {year} (Calendar Pending / No Events Recorded)",
            )

        events: List[HistoricalEvent] = []
        past_scheduled_count = 0
        upcoming_count = 0

        for race in raw_races:
            event = self._parse_race_entry(race, year, now_utc)
            events.append(event)
            if event.status_label == "Scheduled (Past Date)":
                past_scheduled_count += 1
            elif event.is_upcoming:
                upcoming_count += 1

        total_rounds = len(events)
        # An in-progress season has upcoming events remaining, or is currently underway in current year
        is_in_progress = (upcoming_count > 0 and (past_scheduled_count > 0 or year == now_utc.year))

        # Calendar completeness != race completion
        # Calendar endpoint cannot confirm race completion; completed_rounds is strictly 0 for calendar queries
        if is_in_progress:
            status_display = f"Season {year} Calendar (In-Progress: {past_scheduled_count} Dates Elapsed, {upcoming_count} Upcoming of {total_rounds} Scheduled)"
        elif upcoming_count == 0 and past_scheduled_count == total_rounds and total_rounds > 0:
            status_display = f"Season {year} Calendar (Schedule Concluded: {total_rounds} Scheduled Dates Elapsed; Results Required to Confirm Completion)"
        elif upcoming_count == total_rounds and total_rounds > 0:
            status_display = f"Season {year} Calendar (Upcoming Championship: {total_rounds} Rounds Scheduled)"
        else:
            status_display = f"Season {year} Calendar ({total_rounds} Rounds Listed)"

        return SeasonCalendar(
            year=year,
            total_rounds=total_rounds,
            completed_rounds=0,  # Never invent completed rounds from calendar-only queries
            upcoming_rounds=upcoming_count,
            past_scheduled_rounds=past_scheduled_count,
            is_in_progress=is_in_progress,
            events=events,
            status_display=status_display,
        )

    def get_event(
        self,
        year: int,
        round_num: int,
        ref_date: Optional[datetime] = None,
    ) -> Optional[HistoricalEvent]:
        """
        Retrieves a single historical Grand Prix event by season and round number.
        """
        calendar = self.get_season_calendar(year, ref_date=ref_date)
        for ev in calendar.events:
            if ev.round == round_num:
                return ev
        return None

    def get_available_seasons(self) -> List[int]:
        """
        Retrieves all season years recorded in the provider archive.
        """
        try:
            seasons_raw = self.api_client.get_all_pages(
                endpoint="seasons",
                table_key="SeasonTable",
                items_key="Seasons",
            )
            years: List[int] = []
            for s in seasons_raw:
                try:
                    years.append(int(s.get("season")))
                except (ValueError, TypeError):
                    continue
            return sorted(years)
        except Exception as e:
            logger.warning(f"Error fetching available seasons: {e}")
            return []

    def _parse_race_entry(
        self,
        race: Dict[str, Any],
        year: int,
        now_utc: datetime,
    ) -> HistoricalEvent:
        """Parses a single race dictionary into a HistoricalEvent dataclass."""
        try:
            round_int = int(race.get("round", 0))
        except (ValueError, TypeError):
            round_int = 0

        event_name = race.get("raceName") or f"Round {round_int}"
        circuit_info = race.get("Circuit", {})
        circuit_id = circuit_info.get("circuitId") or "unknown"
        circuit_name = circuit_info.get("circuitName") or None

        date_str = race.get("date")
        time_str = race.get("time")
        url = race.get("url")

        is_completed, is_upcoming, is_cancelled, status_label = self._evaluate_event_status(
            date_str=date_str,
            time_str=time_str,
            race_name=event_name,
            now_utc=now_utc,
        )

        return HistoricalEvent(
            season=year,
            round=round_int,
            event_name=event_name,
            circuit_id=circuit_id,
            circuit_name=circuit_name,
            date=date_str if date_str else None,
            time=time_str if time_str else None,
            url=url if url else None,
            is_completed=is_completed,
            is_upcoming=is_upcoming,
            is_cancelled=is_cancelled,
            status_label=status_label,
        )

    @staticmethod
    def _evaluate_event_status(
        date_str: Optional[str],
        time_str: Optional[str],
        race_name: str,
        now_utc: datetime,
    ) -> Tuple[bool, bool, bool, str]:
        """
        Evaluates whether an event is scheduled in the past or upcoming.

        IMPORTANT CALENDAR DATA LIMITATIONS:
        1. Calendar Date Passage != Confirmed Completion:
           A past scheduled date indicates that the scheduled date has passed, but does NOT
           prove that the race took place or was completed (e.g. cancellations, weather abandonments,
           or unannounced postponements). Therefore, is_completed is strictly False and status is
           'Scheduled (Past Date)'. Official completion requires verifying race classification results.
        2. No Authoritative Cancellation Field:
           Jolpica's Ergast-compatible calendar endpoint does NOT provide an authoritative structured
           cancellation status field. Any raceName matching (e.g. '[Cancelled]') is purely an advisory
           heuristic, not confirmed provider status.
        3. Missing Event Dates:
           Missing or invalid dates are marked as 'Date Unknown' with is_completed=False and is_upcoming=False.
        4. Rescheduled Events:
           The provider updates dates in-place without historical rescheduling audit trails;
           original dates are never fabricated.

        Returns:
            Tuple[is_completed, is_upcoming, is_cancelled, status_label]
        """
        # Advisory heuristic: check if raceName contains "cancelled".
        # The provider's Ergast-compatible schema does NOT establish authoritative cancellation status.
        # Therefore, is_cancelled remains False, and only the advisory status_label is updated.
        if "cancelled" in race_name.lower():
            return False, False, False, "Advisory: Likely Cancelled"

        if not date_str or not str(date_str).strip():
            return False, False, False, "Date Unknown"

        clean_date = str(date_str).strip()
        try:
            # Construct ISO-8601 string if time is available
            if time_str and ":" in str(time_str):
                clean_time = str(time_str).strip().rstrip("Z")
                dt_iso = f"{clean_date}T{clean_time}+00:00"
            else:
                # When time is omitted, treat as date-only scheduled event at end of day UTC
                dt_iso = f"{clean_date}T23:59:59+00:00"

            event_dt = datetime.fromisoformat(dt_iso)

            if event_dt < now_utc:
                # Calendar date elapsed, but completion has not been confirmed from results
                return False, False, False, "Scheduled (Past Date)"
            else:
                return False, True, False, "Upcoming"

        except (ValueError, TypeError, OverflowError) as e:
            logger.debug(f"Could not parse event date for round status: {e}")
            return False, False, False, "Date Unknown"
