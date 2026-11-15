"""
ApexGrid — Historical Season Analytics Service
===============================================
Computes comprehensive championship season statistics across 1950–2026.
Enforces:
- In-progress / current seasons (e.g. 2026) are dynamically identified and explicitly labeled provisional.
- Standings and championship points use provider-supplied values.
- Completed race counts are based on actual official classifications, not calendar dates.
- Transparent reporting of qualifying coverage limitations.
- Fetch-only: In-memory calculations; zero disk persistence.
"""

from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict

from src.historical.api_client import HistoricalAPIClient
from src.historical.events_service import HistoricalEventsService
from src.historical.classification_service import HistoricalClassificationService
from src.historical.qualifying_service import HistoricalQualifyingService
from src.historical.analytics.models import SeasonAnalyticsSummary
from src.utils.logging_utils import get_logger

logger = get_logger("historical_season_analytics")


class SeasonAnalyticsService:
    """Computes verified historical championship season analytics."""

    def __init__(
        self,
        api_client: Optional[HistoricalAPIClient] = None,
        events_service: Optional[HistoricalEventsService] = None,
        classification_service: Optional[HistoricalClassificationService] = None,
        qualifying_service: Optional[HistoricalQualifyingService] = None,
    ):
        self.api_client = api_client or HistoricalAPIClient()
        self.events_service = events_service or HistoricalEventsService(api_client=self.api_client)
        self.classification_service = classification_service or HistoricalClassificationService(api_client=self.api_client)
        self.qualifying_service = qualifying_service or HistoricalQualifyingService(api_client=self.api_client)

    def get_season_summary(
        self,
        year: int,
    ) -> Optional[SeasonAnalyticsSummary]:
        """
        Retrieves season analytics for a specific championship season.

        Args:
            year: Championship season year (1950–2026).

        Returns:
            Optional[SeasonAnalyticsSummary]: Season statistics or None if unavailable.
        """
        # 1. Fetch calendar
        calendar = self.events_service.get_season_calendar(year)
        total_scheduled = calendar.total_rounds if calendar else 0

        # 2. Fetch completed race classifications
        season_classifications = self.classification_service.get_season_classifications(year)
        completed_rounds = len(season_classifications)

        if total_scheduled == 0 and completed_rounds == 0:
            logger.info(f"No calendar or race data found for season {year}")
            return None

        # Determine in-progress state
        is_in_progress = (completed_rounds < total_scheduled) or (year == 2026 and completed_rounds < total_scheduled)

        if completed_rounds == 0:
            status_display = f"Season {year} (Scheduled / Pending: 0/{total_scheduled} Rounds Completed)"
        elif is_in_progress:
            status_display = f"Season {year} (In-Progress / Provisional: {completed_rounds}/{total_scheduled} Rounds Completed)"
        else:
            status_display = f"Season {year} (Completed Championship: {completed_rounds} Rounds)"

        # 3. Extract unique participating drivers and constructors
        driver_ids = set()
        constructor_ids = set()
        driver_points: Dict[str, float] = defaultdict(float)
        driver_wins: Dict[str, int] = defaultdict(int)
        driver_teams: Dict[str, str] = {}
        driver_names: Dict[str, str] = {}

        constructor_points: Dict[str, float] = defaultdict(float)
        constructor_wins: Dict[str, int] = defaultdict(int)

        for rnd, entries in season_classifications.items():
            for e in entries:
                driver_ids.add(e.driver_id)
                constructor_ids.add(e.constructor_id)
                driver_names[e.driver_id] = e.driver_name
                driver_teams[e.driver_id] = e.constructor_name

                driver_points[e.driver_id] += e.points
                constructor_points[e.constructor_name] += e.points

                if e.position == 1:
                    driver_wins[e.driver_id] += 1
                    constructor_wins[e.constructor_name] += 1

        # 4. Fetch official driver standings if available from provider
        top_drivers_standings: List[Tuple[int, str, str, float, int]] = []
        try:
            standings_resp = self.api_client.get_json(f"{year}/driverStandings")
            mrdata = standings_resp.get("MRData", {})
            standings_lists = mrdata.get("StandingsTable", {}).get("StandingsLists", [])
            if standings_lists:
                driver_standings_raw = standings_lists[0].get("DriverStandings", [])
                for item in driver_standings_raw[:10]:
                    try:
                        rank = int(item.get("position", 0))
                    except (ValueError, TypeError):
                        rank = 0
                    d_info = item.get("Driver", {})
                    d_name = f"{d_info.get('givenName', '')} {d_info.get('familyName', '')}".strip() or d_info.get("driverId", "Unknown")
                    c_lists = item.get("Constructors", [])
                    c_name = c_lists[0].get("name", "Unknown") if c_lists else "Unknown"
                    try:
                        pts = float(item.get("points", 0.0))
                    except (ValueError, TypeError):
                        pts = 0.0
                    try:
                        wns = int(item.get("wins", 0))
                    except (ValueError, TypeError):
                        wns = 0
                    top_drivers_standings.append((rank, d_name, c_name, pts, wns))
        except Exception:
            pass

        # Fallback to computing from classified race results if standings endpoint was empty
        if not top_drivers_standings and driver_points:
            sorted_drivers = sorted(driver_points.items(), key=lambda x: (x[1], driver_wins[x[0]]), reverse=True)[:10]
            for idx, (d_id, pts) in enumerate(sorted_drivers, 1):
                top_drivers_standings.append((
                    idx,
                    driver_names.get(d_id, d_id),
                    driver_teams.get(d_id, "Unknown"),
                    round(pts, 2),
                    driver_wins.get(d_id, 0),
                ))

        # 5. Fetch official constructor standings if available
        top_constructors_standings: List[Tuple[int, str, float, int]] = []
        try:
            c_standings_resp = self.api_client.get_json(f"{year}/constructorStandings")
            mrdata_c = c_standings_resp.get("MRData", {})
            c_lists = mrdata_c.get("StandingsTable", {}).get("StandingsLists", [])
            if c_lists:
                c_raw = c_lists[0].get("ConstructorStandings", [])
                for item in c_raw[:10]:
                    try:
                        rank = int(item.get("position", 0))
                    except (ValueError, TypeError):
                        rank = 0
                    c_name = item.get("Constructor", {}).get("name", "Unknown")
                    try:
                        pts = float(item.get("points", 0.0))
                    except (ValueError, TypeError):
                        pts = 0.0
                    try:
                        wns = int(item.get("wins", 0))
                    except (ValueError, TypeError):
                        wns = 0
                    top_constructors_standings.append((rank, c_name, pts, wns))
        except Exception:
            pass

        if not top_constructors_standings and constructor_points:
            sorted_constructors = sorted(constructor_points.items(), key=lambda x: (x[1], constructor_wins[x[0]]), reverse=True)[:10]
            for idx, (c_name, pts) in enumerate(sorted_constructors, 1):
                top_constructors_standings.append((
                    idx,
                    c_name,
                    round(pts, 2),
                    constructor_wins.get(c_name, 0),
                ))

        # 6. Qualifying coverage evaluation
        if year < 1994:
            q_summary = f"Qualifying session data unavailable for season {year} (Ergast/Jolpica archive starts in 1994)."
        else:
            season_q = self.qualifying_service.get_season_qualifying(year)
            q_count = len(season_q)
            q_summary = f"Qualifying sessions recorded for {q_count}/{completed_rounds} completed rounds."

        race_summary = f"Official race classifications available for {completed_rounds}/{total_scheduled} scheduled rounds."

        coverage_notes = []
        if is_in_progress:
            coverage_notes.append("Season is currently in progress or incomplete; standings and stats are provisional.")
        if year < 1994:
            coverage_notes.append("Qualifying times are not present in provider data for pre-1994 championships.")
        if year < 1958:
            coverage_notes.append("The World Constructors' Championship was established in 1958; pre-1958 team standings are unofficial aggregations.")

        return SeasonAnalyticsSummary(
            season=year,
            total_scheduled_rounds=total_scheduled,
            completed_rounds=completed_rounds,
            is_in_progress=is_in_progress,
            status_display=status_display,
            participating_drivers_count=len(driver_ids),
            participating_constructors_count=len(constructor_ids),
            top_drivers_standings=top_drivers_standings,
            top_constructors_standings=top_constructors_standings,
            qualifying_coverage_summary=q_summary,
            race_results_coverage_summary=race_summary,
            coverage_notes=coverage_notes,
        )
