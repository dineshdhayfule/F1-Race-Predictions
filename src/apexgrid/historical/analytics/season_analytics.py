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

from dataclasses import replace
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict

from apexgrid.historical.api_client import HistoricalAPIClient
from apexgrid.historical.events_service import HistoricalEventsService
from apexgrid.historical.classification_service import HistoricalClassificationService
from apexgrid.historical.qualifying_service import HistoricalQualifyingService
from apexgrid.historical.analytics.models import (
    ConstructorSeasonStanding,
    DriverSeasonStanding,
    SeasonAnalyticsSummary,
)
from apexgrid.utils.logging_utils import get_logger

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

        from apexgrid.utils.season_utils import get_active_season
        # Determine in-progress state
        is_in_progress = (completed_rounds < total_scheduled) or (year >= get_active_season() and completed_rounds < total_scheduled)

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
        driver_podiums: Dict[str, int] = defaultdict(int)
        driver_dnfs: Dict[str, int] = defaultdict(int)
        driver_dns: Dict[str, int] = defaultdict(int)
        driver_poles: Dict[str, int] = defaultdict(int)
        driver_teams: Dict[str, str] = {}
        driver_names: Dict[str, str] = {}

        constructor_points: Dict[str, float] = defaultdict(float)
        constructor_wins: Dict[str, int] = defaultdict(int)
        constructor_podiums: Dict[str, int] = defaultdict(int)

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
                if e.position in (1, 2, 3):
                    driver_podiums[e.driver_id] += 1
                    constructor_podiums[e.constructor_name] += 1

                pos_text = str(e.position_text or "").strip().upper()
                st_lower = str(e.status or "").strip().lower()
                is_dns = (
                    pos_text in ("W", "DNQ", "DNPQ", "DNA", "DNS", "F", "N")
                    or st_lower in (
                        "did not start",
                        "withdrew",
                        "not started",
                        "did not qualify",
                        "did not prequalify",
                        "failed to qualify",
                    )
                )
                is_running_finish = (
                    st_lower == "finished"
                    or st_lower == "lapped"
                    or st_lower.startswith("+")
                )
                is_disqualified = (
                    pos_text == "D"
                    or st_lower in ("disqualified", "excluded")
                )
                is_retired = (not is_running_finish and not is_dns and not is_disqualified)
                if is_dns:
                    driver_dns[e.driver_id] += 1
                elif is_retired:
                    driver_dnfs[e.driver_id] += 1

        # Qualifying coverage evaluation (computed before standings so pole counts are available)
        if year < 1994:
            q_summary = f"Qualifying session data unavailable for season {year} (Ergast/Jolpica archive starts in 1994)."
        else:
            season_q = self.qualifying_service.get_season_qualifying(year)
            q_count = len(season_q)
            q_summary = f"Qualifying sessions recorded for {q_count}/{completed_rounds} completed rounds."
            for q_entries in season_q.values():
                for qe in q_entries:
                    if qe.position == 1:
                        driver_poles[qe.driver_id] += 1

        # 4. Fetch official driver standings if available from provider (full grid)
        full_drivers_standings: List[DriverSeasonStanding] = []
        try:
            standings_resp = self.api_client.get_json(f"{year}/driverStandings")
            mrdata = standings_resp.get("MRData", {})
            standings_lists = mrdata.get("StandingsTable", {}).get("StandingsLists", [])
            if standings_lists:
                driver_standings_raw = standings_lists[0].get("DriverStandings", [])
                for idx, item in enumerate(driver_standings_raw, 1):
                    try:
                        rank = int(item.get("position", idx))
                    except (ValueError, TypeError):
                        rank = idx
                    d_info = item.get("Driver", {})
                    d_id = str(d_info.get("driverId", ""))
                    d_name = f"{d_info.get('givenName', '')} {d_info.get('familyName', '')}".strip() or d_id or "Unknown"
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
                    full_drivers_standings.append(
                        DriverSeasonStanding(
                            rank=rank,
                            driver_name=d_name,
                            team_name=c_name,
                            points=pts,
                            wins=wns,
                            podiums=driver_podiums.get(d_id, 0),
                            poles=driver_poles.get(d_id, 0),
                            dnfs=driver_dnfs.get(d_id, 0),
                            dns=driver_dns.get(d_id, 0),
                        )
                    )
        except Exception:
            pass

        # Fallback to computing from classified race results if standings endpoint was empty
        if not full_drivers_standings and driver_points:
            sorted_drivers = sorted(driver_points.items(), key=lambda x: (x[1], driver_wins[x[0]]), reverse=True)
            for idx, (d_id, pts) in enumerate(sorted_drivers, 1):
                full_drivers_standings.append(
                    DriverSeasonStanding(
                        rank=idx,
                        driver_name=driver_names.get(d_id, d_id),
                        team_name=driver_teams.get(d_id, "Unknown"),
                        points=round(pts, 2),
                        wins=driver_wins.get(d_id, 0),
                        podiums=driver_podiums.get(d_id, 0),
                        poles=driver_poles.get(d_id, 0),
                        dnfs=driver_dnfs.get(d_id, 0),
                        dns=driver_dns.get(d_id, 0),
                    )
                )

        if full_drivers_standings:
            d_leader_pts = max(s.points for s in full_drivers_standings)
            full_drivers_standings = [
                replace(s, gap_to_leader=round(max(0.0, d_leader_pts - s.points), 2))
                for s in full_drivers_standings
            ]

        top_drivers_standings: List[Tuple[int, str, str, float, int]] = [
            (s.rank, s.driver_name, s.team_name, s.points, s.wins)
            for s in full_drivers_standings[:10]
        ]

        # 5. Fetch official constructor standings if available (full grid)
        full_constructors_standings: List[ConstructorSeasonStanding] = []
        try:
            c_standings_resp = self.api_client.get_json(f"{year}/constructorStandings")
            mrdata_c = c_standings_resp.get("MRData", {})
            c_lists = mrdata_c.get("StandingsTable", {}).get("StandingsLists", [])
            if c_lists:
                c_raw = c_lists[0].get("ConstructorStandings", [])
                for idx, item in enumerate(c_raw, 1):
                    try:
                        rank = int(item.get("position", idx))
                    except (ValueError, TypeError):
                        rank = idx
                    c_name = item.get("Constructor", {}).get("name", "Unknown")
                    try:
                        pts = float(item.get("points", 0.0))
                    except (ValueError, TypeError):
                        pts = 0.0
                    try:
                        wns = int(item.get("wins", 0))
                    except (ValueError, TypeError):
                        wns = 0
                    full_constructors_standings.append(
                        ConstructorSeasonStanding(
                            rank=rank,
                            team_name=c_name,
                            points=pts,
                            wins=wns,
                            podiums=constructor_podiums.get(c_name, 0),
                        )
                    )
        except Exception:
            pass

        if not full_constructors_standings and constructor_points:
            sorted_constructors = sorted(constructor_points.items(), key=lambda x: (x[1], constructor_wins[x[0]]), reverse=True)
            for idx, (c_name, pts) in enumerate(sorted_constructors, 1):
                full_constructors_standings.append(
                    ConstructorSeasonStanding(
                        rank=idx,
                        team_name=c_name,
                        points=round(pts, 2),
                        wins=constructor_wins.get(c_name, 0),
                        podiums=constructor_podiums.get(c_name, 0),
                    )
                )

        if full_constructors_standings:
            c_leader_pts = max(s.points for s in full_constructors_standings)
            full_constructors_standings = [
                replace(s, gap_to_leader=round(max(0.0, c_leader_pts - s.points), 2))
                for s in full_constructors_standings
            ]

        top_constructors_standings: List[Tuple[int, str, float, int]] = [
            (s.rank, s.team_name, s.points, s.wins)
            for s in full_constructors_standings[:10]
        ]

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
            full_drivers_standings=full_drivers_standings,
            full_constructors_standings=full_constructors_standings,
        )
