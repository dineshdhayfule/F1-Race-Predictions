"""
ApexGrid — Historical Qualifying Analytics Service
===================================================
Analyzes Formula 1 qualifying performance, session format distribution, and grid differences.
Enforces:
- Strict preservation of historical qualifying formats (QB, QO, QA, Knockout).
- Zero fabrication of Q1/Q2/Q3 lap times for historical single-session formats.
- Qualifying-to-grid delta calculation strictly when both positions are independently valid (>0).
- Transparent handling of pre-1994 data absence.
- Fetch-only: In-memory calculations; zero disk persistence.
"""

from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict

from src.historical.api_client import HistoricalAPIClient
from src.historical.qualifying_service import HistoricalQualifyingService
from src.historical.grid_service import HistoricalGridService
from src.historical.analytics.models import QualifyingAnalyticsSummary
from src.utils.logging_utils import get_logger

logger = get_logger("historical_qualifying_analytics")


class QualifyingAnalyticsService:
    """Computes verified historical qualifying analytics."""

    def __init__(
        self,
        api_client: Optional[HistoricalAPIClient] = None,
        qualifying_service: Optional[HistoricalQualifyingService] = None,
        grid_service: Optional[HistoricalGridService] = None,
    ):
        self.api_client = api_client or HistoricalAPIClient()
        self.qualifying_service = qualifying_service or HistoricalQualifyingService(api_client=self.api_client)
        self.grid_service = grid_service or HistoricalGridService(api_client=self.api_client)

    def get_driver_qualifying_summary(
        self,
        driver_id: str,
        season: Optional[int] = None,
    ) -> Optional[QualifyingAnalyticsSummary]:
        """
        Computes qualifying performance summary for a driver across a career or single season.

        Args:
            driver_id: Canonical Ergast driver ID (e.g. 'verstappen', 'senna').
            season: Optional championship season filter.

        Returns:
            Optional[QualifyingAnalyticsSummary]: Qualifying statistics or None if unavailable.
        """
        driver_clean = driver_id.strip().lower()

        # If season is specified and < 1994, return summary with explicit unavailable note
        if season is not None and season < 1994:
            return QualifyingAnalyticsSummary(
                driver_id=driver_clean,
                driver_name=driver_clean,
                season=season,
                total_sessions=0,
                pole_positions=0,
                front_row_starts=0,
                top_10_starts=0,
                avg_qualifying_position=None,
                formats_encountered=["Pre-1994 Archive (Unavailable)"],
                position_distribution={},
                qualifying_to_grid_deltas=[],
                avg_grid_drop_or_gain=None,
                coverage_limitations=["Official qualifying session lap times are not available in provider data for pre-1994 seasons."],
            )

        # Retrieve qualifying results
        if season is not None:
            season_quali = self.qualifying_service.get_season_qualifying(season)
            season_grids = self.grid_service.get_season_starting_grids(season)
            rounds_to_process = [(season, rnd, q_entries) for rnd, q_entries in season_quali.items()]
        else:
            # Query driver career qualifying sessions
            # In Ergast: drivers/{driver_id}/qualifying.json
            endpoint = f"drivers/{driver_clean}/qualifying"
            try:
                races_raw = self.api_client.get_all_pages(
                    endpoint=endpoint,
                    table_key="RaceTable",
                    items_key="Races",
                )
            except Exception as e:
                logger.warning(f"Error querying career qualifying for {driver_clean}: {e}")
                return None

            rounds_to_process = []
            season_grids = {}
            for race in races_raw:
                try:
                    s_int = int(race.get("season", 0))
                    r_int = int(race.get("round", 0))
                except (ValueError, TypeError):
                    continue
                q_list = race.get("QualifyingResults", [])
                if q_list:
                    parsed_entries = self.qualifying_service._parse_qualifying_results(q_list, s_int)
                    rounds_to_process.append((s_int, r_int, parsed_entries))

        if not rounds_to_process:
            return None

        total_sessions = 0
        poles = 0
        front_rows = 0
        top_10s = 0
        q_positions: List[int] = []
        driver_name = driver_clean
        formats_set = set()
        pos_distribution: Dict[str, int] = defaultdict(int)
        grid_deltas: List[int] = []

        for s_year, rnd, q_entries in rounds_to_process:
            # Find driver entry
            driver_q = next((q for q in q_entries if q.driver_id == driver_clean), None)
            if not driver_q:
                continue

            total_sessions += 1
            driver_name = driver_q.driver_name
            formats_set.add(driver_q.session_format)
            pos = driver_q.position
            q_positions.append(pos)

            if pos == 1:
                poles += 1
                pos_distribution["P1"] += 1
            elif pos == 2:
                front_rows += 1
                pos_distribution["P2"] += 1
            elif pos <= 10:
                top_10s += 1
                pos_distribution[f"P{pos}"] += 1
            else:
                pos_distribution[f"P{pos}"] += 1

            if pos in (1, 2):
                front_rows += 1 if pos == 1 else 0  # avoid double counting if P1 already counted above

            # Check starting grid to compute qualifying-to-grid difference (e.g. penalties)
            # Only when both positions are valid > 0
            if season is not None:
                grid_entries = season_grids.get(rnd, [])
                driver_grid = next((g for g in grid_entries if g.driver_id == driver_clean), None)
                if driver_grid and driver_grid.grid_position and driver_grid.grid_position > 0:
                    # delta = grid_pos - quali_pos (positive means dropped back on grid due to penalty)
                    delta = driver_grid.grid_position - pos
                    grid_deltas.append(delta)

        if total_sessions == 0:
            return None

        avg_q = sum(q_positions) / len(q_positions) if q_positions else None
        avg_delta = sum(grid_deltas) / len(grid_deltas) if grid_deltas else None

        coverage_limitations = []
        if season is None or season < 1994:
            coverage_limitations.append("Provider dataset records qualifying sessions starting from the 1994 championship season onward.")

        return QualifyingAnalyticsSummary(
            driver_id=driver_clean,
            driver_name=driver_name,
            season=season,
            total_sessions=total_sessions,
            pole_positions=poles,
            front_row_starts=poles + (front_rows - poles),
            top_10_starts=poles + (front_rows - poles) + top_10s,
            avg_qualifying_position=round(avg_q, 2) if avg_q is not None else None,
            formats_encountered=sorted(list(formats_set)),
            position_distribution=dict(sorted(pos_distribution.items())),
            qualifying_to_grid_deltas=grid_deltas,
            avg_grid_drop_or_gain=round(avg_delta, 2) if avg_delta is not None else None,
            coverage_limitations=coverage_limitations,
        )
