"""
ApexGrid — Historical Race Performance Analytics Service
=========================================================
Computes race execution statistics, position distributions, and overtakes/positions gained.
Enforces:
- Positions gained/lost strictly computed when BOTH grid and finish positions are valid positive integers.
- Non-standard grids (grid == 0, pit lane, unrecorded) are strictly excluded from positions-gained calculations.
- Accurate DNF/retirement categorization based on provider classification status.
- Fetch-only: In-memory calculations; zero disk persistence.
"""

from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict

from apexgrid.historical.api_client import HistoricalAPIClient
from apexgrid.historical.classification_service import HistoricalClassificationService
from apexgrid.historical.analytics.models import RacePerformanceSummary
from apexgrid.utils.logging_utils import get_logger

logger = get_logger("historical_race_analytics")


class RacePerformanceAnalyticsService:
    """Computes verified race performance analytics across Formula 1 history."""

    def __init__(
        self,
        api_client: Optional[HistoricalAPIClient] = None,
        classification_service: Optional[HistoricalClassificationService] = None,
    ):
        self.api_client = api_client or HistoricalAPIClient()
        self.classification_service = classification_service or HistoricalClassificationService(api_client=self.api_client)

    def get_driver_race_performance(
        self,
        driver_id: str,
        season: Optional[int] = None,
    ) -> Optional[RacePerformanceSummary]:
        """
        Computes race performance summary for a driver over career or a specific season.

        Args:
            driver_id: Canonical Ergast driver ID (e.g. 'alonso', 'prost').
            season: Optional championship season filter.

        Returns:
            Optional[RacePerformanceSummary]: Performance statistics or None if not found.
        """
        driver_clean = driver_id.strip().lower()
        if season is not None:
            endpoint = f"{season}/drivers/{driver_clean}/results"
        else:
            endpoint = f"drivers/{driver_clean}/results"

        try:
            races_raw = self.api_client.get_all_pages(
                endpoint=endpoint,
                table_key="RaceTable",
                items_key="Races",
            )
        except Exception as e:
            logger.warning(f"Error querying race results for {driver_clean}: {e}")
            return None

        if not races_raw:
            return None

        driver_name = driver_clean
        total_races = 0
        starts = 0
        classified_finishes = 0
        wins = 0
        podiums = 0
        points_finishes = 0
        total_points = 0.0
        dnfs = 0
        positions_gained_list: List[int] = []
        finishing_dist: Dict[str, int] = defaultdict(int)
        retirements: Dict[str, int] = defaultdict(int)

        for race in races_raw:
            for item in race.get("Results", []):
                total_races += 1
                d_info = item.get("Driver", {})
                if d_info.get("givenName") and d_info.get("familyName"):
                    driver_name = f"{d_info['givenName']} {d_info['familyName']}"

                status = str(item.get("status") or "Unknown")
                is_dns = status.lower() in ("did not start", "withdrew", "not started", "excluded") or item.get("positionText") in ("W", "F")
                if not is_dns:
                    starts += 1

                pos_raw = item.get("position")
                pos_int: Optional[int] = None
                try:
                    pos_int = int(pos_raw) if pos_raw is not None else None
                except (ValueError, TypeError):
                    pos_int = None

                is_classified = (
                    status == "Finished"
                    or "+ lap" in status.lower()
                    or "+ laps" in status.lower()
                    or (pos_int is not None and str(item.get("positionText", "")).isdigit())
                )

                if is_classified:
                    classified_finishes += 1
                    if pos_int == 1:
                        wins += 1
                        finishing_dist["P1"] += 1
                    elif pos_int in (2, 3):
                        podiums += 1
                        finishing_dist[f"P{pos_int}"] += 1
                    elif pos_int is not None and pos_int <= 10:
                        finishing_dist["P4-P10"] += 1
                    else:
                        finishing_dist["P11+"] += 1
                else:
                    dnfs += 1
                    finishing_dist["DNF"] += 1
                    retirements[status.strip()] += 1

                # Points
                try:
                    pts = float(item.get("points", 0.0))
                    total_points += pts
                    if pts > 0.0:
                        points_finishes += 1
                except (ValueError, TypeError):
                    pass

                # Positions gained/lost strictly computed when BOTH grid and finish are positive integers
                # Safe: Never calculate when grid == '0', None, or finish is None/DNF
                grid_raw = item.get("grid")
                try:
                    grid_int = int(grid_raw) if grid_raw is not None else None
                except (ValueError, TypeError):
                    grid_int = None

                if grid_int is not None and grid_int > 0 and pos_int is not None and pos_int > 0 and is_classified:
                    # Gained positions = grid_pos - finish_pos
                    # e.g., Started P10, finished P4 -> gained +6
                    positions_gained_list.append(grid_int - pos_int)

        if total_races == 0:
            return None

        dnf_rate = (dnfs / starts * 100.0) if starts > 0 else 0.0
        total_gained = sum(positions_gained_list) if positions_gained_list else None
        avg_gained = (sum(positions_gained_list) / len(positions_gained_list)) if positions_gained_list else None

        coverage_notes = []
        coverage_notes.append("Positions gained calculation strictly excludes non-standard starts (grid == '0', pit lane, unrecorded) and retirements.")

        return RacePerformanceSummary(
            driver_id=driver_clean,
            driver_name=driver_name,
            season=season,
            total_races=total_races,
            starts=starts,
            classified_finishes=classified_finishes,
            wins=wins,
            podiums=podiums + wins,  # Total podiums = P1 + P2 + P3
            points_finishes=points_finishes,
            total_points=round(total_points, 2),
            dnfs=dnfs,
            dnf_rate_pct=round(dnf_rate, 1),
            positions_gained_total=total_gained,
            avg_positions_gained=round(avg_gained, 2) if avg_gained is not None else None,
            valid_grid_starts_count=len(positions_gained_list),
            finishing_distribution=dict(sorted(finishing_dist.items())),
            retirement_breakdown=dict(sorted(retirements.items(), key=lambda x: x[1], reverse=True)),
            coverage_notes=coverage_notes,
        )
