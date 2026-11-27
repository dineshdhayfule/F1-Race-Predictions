"""
ApexGrid — Historical Teammate Analytics Service
=================================================
Performs verified teammate head-to-head comparisons across Formula 1 history.
Enforces:
- Teammate constraint: drivers are only compared when entered for the same constructor in the same event.
- Shared-drive preservation: shared car entries are treated as ties rather than artificial losses.
- Missing qualifying data is never treated as a loss.
- Transparent denominators explaining exclusions (e.g. double DNFs, missing qualifying sessions).
- Fetch-only: In-memory calculations; zero disk persistence.
"""

from typing import Optional, List, Tuple, Dict, Any

from apexgrid.historical.api_client import HistoricalAPIClient
from apexgrid.historical.classification_service import HistoricalClassificationService
from apexgrid.historical.qualifying_service import HistoricalQualifyingService
from apexgrid.historical.analytics.models import TeammateComparisonStats
from apexgrid.utils.logging_utils import get_logger

logger = get_logger("historical_teammate_analytics")


class TeammateAnalyticsService:
    """Computes verified head-to-head comparisons between Formula 1 teammates."""

    def __init__(
        self,
        api_client: Optional[HistoricalAPIClient] = None,
        classification_service: Optional[HistoricalClassificationService] = None,
        qualifying_service: Optional[HistoricalQualifyingService] = None,
    ):
        self.api_client = api_client or HistoricalAPIClient()
        self.classification_service = classification_service or HistoricalClassificationService(api_client=self.api_client)
        self.qualifying_service = qualifying_service or HistoricalQualifyingService(api_client=self.api_client)

    def compare_teammates(
        self,
        driver1_id: str,
        driver2_id: str,
        season: int,
        constructor_id: Optional[str] = None,
    ) -> Optional[TeammateComparisonStats]:
        """
        Computes the head-to-head comparison between two drivers in a specific season.

        Args:
            driver1_id: Canonical Ergast driver ID for driver 1 (e.g. 'hamilton').
            driver2_id: Canonical Ergast driver ID for driver 2 (e.g. 'rosberg').
            season: Championship season year.
            constructor_id: Optional specific constructor filter. If None, auto-detects shared constructor.

        Returns:
            Optional[TeammateComparisonStats]: Head-to-head statistics or None if not teammates.
        """
        d1 = driver1_id.strip().lower()
        d2 = driver2_id.strip().lower()

        # 1. Fetch race classifications for the season
        season_races = self.classification_service.get_season_classifications(season)
        if not season_races:
            logger.info(f"No race classification records found for season {season}")
            return None

        # 2. Fetch qualifying results for the season (empty for pre-1994)
        season_quali = self.qualifying_service.get_season_qualifying(season) if season >= 1994 else {}

        # 3. Identify shared rounds
        matched_rounds = []
        c_name = "Unknown Constructor"
        d1_display = d1
        d2_display = d2

        for rnd, entries in season_races.items():
            d1_entry = None
            d2_entry = None
            for e in entries:
                if e.driver_id == d1:
                    d1_entry = e
                    d1_display = e.driver_name
                elif e.driver_id == d2:
                    d2_entry = e
                    d2_display = e.driver_name

            if d1_entry and d2_entry:
                # Check that they drove for the same constructor
                if d1_entry.constructor_id == d2_entry.constructor_id:
                    if constructor_id is None or d1_entry.constructor_id == constructor_id.strip().lower():
                        c_name = d1_entry.constructor_name
                        matched_rounds.append((rnd, d1_entry, d2_entry))

        if not matched_rounds:
            logger.info(f"No cohabited races found for {d1} and {d2} in {season}")
            return None

        races_cohabited = len(matched_rounds)
        d1_race_ahead = 0
        d2_race_ahead = 0
        both_finished_count = 0
        d1_wins = 0
        d2_wins = 0
        d1_podiums = 0
        d2_podiums = 0
        d1_points = 0.0
        d2_points = 0.0
        d1_dnfs = 0
        d2_dnfs = 0

        for rnd, e1, e2 in matched_rounds:
            # Points
            d1_points += e1.points
            d2_points += e2.points

            # Wins
            if e1.position == 1:
                d1_wins += 1
            if e2.position == 1:
                d2_wins += 1

            # Podiums
            if e1.position in (1, 2, 3):
                d1_podiums += 1
            if e2.position in (1, 2, 3):
                d2_podiums += 1

            # DNF check
            e1_classified = e1.status == "Finished" or "+" in e1.status or (e1.position is not None and e1.position_text.isdigit())
            e2_classified = e2.status == "Finished" or "+" in e2.status or (e2.position is not None and e2.position_text.isdigit())

            if not e1_classified:
                d1_dnfs += 1
            if not e2_classified:
                d2_dnfs += 1

            # Race finish head-to-head
            if e1_classified and e2_classified:
                both_finished_count += 1
                if e1.position is not None and e2.position is not None:
                    if e1.position < e2.position:
                        d1_race_ahead += 1
                    elif e2.position < e1.position:
                        d2_race_ahead += 1
                    # Shared car tie -> neither ahead
            elif e1_classified and not e2_classified:
                d1_race_ahead += 1
            elif e2_classified and not e1_classified:
                d2_race_ahead += 1
            # If both retired, excluded from finish comparison denominator

        # 4. Qualifying head-to-head
        d1_q_ahead = 0
        d2_q_ahead = 0
        q_sessions_compared = 0
        notes = []

        if season < 1994:
            q_coverage_note = "Qualifying head-to-head unavailable for pre-1994 season (session results absent in provider data)."
            notes.append(q_coverage_note)
        else:
            for rnd, _, _ in matched_rounds:
                q_entries = season_quali.get(rnd, [])
                q1_entry = next((q for q in q_entries if q.driver_id == d1), None)
                q2_entry = next((q for q in q_entries if q.driver_id == d2), None)

                # Both must have valid qualifying positions to establish a fair head-to-head
                if q1_entry and q2_entry and q1_entry.position > 0 and q2_entry.position > 0:
                    q_sessions_compared += 1
                    if q1_entry.position < q2_entry.position:
                        d1_q_ahead += 1
                    elif q2_entry.position < q1_entry.position:
                        d2_q_ahead += 1
                else:
                    # Missing or non-participating session excluded from comparison denominator
                    pass

            q_coverage_note = f"Qualifying head-to-head compared across {q_sessions_compared}/{races_cohabited} sessions where both drivers set classifications."

        if both_finished_count < races_cohabited:
            notes.append(f"Race head-to-head includes {races_cohabited - both_finished_count} rounds where one driver suffered a DNF.")

        return TeammateComparisonStats(
            driver1_id=d1,
            driver1_name=d1_display,
            driver2_id=d2,
            driver2_name=d2_display,
            constructor_name=c_name,
            season=season,
            races_cohabited=races_cohabited,
            both_finished_count=both_finished_count,
            race_h2h=(d1_race_ahead, d2_race_ahead),
            qualifying_sessions_compared=q_sessions_compared,
            qualifying_h2h=(d1_q_ahead, d2_q_ahead),
            qualifying_coverage_note=q_coverage_note,
            wins_comparison=(d1_wins, d2_wins),
            podiums_comparison=(d1_podiums, d2_podiums),
            points_comparison=(round(d1_points, 2), round(d2_points, 2)),
            dnfs_comparison=(d1_dnfs, d2_dnfs),
            notes=notes,
        )
