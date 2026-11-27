"""
ApexGrid — Historical Race Classification Service
==================================================
Dedicated service for retrieving Formula 1 official race finishing classifications across 1950–2026.
Retrieves data dynamically via HistoricalAPIClient from Jolpica's Ergast-compatible API.
Enforces:
- Official provider-awarded points (floating-point preservation, e.g. 8.14 pts; never recalculated).
- Shared-drive records preserved as distinct entries (e.g. 1951 Round 7 Farina / Bonetto).
- Raw status preservation ("Finished", "Engine", "Disqualified", "+1 Lap", etc.).
- Positions gained/lost calculated safely (None if grid position or finish position is unclassified).
- Fastest-lap rank and lap times preserved when supplied (2004+).
- Strict separation: Grand Prix classifications strictly exclude Sprint results.
- In-memory processing with zero filesystem persistence.
"""

from typing import Optional, List, Dict, Any

from apexgrid.historical.api_client import HistoricalAPIClient
from apexgrid.historical.models import RaceResultEntry, calculate_positions_gained, interpret_grid_position
from apexgrid.utils.logging_utils import get_logger

logger = get_logger("historical_classification_service")


class HistoricalClassificationService:
    """
    Fetch-only service for Formula 1 official race classifications.
    All data is retrieved on-demand and retained strictly in process memory.
    """

    def __init__(self, api_client: Optional[HistoricalAPIClient] = None):
        self.api_client = api_client or HistoricalAPIClient()

    def get_race_classification(
        self,
        year: int,
        round_num: int,
        bypass_cache: bool = False,
    ) -> List[RaceResultEntry]:
        """
        Retrieves official Grand Prix race finishing classification for a given round.

        Args:
            year: Championship season year (1950–2026).
            round_num: Championship round number.
            bypass_cache: If True, forces fresh retrieval from API.

        Returns:
            List[RaceResultEntry]: Official classified results in finishing order.
        """
        endpoint = f"{year}/{round_num}/results"
        try:
            response = self.api_client.get_json(endpoint, bypass_cache=bypass_cache)
        except Exception as e:
            logger.warning(f"Error fetching race results for {year} Round {round_num}: {e}")
            return []

        mrdata = response.get("MRData", {})
        race_table = mrdata.get("RaceTable", {})
        races = race_table.get("Races", [])

        if not races:
            return []

        raw_results = races[0].get("Results", [])
        if not raw_results:
            return []

        return self._parse_race_results(raw_results)

    def get_season_classifications(
        self,
        year: int,
        bypass_cache: bool = False,
    ) -> Dict[int, List[RaceResultEntry]]:
        """
        Retrieves official race finishing classifications across an entire season.

        Returns:
            Dict[int, List[RaceResultEntry]]: Mapping of round_num -> list of classified results.
        """
        endpoint = f"{year}/results"
        try:
            races_raw = self.api_client.get_all_pages(
                endpoint=endpoint,
                table_key="RaceTable",
                items_key="Races",
                bypass_cache=bypass_cache,
            )
        except Exception as e:
            logger.warning(f"Error fetching season classifications for {year}: {e}")
            return {}

        season_results: Dict[int, List[RaceResultEntry]] = {}
        seen_entries: Dict[int, set] = {}

        for race in races_raw:
            try:
                round_int = int(race.get("round", 0))
            except (ValueError, TypeError):
                continue

            results = race.get("Results", [])
            if not results:
                continue

            parsed_list = self._parse_race_results(results)
            if round_int not in season_results:
                season_results[round_int] = []
                seen_entries[round_int] = set()

            for entry in parsed_list:
                # Key to deduplicate repeated records from overlapping pages while preserving shared drives
                entry_key = (entry.driver_id, entry.constructor_id, entry.position_text, entry.raw_grid_value)
                if entry_key not in seen_entries[round_int]:
                    seen_entries[round_int].add(entry_key)
                    season_results[round_int].append(entry)

        return season_results

    def is_race_confirmed_completed(
        self,
        year: int,
        round_num: int,
    ) -> bool:
        """
        Establishes true confirmation of race completion.
        Unlike calendar scheduled dates, a non-empty official race classification confirms
        that the race actually took place and was officially classified by the FIA.
        """
        results = self.get_race_classification(year, round_num)
        return len(results) > 0

    def _parse_race_results(
        self,
        raw_results: List[Dict[str, Any]],
    ) -> List[RaceResultEntry]:
        """
        Normalizes provider race result records into typed RaceResultEntry objects.
        Preserves raw status strings, fractional points, and shared drive entries.
        """
        entries: List[RaceResultEntry] = []

        for item in raw_results:
            if not isinstance(item, dict):
                continue

            # Parse position as optional int
            pos_raw = item.get("position")
            try:
                pos_int = int(pos_raw) if pos_raw is not None else None
            except (ValueError, TypeError):
                pos_int = None

            # Preserve raw position text ("1", "15", "R", "D", "W", etc.)
            position_text = str(item.get("positionText") or pos_raw or "")

            driver_info = item.get("Driver") or {}
            driver_id = str(driver_info.get("driverId") or "unknown")
            given_name = driver_info.get("givenName")
            family_name = driver_info.get("familyName")
            if given_name and family_name:
                driver_name = f"{given_name} {family_name}"
            elif family_name:
                driver_name = family_name
            else:
                driver_name = driver_id

            driver_number = item.get("number") or driver_info.get("permanentNumber")

            constructor_info = item.get("Constructor") or {}
            constructor_id = str(constructor_info.get("constructorId") or "unknown")
            constructor_name = constructor_info.get("name") or constructor_id

            # Grid position interpretation
            raw_grid = item.get("grid")
            raw_grid_str = str(raw_grid).strip() if raw_grid is not None else ""
            parsed_grid_pos, _ = interpret_grid_position(raw_grid)

            # Positions gained calculation (strictly None if grid or finish is non-standard)
            positions_gained = calculate_positions_gained(parsed_grid_pos, pos_int)

            # Time / Gap
            time_info = item.get("Time") or {}
            race_time_or_gap = time_info.get("time")

            # Laps completed
            try:
                laps = int(item.get("laps", 0))
            except (ValueError, TypeError):
                laps = 0

            # Points (preserve float precision for 8.14, 1.5, 0.5)
            try:
                points_val = float(item.get("points", 0.0))
            except (ValueError, TypeError):
                points_val = 0.0

            # Status (preserve raw source string: "Finished", "Engine", "Disqualified", "+1 Lap")
            status = str(item.get("status") or "Unknown")

            # Fastest Lap (available from 2004 onwards)
            fastest_lap_info = item.get("FastestLap") or {}
            fl_rank = None
            if "rank" in fastest_lap_info:
                try:
                    fl_rank = int(fastest_lap_info["rank"])
                except (ValueError, TypeError):
                    fl_rank = None

            fl_time = None
            if "Time" in fastest_lap_info and isinstance(fastest_lap_info["Time"], dict):
                fl_time = fastest_lap_info["Time"].get("time")

            entries.append(
                RaceResultEntry(
                    position=pos_int,
                    position_text=position_text,
                    driver_id=driver_id,
                    driver_name=driver_name,
                    driver_number=str(driver_number) if driver_number else None,
                    constructor_id=constructor_id,
                    constructor_name=constructor_name,
                    grid_position=parsed_grid_pos,
                    raw_grid_value=raw_grid_str,
                    positions_gained=positions_gained,
                    race_time_or_gap=race_time_or_gap,
                    laps_completed=laps,
                    points=points_val,
                    status=status,
                    fastest_lap_rank=fl_rank,
                    fastest_lap_time=fl_time,
                )
            )

        return entries
