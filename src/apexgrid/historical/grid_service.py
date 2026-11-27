"""
ApexGrid — Historical Starting Grid Service
============================================
Dedicated service for retrieving Formula 1 official starting grids across 1950–2026.
Retrieves data dynamically via HistoricalAPIClient from Jolpica's Ergast-compatible API.
Enforces:
- Strict independence from qualifying results (never derives grid from qualifying).
- Strict independence from race results (never infers grid from finish order).
- Raw grid value preservation (grid == '0' is preserved as unclassified/non-standard, never assumed pit lane).
- Missing grid positions remain None (zero penalty fabrication).
- Complete isolation from src/data/grid_retriever.py (zero filesystem persistence).
"""

from typing import Optional, List, Dict, Any

from apexgrid.historical.api_client import HistoricalAPIClient
from apexgrid.historical.models import StartingGridResultEntry, interpret_grid_position
from apexgrid.utils.logging_utils import get_logger

logger = get_logger("historical_grid_service")


class HistoricalGridService:
    """
    Fetch-only service for Formula 1 recorded starting grids.
    All data is retrieved on-demand and retained strictly in process memory.
    """

    def __init__(self, api_client: Optional[HistoricalAPIClient] = None):
        self.api_client = api_client or HistoricalAPIClient()

    def get_starting_grid(
        self,
        year: int,
        round_num: int,
        bypass_cache: bool = False,
    ) -> List[StartingGridResultEntry]:
        """
        Retrieves the recorded starting grid for a specific Grand Prix.
        The verified source in Ergast/Jolpica is the official 'grid' attribute in results.json.

        Args:
            year: Championship season year (1950–2026).
            round_num: Championship round number.
            bypass_cache: If True, bypasses memory cache.

        Returns:
            List[StartingGridResultEntry]: Grid entries ordered by recorded starting slot.
        """
        endpoint = f"{year}/{round_num}/results"
        try:
            response = self.api_client.get_json(endpoint, bypass_cache=bypass_cache)
        except Exception as e:
            logger.warning(f"Error fetching starting grid for {year} Round {round_num}: {e}")
            return []

        mrdata = response.get("MRData", {})
        race_table = mrdata.get("RaceTable", {})
        races = race_table.get("Races", [])

        if not races:
            return []

        raw_results = races[0].get("Results", [])
        if not raw_results:
            return []

        return self._parse_starting_grid_entries(raw_results)

    def get_season_starting_grids(
        self,
        year: int,
        bypass_cache: bool = False,
    ) -> Dict[int, List[StartingGridResultEntry]]:
        """
        Retrieves starting grids across all completed rounds in a season.

        Returns:
            Dict[int, List[StartingGridResultEntry]]: Mapping of round_num -> starting grid entries.
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
            logger.warning(f"Error fetching season starting grids for {year}: {e}")
            return {}

        season_grids: Dict[int, List[StartingGridResultEntry]] = {}
        seen_entries: Dict[int, set] = {}

        for race in races_raw:
            try:
                round_int = int(race.get("round", 0))
            except (ValueError, TypeError):
                continue

            results = race.get("Results", [])
            if not results:
                continue

            parsed_list = self._parse_starting_grid_entries(results)
            if round_int not in season_grids:
                season_grids[round_int] = []
                seen_entries[round_int] = set()

            for entry in parsed_list:
                entry_key = (entry.driver_id, entry.constructor_name, entry.raw_grid_value)
                if entry_key not in seen_entries[round_int]:
                    seen_entries[round_int].add(entry_key)
                    season_grids[round_int].append(entry)

        return season_grids

    def _parse_starting_grid_entries(
        self,
        raw_results: List[Dict[str, Any]],
    ) -> List[StartingGridResultEntry]:
        """
        Parses raw race results dictionaries to extract recorded starting grid positions.
        Preserves raw grid strings and strictly avoids fabricating penalties or qualifying cross-references.
        """
        entries: List[StartingGridResultEntry] = []

        for item in raw_results:
            if not isinstance(item, dict):
                continue

            raw_grid = item.get("grid")
            raw_grid_str = str(raw_grid).strip() if raw_grid is not None else ""

            # Interpret grid position safely without pit lane assumptions
            parsed_grid_pos, grid_status = interpret_grid_position(raw_grid)

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

            constructor_info = item.get("Constructor") or {}
            constructor_name = constructor_info.get("name") or str(constructor_info.get("constructorId") or "Unknown")

            # Source notes: explicitly attribute to provider-supplied race classification data
            source_notes = "Jolpica provider-supplied race classifications starting grid attribute"

            entries.append(
                StartingGridResultEntry(
                    grid_position=parsed_grid_pos,
                    raw_grid_value=raw_grid_str,
                    qualifying_position=None,  # Never derived or fabricated from results endpoint
                    driver_id=driver_id,
                    driver_name=driver_name,
                    constructor_name=constructor_name,
                    grid_status=grid_status,
                    source_notes=source_notes,
                )
            )

        # Sort: Valid positive grid positions ascending (1, 2, 3...), followed by unclassified/None
        def sort_key(e: StartingGridResultEntry):
            if e.grid_position is not None and e.grid_position > 0:
                return (0, e.grid_position)
            return (1, 999)

        entries.sort(key=sort_key)
        return entries
