"""
ApexGrid — Historical Qualifying Service
=========================================
Dedicated service for retrieving Formula 1 qualifying classifications across 1950–2026.
Retrieves data dynamically via HistoricalAPIClient from Jolpica's Ergast-compatible API.
Enforces:
- Accurate session format handling (1950–1993 unavailable; 1994–2005 single session; 2006+ knockout Q1/Q2/Q3).
- Zero fabrication of missing qualifying lap times or positions.
- Strict isolation from race starting grids (never uses grid as qualifying).
- In-memory processing with zero filesystem persistence.
"""

from typing import Optional, List, Dict, Any

from src.historical.api_client import HistoricalAPIClient, HistoricalAPIError
from src.historical.models import QualifyingResultEntry
from src.utils.logging_utils import get_logger

logger = get_logger("historical_qualifying_service")


class HistoricalQualifyingService:
    """
    Fetch-only service for Formula 1 official qualifying results.
    All data is retrieved on-demand and retained strictly in process memory.
    """

    def __init__(self, api_client: Optional[HistoricalAPIClient] = None):
        self.api_client = api_client or HistoricalAPIClient()

    def get_round_qualifying(
        self,
        year: int,
        round_num: int,
        bypass_cache: bool = False,
    ) -> List[QualifyingResultEntry]:
        """
        Retrieves official qualifying classification for a specific Grand Prix.

        Args:
            year: Championship season year (1950–2026).
            round_num: Championship round number.
            bypass_cache: If True, forces fresh retrieval from API.

        Returns:
            List[QualifyingResultEntry]: Ordered qualifying classification,
                                         or empty list if unrecorded in provider archive.
        """
        endpoint = f"{year}/{round_num}/qualifying"
        try:
            response = self.api_client.get_json(endpoint, bypass_cache=bypass_cache)
        except Exception as e:
            logger.warning(f"Error fetching qualifying for {year} Round {round_num}: {e}")
            return []

        mrdata = response.get("MRData", {})
        race_table = mrdata.get("RaceTable", {})
        races = race_table.get("Races", [])

        if not races:
            return []

        raw_results = races[0].get("QualifyingResults", [])
        if not raw_results:
            return []

        return self._parse_qualifying_results(raw_results, year)

    def get_season_qualifying(
        self,
        year: int,
        bypass_cache: bool = False,
    ) -> Dict[int, List[QualifyingResultEntry]]:
        """
        Retrieves qualifying classifications across an entire season.

        Returns:
            Dict[int, List[QualifyingResultEntry]]: Mapping of round_num -> list of qualifying entries.
        """
        if year < 1994:
            # Pre-1994 qualifying session results are absent in provider archives
            return {}

        endpoint = f"{year}/qualifying"
        season_results: Dict[int, List[QualifyingResultEntry]] = {}
        offset = 0
        page_size = 100

        try:
            while True:
                response = self.api_client.get_json(
                    endpoint,
                    params={"limit": page_size, "offset": offset},
                    bypass_cache=bypass_cache,
                )
                mrdata = response.get("MRData", {})
                try:
                    total = int(mrdata.get("total", 0))
                except (ValueError, TypeError):
                    total = 0

                race_table = mrdata.get("RaceTable", {})
                races = race_table.get("Races", [])
                if not races:
                    break

                total_q_results_in_page = 0
                for race in races:
                    try:
                        round_int = int(race.get("round", 0))
                    except (ValueError, TypeError):
                        continue

                    q_results = race.get("QualifyingResults", [])
                    total_q_results_in_page += len(q_results)
                    if q_results:
                        parsed = self._parse_qualifying_results(q_results, year)
                        if round_int not in season_results:
                            season_results[round_int] = []
                        season_results[round_int].extend(parsed)

                offset += total_q_results_in_page
                if offset >= total or total_q_results_in_page == 0:
                    break

        except HistoricalAPIError:
            raise
        except Exception as e:
            logger.warning(f"Error fetching season qualifying for {year}: {e}")
            raise HistoricalAPIError(f"Failed to fetch qualifying for season {year}: {e}") from e

        return season_results

    def get_driver_qualifying_results(
        self,
        driver_id: str,
        season: Optional[int] = None,
        bypass_cache: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves direct driver qualifying records using 'drivers/{driver_id}/qualifying'
        or '{season}/drivers/{driver_id}/qualifying'.
        Each race item in the response contains exactly one qualifying entry for the driver,
        allowing Ergast MRData pagination to advance in homogeneous units without mismatch.
        """
        clean_id = driver_id.strip().lower()
        if season:
            endpoint = f"{season}/drivers/{clean_id}/qualifying"
        else:
            endpoint = f"drivers/{clean_id}/qualifying"

        return self.api_client.get_all_pages(
            endpoint=endpoint,
            table_key="RaceTable",
            items_key="Races",
            bypass_cache=bypass_cache,
        )

    def _parse_qualifying_results(
        self,
        raw_results: List[Dict[str, Any]],
        year: int,
    ) -> List[QualifyingResultEntry]:
        """
        Normalizes provider qualifying records into typed QualifyingResultEntry objects.
        Detects session format (Knockout vs Single Session) and computes gap to pole if possible.
        """
        entries: List[QualifyingResultEntry] = []
        pole_time_str: Optional[str] = None

        for item in raw_results:
            if not isinstance(item, dict):
                continue

            try:
                pos = int(item.get("position", 0))
            except (ValueError, TypeError):
                continue

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

            driver_code = driver_info.get("code")

            constructor_info = item.get("Constructor") or {}
            constructor_id = str(constructor_info.get("constructorId") or "unknown")
            constructor_name = constructor_info.get("name") or constructor_id

            q1 = item.get("Q1")
            q2 = item.get("Q2")
            q3 = item.get("Q3")

            # Determine best lap time
            # For knockout: Q3 if present, else Q2, else Q1
            best_lap = q3 or q2 or q1

            # Detect historical session format accurately without forcing modern knockout assumptions
            if year >= 2006:
                session_format = "Knockout"
                session_code = "Q"
            elif year == 2005 and q1 and q2:
                # 2005 rounds 1-6 used Aggregate Qualifying (Saturday + Sunday combined times)
                session_format = "Qualifying Aggregate (QA)"
                session_code = "QA"
            elif 2003 <= year <= 2005:
                session_format = "Single Lap Shootout (QO/QB)"
                session_code = "QB"
            elif 1994 <= year <= 2002:
                session_format = "Qualifying Best Lap (QB)"
                session_code = "QB"
            else:
                session_format = "Pre-1994 Archive"
                session_code = "QB"

            # Track pole lap time (P1)
            if pos == 1 and best_lap:
                pole_time_str = best_lap

            gap_to_pole = self._calculate_gap(pole_time_str, best_lap) if (pos > 1 and pole_time_str and best_lap) else None

            entries.append(
                QualifyingResultEntry(
                    position=pos,
                    driver_id=driver_id,
                    driver_name=driver_name,
                    driver_code=driver_code if driver_code else None,
                    constructor_id=constructor_id,
                    constructor_name=constructor_name,
                    q1=q1 if q1 else None,
                    q2=q2 if q2 else None,
                    q3=q3 if q3 else None,
                    best_lap_time=best_lap if best_lap else None,
                    gap_to_pole=gap_to_pole,
                    session_format=session_format,
                    session_code=session_code,
                )
            )

        # Sort explicitly by official qualifying position
        entries.sort(key=lambda x: x.position)
        return entries

    @classmethod
    def parse_alpha_qualifying_result(
        cls,
        item: Dict[str, Any],
        session_type: Optional[str] = None,
        pole_time_str: Optional[str] = None,
    ) -> Optional[QualifyingResultEntry]:
        """
        Parses a Jolpica Alpha Results payload entry.
        Preserves original session identifiers (QB, QO, QA) and components without
        forcing modern knockout assumptions.
        """
        if not isinstance(item, dict):
            return None

        try:
            pos = int(item.get("position", 0))
        except (ValueError, TypeError):
            pos = 0

        driver_info = item.get("driver") or {}
        driver_id = str(driver_info.get("id") or "unknown")
        given_name = driver_info.get("given_name")
        family_name = driver_info.get("family_name")
        driver_name = f"{given_name} {family_name}".strip() if (given_name or family_name) else driver_id
        driver_code = driver_info.get("abbreviation")

        team_info = item.get("team") or {}
        constructor_id = str(team_info.get("id") or "unknown")
        constructor_name = team_info.get("name") or constructor_id

        # Alpha puts session segments under components: e.g. {'Q1': {'time': '...'}, 'Q2': ...}
        components = item.get("components") or {}
        q1 = components.get("Q1", {}).get("time") if isinstance(components.get("Q1"), dict) else None
        q2 = components.get("Q2", {}).get("time") if isinstance(components.get("Q2"), dict) else None
        q3 = components.get("Q3", {}).get("time") if isinstance(components.get("Q3"), dict) else None

        best_lap = item.get("time") or q3 or q2 or q1

        # Detect format from session_type or component keys
        st = str(session_type or "").upper()
        if st == "QA":
            fmt = "Qualifying Aggregate (QA)"
        elif st == "QO":
            fmt = "Qualifying Shootout Order (QO)"
        elif st == "QB":
            fmt = "Qualifying Best Lap (QB)"
        elif q3 or q2:
            fmt = "Knockout"
        else:
            fmt = "Single Session"

        gap = cls._calculate_gap(pole_time_str, best_lap) if (pos > 1 and pole_time_str and best_lap) else None

        return QualifyingResultEntry(
            position=pos,
            driver_id=driver_id,
            driver_name=driver_name,
            driver_code=driver_code,
            constructor_id=constructor_id,
            constructor_name=constructor_name,
            q1=q1,
            q2=q2,
            q3=q3,
            best_lap_time=best_lap,
            gap_to_pole=gap,
            session_format=fmt,
            session_code=st if st else None,
        )

    @staticmethod
    def _calculate_gap(pole_str: str, driver_str: str) -> Optional[str]:
        """Calculates delta string between driver lap time and pole lap time."""
        def parse_to_millis(time_str: str) -> Optional[int]:
            clean = time_str.strip()
            try:
                if ":" in clean:
                    parts = clean.split(":")
                    minutes = int(parts[0])
                    seconds = float(parts[1])
                    return round((minutes * 60 + seconds) * 1000)
                else:
                    return round(float(clean) * 1000)
            except Exception:
                return None

        pole_ms = parse_to_millis(pole_str)
        driver_ms = parse_to_millis(driver_str)

        if pole_ms is not None and driver_ms is not None:
            delta_ms = driver_ms - pole_ms
            if delta_ms >= 0:
                delta_sec = delta_ms / 1000.0
                return f"+{delta_sec:.3f}"
        return None
