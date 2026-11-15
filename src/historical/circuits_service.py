"""
ApexGrid — Historical Circuits Service
=======================================
Dedicated service for retrieving Formula 1 circuit metadata across 1950–2026.
Retrieves metadata dynamically via HistoricalAPIClient from Jolpica's Ergast-compatible API.
Enforces:
- Strictly provider-supplied metadata (circuitId, name, locality, country, lat, long, URL).
- Missing attributes remain None (zero fabrication of length, corners, direction, or records).
- Historical circuit names and layouts are treated independently from modern tracks.json.
- In-memory processing with zero filesystem persistence.
"""

from typing import Optional, List, Dict, Any

from src.historical.api_client import HistoricalAPIClient
from src.historical.models import HistoricalCircuit
from src.utils.logging_utils import get_logger

logger = get_logger("historical_circuits_service")


class HistoricalCircuitsService:
    """
    Fetch-only service for Formula 1 circuit information.
    All data is retrieved on-demand and retained strictly in process memory.
    """

    def __init__(self, api_client: Optional[HistoricalAPIClient] = None):
        self.api_client = api_client or HistoricalAPIClient()

    def get_all_circuits(self, bypass_cache: bool = False) -> List[HistoricalCircuit]:
        """
        Retrieves all historical and modern circuits recorded in the provider archive (~77 venues).
        Uses automatic pagination to collect all entries in RAM.
        """
        try:
            circuits_raw = self.api_client.get_all_pages(
                endpoint="circuits",
                table_key="CircuitTable",
                items_key="Circuits",
                ttl_seconds=3600.0,  # 1 hour cache TTL for static circuit definitions
            )
            circuits: List[HistoricalCircuit] = []
            for c_data in circuits_raw:
                circuit = self._parse_circuit_dict(c_data)
                if circuit:
                    circuits.append(circuit)
            return circuits
        except Exception as e:
            logger.warning(f"Error fetching all historical circuits: {e}")
            return []

    def get_circuit(self, circuit_id: str, bypass_cache: bool = False) -> Optional[HistoricalCircuit]:
        """
        Retrieves metadata for a specific circuit by its provider ID (e.g. 'silverstone', 'monza', 'bremgarten').

        Args:
            circuit_id: Provider circuit identifier string.
            bypass_cache: If True, bypasses memory cache.

        Returns:
            HistoricalCircuit if found, None if unrecorded or invalid.
        """
        if not circuit_id or not str(circuit_id).strip():
            return None

        clean_id = str(circuit_id).strip().lower()
        endpoint = f"circuits/{clean_id}"

        try:
            response = self.api_client.get_json(endpoint, bypass_cache=bypass_cache, ttl_seconds=3600.0)
            mrdata = response.get("MRData", {})
            circuit_table = mrdata.get("CircuitTable", {})
            circuits_list = circuit_table.get("Circuits", [])

            if not circuits_list:
                return None

            return self._parse_circuit_dict(circuits_list[0])

        except Exception as e:
            logger.warning(f"Error fetching circuit '{clean_id}': {e}")
            return None

    def get_season_circuits(self, year: int, bypass_cache: bool = False) -> List[HistoricalCircuit]:
        """
        Retrieves all circuits that hosted a Grand Prix in a given season.
        """
        endpoint = f"{year}/circuits"
        try:
            circuits_raw = self.api_client.get_all_pages(
                endpoint=endpoint,
                table_key="CircuitTable",
                items_key="Circuits",
                ttl_seconds=1800.0,
            )
            circuits: List[HistoricalCircuit] = []
            for c_data in circuits_raw:
                circuit = self._parse_circuit_dict(c_data)
                if circuit:
                    circuits.append(circuit)
            return circuits
        except Exception as e:
            logger.warning(f"Error fetching circuits for season {year}: {e}")
            return []

    @staticmethod
    def _parse_circuit_dict(data: Dict[str, Any]) -> Optional[HistoricalCircuit]:
        """
        Parses a provider circuit dictionary into a typed HistoricalCircuit.
        Preserves provider fields; missing fields are set to None.
        """
        if not isinstance(data, dict):
            return None

        circuit_id = data.get("circuitId")
        if not circuit_id:
            return None

        # Do not synthesize circuit name from ID; preserve None if omitted by provider
        circuit_name = data.get("circuitName") or None
        url = data.get("url")

        location = data.get("Location") or {}
        locality = location.get("locality")
        country = location.get("country")

        lat = None
        long = None

        if "lat" in location and location["lat"] is not None:
            try:
                lat = float(location["lat"])
            except (ValueError, TypeError):
                lat = None

        if "long" in location and location["long"] is not None:
            try:
                long = float(location["long"])
            except (ValueError, TypeError):
                long = None

        return HistoricalCircuit(
            circuit_id=str(circuit_id),
            name=circuit_name,
            locality=locality if locality else None,
            country=country if country else None,
            latitude=lat,
            longitude=long,
            url=url if url else None,
        )
