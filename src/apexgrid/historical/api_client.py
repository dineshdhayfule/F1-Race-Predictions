"""
ApexGrid — Historical F1 Fetch-Only API Client
===============================================
Zero-persistence, fetch-only HTTP client for Jolpica-F1 and Ergast API endpoints.
Enforces:
- Descriptive User-Agent header containing app name and version.
- Process-scoped in-memory cache to avoid redundant network requests.
- Strict rate limiting (max 4 req/sec, max 500 req/hour).
- Exponential backoff and Retry-After handling for HTTP 429.
- Ergast MRData automatic pagination (total, limit, offset).
- ZERO disk writes: all responses live strictly in process memory.
"""

import json
import time
import urllib.request
import urllib.error
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse
from typing import Dict, Any, Optional, List, Tuple

from apexgrid.historical.in_memory_cache import InMemoryTTLCache
from apexgrid.historical.rate_limiter import RateLimiter
from apexgrid.utils.logging_utils import get_logger

logger = get_logger("historical_api_client")

DEFAULT_USER_AGENT = "ApexGrid-F1/1.0 (Motorsport Intelligence Platform; fetch-only)"
DEFAULT_BASE_URL = "https://api.jolpi.ca/ergast/f1"


class HistoricalAPIError(Exception):
    """Base exception for historical API retrieval errors."""
    pass


class RateLimitExceededError(HistoricalAPIError):
    """Raised when rate limits are exhausted and maximum retries fail."""
    pass


class HistoricalAPIClient:
    """
    Fetch-only HTTP client for retrieving historical F1 data in RAM.
    Does not write or persist any response to disk.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        user_agent: str = DEFAULT_USER_AGENT,
        cache: Optional[InMemoryTTLCache] = None,
        rate_limiter: Optional[RateLimiter] = None,
        timeout_seconds: float = 10.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.cache = cache or InMemoryTTLCache()
        self.rate_limiter = rate_limiter or RateLimiter()
        self.timeout = float(timeout_seconds)
        self.max_retries = int(max_retries)

    def _build_url(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Constructs full URL with normalized query parameters."""
        endpoint = endpoint.strip("/")
        if not endpoint.endswith(".json"):
            endpoint = f"{endpoint}.json"

        full_url = f"{self.base_url}/{endpoint}"
        if params:
            query_str = urlencode({k: v for k, v in params.items() if v is not None})
            if query_str:
                full_url = f"{full_url}?{query_str}"
        return full_url

    def get_json(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[float] = None,
        bypass_cache: bool = False,
    ) -> Dict[str, Any]:
        """
        Fetches JSON from the API or returns from in-memory cache.
        Strictly in-memory: no files are created or read from disk.
        """
        url = self._build_url(endpoint, params)

        # 1. Check in-memory cache
        if not bypass_cache:
            cached_data = self.cache.get(url)
            if cached_data is not None:
                logger.debug(f"Memory cache HIT: {url}")
                return cached_data

        # 2. Fetch with rate limiting & retries
        data = self._fetch_with_retries(url)

        # 3. Store in memory cache
        if data is not None:
            self.cache.set(url, data, ttl_seconds=ttl_seconds)

        return data

    def _fetch_with_retries(self, url: str) -> Dict[str, Any]:
        """Dispatches HTTP GET with backoff and retry handling."""
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }

        attempt = 0
        backoff = 1.0

        while attempt < self.max_retries:
            attempt += 1
            # Respect Jolpica rate limiter
            self.rate_limiter.wait_for_slot()

            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    status_code = response.status
                    raw_bytes = response.read()
                    data = json.loads(raw_bytes.decode("utf-8"))
                    return data

            except urllib.error.HTTPError as e:
                if e.code == 429:
                    retry_after = e.headers.get("Retry-After")
                    delay = self.rate_limiter.handle_rate_limit_response(retry_after)
                    logger.warning(f"HTTP 429 Rate limited for {url}. Waiting {delay:.1f}s before retry {attempt}/{self.max_retries}...")
                    time.sleep(delay)
                    continue
                elif e.code in (500, 502, 503, 504):
                    logger.warning(f"HTTP {e.code} for {url}. Retrying ({attempt}/{self.max_retries}) in {backoff:.1f}s...")
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                elif e.code == 404:
                    # Endpoint not found or season not run
                    logger.debug(f"HTTP 404 Not Found: {url}")
                    return {"MRData": {"total": "0"}}
                else:
                    raise HistoricalAPIError(f"HTTP {e.code} error requesting {url}: {e.reason}") from e

            except urllib.error.URLError as e:
                logger.warning(f"Network error connecting to {url}: {e.reason}. Retry {attempt}/{self.max_retries}...")
                time.sleep(backoff)
                backoff *= 2.0
                continue
            except (json.JSONDecodeError, TimeoutError) as e:
                logger.warning(f"JSON decode/timeout error for {url}: {e}. Retry {attempt}/{self.max_retries}...")
                time.sleep(backoff)
                backoff *= 2.0
                continue

        raise HistoricalAPIError(f"Failed to fetch {url} after {self.max_retries} retries.")

    def get_all_pages(
        self,
        endpoint: str,
        table_key: str,
        items_key: str,
        params: Optional[Dict[str, Any]] = None,
        page_size: int = 100,
        ttl_seconds: Optional[float] = None,
        bypass_cache: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Handles Ergast pagination transparently using MRData total, limit, and offset.
        All accumulated items reside strictly in process memory.

        Args:
            endpoint: Base endpoint (e.g., '1950/results', 'circuits')
            table_key: Outer table key inside MRData (e.g. 'RaceTable', 'CircuitTable')
            items_key: Inner list key (e.g. 'Races', 'Circuits')
            params: Optional query params
            page_size: Limit per page (default 100, Jolpica recommended max)
            ttl_seconds: Optional TTL cache duration
            bypass_cache: Whether to bypass memory cache and force network fetch
        """
        all_items: List[Dict[str, Any]] = []
        offset = 0
        req_params = dict(params or {})
        req_params["limit"] = page_size

        while True:
            req_params["offset"] = offset
            response = self.get_json(
                endpoint,
                params=req_params,
                ttl_seconds=ttl_seconds,
                bypass_cache=bypass_cache,
            )

            mrdata = response.get("MRData", {})
            try:
                total = int(mrdata.get("total", 0))
            except (ValueError, TypeError):
                total = 0

            try:
                limit_applied = int(mrdata.get("limit", page_size))
            except (ValueError, TypeError):
                limit_applied = page_size

            if limit_applied <= 0:
                limit_applied = page_size

            table = mrdata.get(table_key, {})
            items = table.get(items_key, [])
            if not items:
                break

            all_items.extend(items)

            # In Ergast/Jolpica, offset advances by the page limit (the number of indexed records per page),
            # NOT by the outer container count (len(items)), because MRData.total counts nested leaf items
            # (e.g. Results, QualifyingResults) when querying race endpoints.
            offset += limit_applied

            if total > 0 and offset >= total:
                break

            # Fallback for missing/malformed total metadata (total <= 0):
            # Inspect records returned in this page to safely detect the final page.
            if total <= 0:
                records_in_page = 0
                for it in items:
                    if isinstance(it, dict):
                        for k in ("Results", "QualifyingResults", "SprintResults", "Laps", "PitStops"):
                            if k in it and isinstance(it[k], list):
                                records_in_page += len(it[k])
                                break
                        else:
                            records_in_page += 1
                    else:
                        records_in_page += 1

                if records_in_page == 0 or records_in_page < limit_applied:
                    break

        return all_items
