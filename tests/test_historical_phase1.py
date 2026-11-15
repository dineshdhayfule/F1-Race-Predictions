"""
Unit tests for ApexGrid Historical Analytics Subsystem — Phase 1 Foundation
===========================================================================
Tests API client, in-memory cache, rate limiter, data models, and zero disk persistence.
All tests are fast and mocked (< 5 seconds total).
"""

import time
import json
import threading
from datetime import datetime, timezone, timedelta
from email.utils import format_datetime
import pytest
from unittest.mock import patch, MagicMock

from src.historical.models import (
    calculate_positions_gained,
    interpret_grid_position,
    RaceResultEntry,
    StartingGridResultEntry,
    SeasonSummary,
)
from src.historical.in_memory_cache import InMemoryTTLCache
from src.historical.rate_limiter import RateLimiter
from src.historical.api_client import HistoricalAPIClient, HistoricalAPIError


# -----------------------------------------------------------------------------
# 1. Models & Calculations Tests
# -----------------------------------------------------------------------------

def test_positions_gained_calculation():
    """Verify positions gained is never calculated when grid or finish is missing or invalid."""
    # Standard valid gains
    assert calculate_positions_gained(grid_pos=10, finish_pos=4) == 6
    assert calculate_positions_gained(grid_pos=2, finish_pos=5) == -3
    assert calculate_positions_gained(grid_pos=1, finish_pos=1) == 0

    # Non-standard or missing grid positions (e.g. pit lane / unclassified == 0 or None)
    assert calculate_positions_gained(grid_pos=None, finish_pos=5) is None
    assert calculate_positions_gained(grid_pos=0, finish_pos=5) is None
    assert calculate_positions_gained(grid_pos=-1, finish_pos=5) is None

    # Unclassified finish positions
    assert calculate_positions_gained(grid_pos=5, finish_pos=None) is None
    assert calculate_positions_gained(grid_pos=5, finish_pos=0) is None


def test_interpret_grid_position():
    """Verify raw grid value 0 is NOT automatically labelled pit-lane start, but unclassified/non-standard."""
    pos, status = interpret_grid_position("0")
    assert pos is None
    assert "Unclassified / Non-standard" in status
    assert "(0)" in status

    # Standard grid slot
    pos, status = interpret_grid_position("5")
    assert pos == 5
    assert status == "Grid P5"

    # Missing / None
    pos, status = interpret_grid_position(None)
    assert pos is None
    assert "unrecorded" in status.lower()


def test_float_points_support():
    """Verify fractional points (e.g. 1954 8.14 pts or 1950s shared drives) are preserved without truncation."""
    entry = RaceResultEntry(
        position=1,
        position_text="1",
        driver_id="gonzalez",
        driver_name="Jose Froilan Gonzalez",
        driver_number="12",
        constructor_id="ferrari",
        constructor_name="Ferrari",
        grid_position=2,
        raw_grid_value="2",
        positions_gained=1,
        race_time_or_gap="2:56:14",
        laps_completed=90,
        points=8.14,
        status="Finished",
    )
    assert entry.points == 8.14
    assert isinstance(entry.points, float)


def test_season_summary_in_progress():
    """Verify 2026 / in-progress season is correctly flagged as provisional."""
    # Completed historical season
    s_past = SeasonSummary.create(year=2024, total_rounds=24, completed_rounds=24)
    assert not s_past.is_in_progress
    assert "Completed Championship" in s_past.status_display

    # Active in-progress season
    s_active = SeasonSummary.create(year=2026, total_rounds=24, completed_rounds=3)
    assert s_active.is_in_progress
    assert "In-Progress / Provisional" in s_active.status_display


# -----------------------------------------------------------------------------
# 2. In-Memory Cache Tests (Zero Disk Persistence)
# -----------------------------------------------------------------------------

def test_in_memory_cache_hit_and_miss():
    cache = InMemoryTTLCache(default_ttl_seconds=60.0, max_entries=10)
    assert cache.get("nonexistent") is None
    assert cache.size() == 0

    cache.set("key1", {"data": 123})
    assert cache.size() == 1
    assert cache.get("key1") == {"data": 123}

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_in_memory_cache_ttl_expiration():
    cache = InMemoryTTLCache(default_ttl_seconds=0.05, max_entries=10)
    cache.set("short_lived", "value", ttl_seconds=0.05)
    assert cache.get("short_lived") == "value"

    time.sleep(0.08)
    assert cache.get("short_lived") is None
    assert cache.size() == 0


def test_in_memory_cache_lru_eviction():
    cache = InMemoryTTLCache(default_ttl_seconds=60.0, max_entries=3)
    cache.set("k1", 1)
    cache.set("k2", 2)
    cache.set("k3", 3)
    assert cache.size() == 3

    # Access k1 to make it recently used
    _ = cache.get("k1")

    # Add k4 -> should evict k2 (oldest unaccessed)
    cache.set("k4", 4)
    assert cache.size() == 3
    assert cache.get("k2") is None
    assert cache.get("k1") == 1
    assert cache.get("k3") == 3
    assert cache.get("k4") == 4


# -----------------------------------------------------------------------------
# 3. Rate Limiter Tests
# -----------------------------------------------------------------------------

def test_rate_limiter_min_interval():
    # Min interval 0.05s for fast test
    limiter = RateLimiter(max_per_second=20.0, min_interval_seconds=0.05)
    start = time.time()
    limiter.wait_for_slot()
    limiter.wait_for_slot()
    elapsed = time.time() - start
    assert elapsed >= 0.045


def test_rate_limiter_retry_after_numeric_and_http_date():
    """Verify Retry-After parses numeric seconds, future HTTP-dates, past HTTP-dates, and invalid/missing values."""
    limiter = RateLimiter(max_per_second=10.0, min_interval_seconds=0.01)

    # 1. Numeric seconds
    delay_num = limiter.handle_rate_limit_response("0.1")
    assert delay_num == 0.1
    assert limiter.get_status()["in_backoff"] is True

    # 2. Valid future HTTP-date (e.g. 10 seconds in future)
    future_time = datetime.now(timezone.utc) + timedelta(seconds=10)
    future_date_str = format_datetime(future_time)
    delay_future = limiter.handle_rate_limit_response(future_date_str)
    assert 8.0 <= delay_future <= 12.0

    # 3. Past HTTP-date (should clamp to 0.0s delay safely)
    past_time = datetime.now(timezone.utc) - timedelta(seconds=10)
    past_date_str = format_datetime(past_time)
    delay_past = limiter.handle_rate_limit_response(past_date_str)
    assert delay_past == 0.0

    # 4. Invalid header string -> safe 5.0s fallback
    delay_invalid = limiter.handle_rate_limit_response("not-a-valid-date-or-number")
    assert delay_invalid == 5.0

    # 5. Missing / None header -> safe 2.0s fallback
    delay_missing = limiter.handle_rate_limit_response(None)
    assert delay_missing == 2.0

    # 6. Empty header string -> safe 2.0s fallback
    delay_empty = limiter.handle_rate_limit_response("   ")
    assert delay_empty == 2.0


def test_rate_limiter_concurrency_thread_safety():
    """Verify simultaneous threads cannot violate the rate limit and all slots are allocated correctly."""
    min_interval = 0.01
    limiter = RateLimiter(max_per_second=100.0, max_per_hour=50, min_interval_seconds=min_interval)

    num_threads = 8
    timestamps = []
    lock = threading.Lock()

    def worker():
        limiter.wait_for_slot()
        with lock:
            timestamps.append(time.time())

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2.0)

    assert len(timestamps) == num_threads
    # Verify spacing between consecutive dispatched requests respects min_interval
    timestamps.sort()
    for i in range(1, len(timestamps)):
        delta = timestamps[i] - timestamps[i - 1]
        assert delta >= min_interval - 0.005  # allow small jitter margin

    # Verify hourly counter tracked all requests
    status = limiter.get_status()
    assert status["requests_last_hour"] == num_threads


# -----------------------------------------------------------------------------
# 4. API Client & Pagination Tests
# -----------------------------------------------------------------------------

def test_api_client_custom_user_agent_and_cache_bypass():
    mock_resp = json.dumps({"MRData": {"total": "1", "RaceTable": {"Races": []}}}).encode("utf-8")
    cache = InMemoryTTLCache()
    client = HistoricalAPIClient(cache=cache)

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_http = MagicMock()
        mock_http.read.return_value = mock_resp
        mock_http.status = 200
        mock_http.__enter__.return_value = mock_http
        mock_urlopen.return_value = mock_http

        # First call -> network request
        data1 = client.get_json("2024/results")
        assert data1["MRData"]["total"] == "1"
        assert mock_urlopen.call_count == 1

        # Second call -> in-memory cache hit, zero network calls!
        data2 = client.get_json("2024/results")
        assert data2 == data1
        assert mock_urlopen.call_count == 1

        # Verify User-Agent passed in Request
        req_arg = mock_urlopen.call_args[0][0]
        assert "ApexGrid-F1" in req_arg.headers["User-agent"]


def test_api_client_pagination_accumulation():
    """Verify MRData pagination automatically retrieves and accumulates multiple pages in RAM."""
    client = HistoricalAPIClient(cache=InMemoryTTLCache())

    page1 = {"MRData": {"total": "5", "limit": "2", "offset": "0", "CircuitTable": {"Circuits": [{"id": "c1"}, {"id": "c2"}]}}}
    page2 = {"MRData": {"total": "5", "limit": "2", "offset": "2", "CircuitTable": {"Circuits": [{"id": "c3"}, {"id": "c4"}]}}}
    page3 = {"MRData": {"total": "5", "limit": "2", "offset": "4", "CircuitTable": {"Circuits": [{"id": "c5"}]}}}

    responses = [page1, page2, page3]

    def mock_get_json(endpoint, params=None, ttl_seconds=None, bypass_cache=False):
        offset = params.get("offset", 0)
        if offset == 0:
            return page1
        elif offset == 2:
            return page2
        else:
            return page3

    with patch.object(client, "get_json", side_effect=mock_get_json):
        all_circuits = client.get_all_pages(
            endpoint="circuits",
            table_key="CircuitTable",
            items_key="Circuits",
            page_size=2
        )
        assert len(all_circuits) == 5
        assert [c["id"] for c in all_circuits] == ["c1", "c2", "c3", "c4", "c5"]


# -----------------------------------------------------------------------------
# 5. Zero Filesystem Persistence Assertion
# -----------------------------------------------------------------------------

def test_zero_disk_persistence_enforcement():
    """Verify that no file write operations occur during historical subsystem operations."""
    import builtins
    original_open = builtins.open

    write_calls = []

    def guard_open(file, mode="r", *args, **kwargs):
        if any(m in mode for m in ("w", "a", "x", "+")):
            write_calls.append((file, mode))
            raise AssertionError(f"PROHIBITED: Historical subsystem attempted to write to disk: {file} (mode {mode})")
        return original_open(file, mode, *args, **kwargs)

    with patch("builtins.open", side_effect=guard_open):
        # Execute cache operations
        cache = InMemoryTTLCache()
        cache.set("test_key", {"mock": "data"})
        _ = cache.get("test_key")
        cache.clear()

        # Execute client cache-hit cycle
        client = HistoricalAPIClient(cache=cache)
        client.cache.set("https://api.jolpi.ca/ergast/f1/2026.json", {"MRData": {"total": "0"}})
        _ = client.get_json("2026")

    # Assert no write calls were intercepted
    assert len(write_calls) == 0, f"Detected disk write operations: {write_calls}"
