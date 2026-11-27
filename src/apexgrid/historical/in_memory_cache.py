"""
ApexGrid — Process-Scoped In-Memory Cache
==========================================
Thread-safe LRU and TTL in-memory cache for historical API requests.
STRICT INVARIANT: Memory-only; zero filesystem persistence.
All cached objects are stored in process RAM and discarded upon application exit.
"""

import time
import threading
from collections import OrderedDict
from typing import Any, Optional, Dict, Tuple


class InMemoryTTLCache:
    """
    Process-scoped, thread-safe LRU cache with per-item TTL (Time-To-Live).
    Strictly in-memory: no files, SQLite, or disk directories are ever touched.
    """

    def __init__(self, default_ttl_seconds: float = 1800.0, max_entries: int = 500):
        """
        Args:
            default_ttl_seconds: Default expiration time in seconds (30 minutes).
            max_entries: Maximum number of entries before LRU eviction.
        """
        self._default_ttl = float(default_ttl_seconds)
        self._max_entries = int(max_entries)
        self._lock = threading.Lock()
        # Storage: key -> (value, expire_timestamp)
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieves an item from cache if present and not expired.
        Moves accessed item to end of LRU order.
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            val, expire_at = self._cache[key]
            now = time.time()

            if now > expire_at:
                # Expired: remove and count as miss
                del self._cache[key]
                self._misses += 1
                return None

            # Refresh LRU position
            self._cache.move_to_end(key)
            self._hits += 1
            return val

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        """
        Stores an item in memory with expiration. Evicts oldest item if capacity is reached.
        """
        ttl = float(ttl_seconds) if ttl_seconds is not None else self._default_ttl
        expire_at = time.time() + ttl

        with self._lock:
            if key in self._cache:
                self._cache[key] = (value, expire_at)
                self._cache.move_to_end(key)
                return

            # Check capacity and evict LRU
            if len(self._cache) >= self._max_entries:
                # Pop oldest (FIFO/LRU)
                self._cache.popitem(last=False)

            self._cache[key] = (value, expire_at)

    def clear(self) -> None:
        """Purges all entries from memory."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def size(self) -> int:
        """Returns the current number of cached items."""
        with self._lock:
            return len(self._cache)

    def stats(self) -> Dict[str, Any]:
        """Returns cache telemetry."""
        with self._lock:
            total_lookups = self._hits + self._misses
            hit_ratio = (self._hits / total_lookups) if total_lookups > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_entries": self._max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": hit_ratio,
            }
