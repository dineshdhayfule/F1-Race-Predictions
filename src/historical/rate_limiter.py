"""
ApexGrid — Historical API Rate Limiter
=======================================
Thread-safe rate limiter enforcing Jolpica API compliance:
1. Maximum 4 requests per second (minimum 0.25s interval between requests).
2. Maximum 500 requests per hour for unauthenticated access.
3. Explicit HTTP 429 / Retry-After backoff handling.
"""

import time
import threading
from collections import deque
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional


class RateLimiter:
    """
    Enforces per-second and hourly rate limits, plus external backoff on HTTP 429.
    """

    def __init__(
        self,
        max_per_second: float = 4.0,
        max_per_hour: int = 500,
        min_interval_seconds: float = 0.25,
    ):
        self.max_per_second = float(max_per_second)
        self.max_per_hour = int(max_per_hour)
        self.min_interval = float(min_interval_seconds)

        self._lock = threading.Lock()
        self._last_request_time: float = 0.0
        # Rolling timestamps of requests in the last hour
        self._hourly_requests: deque = deque()
        # Backoff deadline until which requests must wait
        self._backoff_until: float = 0.0

    def wait_for_slot(self) -> None:
        """
        Blocks until a request can be dispatched within rate limits.
        """
        while True:
            with self._lock:
                now = time.time()

                # 1. Check if we are in an external 429 backoff
                if now < self._backoff_until:
                    sleep_time = self._backoff_until - now
                else:
                    # 2. Clean up requests older than 1 hour (3600s)
                    one_hour_ago = now - 3600.0
                    while self._hourly_requests and self._hourly_requests[0] < one_hour_ago:
                        self._hourly_requests.popleft()

                    # 3. Check hourly limit
                    if len(self._hourly_requests) >= self.max_per_hour:
                        oldest = self._hourly_requests[0]
                        sleep_time = (oldest + 3600.0) - now + 0.05
                    else:
                        # 4. Check per-second / min_interval limit
                        time_since_last = now - self._last_request_time
                        if time_since_last < self.min_interval:
                            sleep_time = self.min_interval - time_since_last
                        else:
                            # Slot acquired!
                            self._last_request_time = now
                            self._hourly_requests.append(now)
                            return

            # Sleep outside the lock so other threads can inspect state
            if sleep_time > 0:
                time.sleep(min(sleep_time, 1.0))

    def handle_rate_limit_response(self, retry_after_header: Optional[str] = None) -> float:
        """
        Called when HTTP 429 is encountered. Sets backoff period.
        Supports both numeric seconds and RFC 7231 / RFC 9110 HTTP-date strings.

        Args:
            retry_after_header: Raw string from 'Retry-After' response header.

        Returns:
            float: Number of seconds backed off.
        """
        backoff_sec = 2.0  # default backoff when header is omitted
        if retry_after_header is not None and str(retry_after_header).strip() != "":
            raw_str = str(retry_after_header).strip()
            # 1. Try numeric delay (seconds)
            try:
                backoff_sec = max(float(raw_str), 0.0)
            except ValueError:
                # 2. Try HTTP-date format (e.g. "Wed, 21 Oct 2026 07:28:00 GMT")
                try:
                    dt = parsedate_to_datetime(raw_str)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    now_utc = datetime.now(timezone.utc)
                    delta_seconds = (dt - now_utc).total_seconds()
                    backoff_sec = max(delta_seconds, 0.0)
                except Exception:
                    # Invalid/unparseable header fallback
                    backoff_sec = 5.0

        with self._lock:
            self._backoff_until = max(self._backoff_until, time.time() + backoff_sec)
        return backoff_sec

    def get_status(self) -> dict:
        """Returns current rate limiter statistics."""
        with self._lock:
            now = time.time()
            one_hour_ago = now - 3600.0
            recent_count = sum(1 for t in self._hourly_requests if t >= one_hour_ago)
            return {
                "requests_last_hour": recent_count,
                "hourly_limit": self.max_per_hour,
                "in_backoff": now < self._backoff_until,
                "backoff_remaining": max(0.0, self._backoff_until - now),
            }
