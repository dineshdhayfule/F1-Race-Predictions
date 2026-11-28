"""Lightweight network reachability helpers for ApexGrid CLI."""

import socket
from typing import Tuple

_PROBE_TARGETS: Tuple[Tuple[str, int], ...] = (
    ("api.jolpi.ca", 443),
    ("1.1.1.1", 443),
    ("8.8.8.8", 53),
)


def is_internet_available(timeout: float = 0.5) -> bool:
    """
    Fast, non-blocking TCP reachability probe for CLI online/offline status.
    Returns True if any probe endpoint accepts a connection within `timeout` seconds.
    """
    eff_timeout = min(max(float(timeout), 0.05), 0.5)
    for host, port in _PROBE_TARGETS:
        try:
            with socket.create_connection((host, port), timeout=eff_timeout):
                return True
        except (socket.timeout, OSError):
            continue
    return False
