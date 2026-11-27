"""
Dynamic Season Resolution & Off-Season Telemetry Guards
=======================================================
"""

from datetime import datetime, timezone
from typing import Optional


def get_active_season(ref_dt: Optional[datetime] = None) -> int:
    """Returns the current UTC calendar year as the active championship season."""
    return (ref_dt or datetime.now(timezone.utc)).year


class MissingSessionTelemetryError(RuntimeError):
    """Raised when no completed session telemetry exists for the requested year/round."""

    def __init__(self, year: int, round_num: Optional[int] = None):
        round_label = f" Round {round_num}" if round_num is not None else ""
        super().__init__(
            f"No completed session telemetry found for {year}{round_label}. "
            f"To run predictions on historical seasons, specify `--year <YYYY> --round <R>`."
        )
        self.year = year
        self.round_num = round_num
