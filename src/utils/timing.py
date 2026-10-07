"""
Timing utilities for F1 lap time conversions and formatting.
"""

import re
import numpy as np
import pandas as pd


def timedelta_to_seconds(val):
    """
    Convert a lap time representation (timedelta, string, or number) to float seconds.
    Returns np.nan if invalid, empty, or negative.
    """
    if pd.isna(val) or val is None:
        return np.nan

    if isinstance(val, (int, float, np.integer, np.floating)):
        val_float = float(val)
        return val_float if val_float > 0 else np.nan

    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "nat", "dnf", "dns", "dq"}:
        return np.nan

    # Try pd.to_timedelta first (handles '0 days 00:01:35.123')
    try:
        td = pd.to_timedelta(s)
        secs = td.total_seconds()
        if secs > 0:
            return secs
    except Exception:
        pass

    # Regex parse MM:SS.mmm or HH:MM:SS.mmm
    match = re.search(r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)", s)
    if match:
        h, m, sec = match.groups()
        total = (int(h) if h else 0) * 3600 + int(m) * 60 + float(sec)
        return total if total > 0 else np.nan

    try:
        f = float(s)
        return f if f > 0 else np.nan
    except Exception:
        return np.nan


def seconds_to_laptime(seconds, default="No Time"):
    """
    Convert float seconds into F1 standard M:SS.mmm string format.
    """
    if pd.isna(seconds) or seconds is None or seconds <= 0:
        return default
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}:{s:06.3f}"

