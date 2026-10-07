"""
Unit Tests for Lap Time and Timing Utilities
============================================
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.timing import timedelta_to_seconds, seconds_to_laptime


def test_timedelta_to_seconds_various_formats():
    # String format MM:SS.mmm
    assert abs(timedelta_to_seconds("1:32.456") - 92.456) < 1e-4

    # String format HH:MM:SS.mmm
    assert abs(timedelta_to_seconds("0:01:32.456") - 92.456) < 1e-4

    # Float directly
    assert abs(timedelta_to_seconds(92.456) - 92.456) < 1e-4

    # Pandas Timedelta
    td = pd.Timedelta(seconds=92.456)
    assert abs(timedelta_to_seconds(td) - 92.456) < 1e-4

    # None / NaN / Empty
    assert np.isnan(timedelta_to_seconds(None))
    assert np.isnan(timedelta_to_seconds(np.nan))
    assert np.isnan(timedelta_to_seconds(""))
    assert np.isnan(timedelta_to_seconds("NaT"))


def test_seconds_to_laptime():
    assert seconds_to_laptime(92.456) == "1:32.456"
    assert seconds_to_laptime(np.nan) == "No Time"
    assert seconds_to_laptime(None) == "No Time"
