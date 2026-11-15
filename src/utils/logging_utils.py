"""
Standardized logging utilities for ApexGrid Motorsport Intelligence Platform.
Predicting the Race Beyond the Grid.
"""

import logging
import os
import sys

try:
    from rich.logging import RichHandler
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def get_logger(name: str = "f1_predictor") -> logging.Logger:
    """Returns a standardized logger with Rich styling for interactive TTY or standard formatter for piped output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        if HAS_RICH and is_tty and not os.environ.get("NO_COLOR"):
            handler = RichHandler(
                rich_tracebacks=True,
                show_path=False,
                markup=False,
                log_time_format="%H:%M:%S",
            )
        else:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
            handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
