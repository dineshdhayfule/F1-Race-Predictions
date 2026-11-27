"""
Standardized logging utilities for ApexGrid Motorsport Intelligence Platform.
"""

import logging
import sys
from typing import Optional

try:
    from apexgrid.cli.terminal_ui import get_console, HAS_RICH, _supports_color
except ImportError:
    HAS_RICH = False

    def get_console():
        return None

    def _supports_color():
        return False


class SynchronizedConsoleHandler(logging.Handler):
    """
    A logging handler that coordinates with Rich's Live/Status displays
    by routing output through the shared Rich Console instance.
    Falls back to standard sys.stdout writing when Rich is unavailable,
    disabled, or in non-TTY/redirected environments.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            console = get_console() if (HAS_RICH and _supports_color()) else None
            if console is not None:
                # console.print coordinates with active Status/Live displays,
                # clearing the spinner line, printing the log message, and redrawing the spinner.
                console.print(msg, highlight=False, markup=False, crop=False)
            else:
                sys.stdout.write(msg + "\n")
                sys.stdout.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


_SHARED_HANDLER: Optional[SynchronizedConsoleHandler] = None


def get_console_handler() -> SynchronizedConsoleHandler:
    """Returns the shared synchronized console logging handler singleton."""
    global _SHARED_HANDLER
    if _SHARED_HANDLER is None:
        _SHARED_HANDLER = SynchronizedConsoleHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )
        _SHARED_HANDLER.setFormatter(formatter)
    return _SHARED_HANDLER


def get_logger(name: str = "f1_predictor") -> logging.Logger:
    """
    Returns a configured logger with the synchronized console handler.
    Guarantees no duplicate handlers are attached on repeated calls.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.addHandler(get_console_handler())
    return logger
