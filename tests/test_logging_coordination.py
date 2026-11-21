import io
import logging
import re
import sys
from unittest.mock import patch, MagicMock
import pytest

from src.utils.logging_utils import (
    get_logger,
    get_console_handler,
    SynchronizedConsoleHandler,
)


# =========================================================================
# 1. Formatting & Exact Timestamps Tests
# =========================================================================

def test_exact_timestamp_and_severity_formatting():
    handler = SynchronizedConsoleHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)

    record_info = logging.LogRecord("test", logging.INFO, "test.py", 1, "Simulation in progress", (), None)
    record_warn = logging.LogRecord("test", logging.WARNING, "test.py", 1, "Pace delta high", (), None)
    record_err = logging.LogRecord("test", logging.ERROR, "test.py", 1, "Model failure", (), None)

    msg_info = handler.format(record_info)
    msg_warn = handler.format(record_warn)
    msg_err = handler.format(record_err)

    pattern_info = r"^\d{2}:\d{2}:\d{2} \[INFO\] Simulation in progress$"
    pattern_warn = r"^\d{2}:\d{2}:\d{2} \[WARNING\] Pace delta high$"
    pattern_err = r"^\d{2}:\d{2}:\d{2} \[ERROR\] Model failure$"

    assert re.match(pattern_info, msg_info)
    assert re.match(pattern_warn, msg_warn)
    assert re.match(pattern_err, msg_err)


# =========================================================================
# 2. Duplicate Handler Prevention Tests
# =========================================================================

def test_repeated_get_logger_calls_no_duplicates():
    logger1 = get_logger("test_unique_id")
    logger2 = get_logger("test_unique_id")

    assert logger1 is logger2
    assert len(logger1.handlers) == 1
    assert isinstance(logger1.handlers[0], SynchronizedConsoleHandler)

    # Calling get_console_handler directly also returns the singleton
    assert get_console_handler() is logger1.handlers[0]


# =========================================================================
# 3. Rich Coordination with console.status() Tests
# =========================================================================

def test_logging_during_active_console_status():
    from rich.console import Console

    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=True)

    with patch("src.utils.logging_utils.get_console", return_value=test_console), \
         patch("src.utils.logging_utils._supports_color", return_value=True), \
         patch("src.utils.logging_utils.HAS_RICH", True):

        logger = get_logger("test_status_coordination")
        logger.setLevel(logging.INFO)

        with test_console.status("Simulating Grand Prix (25,000 Monte Carlo draws)..."):
            logger.info("Starting Monte Carlo simulation")
            logger.info("Executing 25,000 draws")
            logger.info("Simulation completed")

    raw_output = buf.getvalue()
    # Confirm all log messages were written
    assert "Starting Monte Carlo simulation" in raw_output
    assert "Executing 25,000 draws" in raw_output
    assert "Simulation completed" in raw_output

    # Confirm ANSI line-clearing sequences (\x1b[2K) were emitted before subsequent logs
    assert "\x1b[2K" in raw_output


def test_spinner_redraw_and_cleanup_on_exception():
    from rich.console import Console

    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=True)

    with patch("src.utils.logging_utils.get_console", return_value=test_console), \
         patch("src.utils.logging_utils._supports_color", return_value=True), \
         patch("src.utils.logging_utils.HAS_RICH", True):

        logger = get_logger("test_status_exception")

        try:
            with test_console.status("Simulating Grand Prix..."):
                logger.info("Running stage")
                raise RuntimeError("Simulated failure")
        except RuntimeError:
            logger.error("Simulation aborted due to error")

    raw_output = buf.getvalue()
    assert "Running stage" in raw_output
    assert "Simulation aborted due to error" in raw_output
    # When status exits, cursor is re-enabled (\x1b[?25h)
    assert "\x1b[?25h" in raw_output


# =========================================================================
# 4. Fallback Tests (Rich Disabled, Non-TTY, Redirection)
# =========================================================================

def test_rich_disabled_fallback(capsys):
    with patch("src.utils.logging_utils.HAS_RICH", False):
        logger = get_logger("test_rich_disabled")
        logger.info("Direct stdout fallback message")
        captured = capsys.readouterr().out
        assert "[INFO] Direct stdout fallback message" in captured


def test_non_tty_output_fallback(capsys):
    with patch("src.utils.logging_utils._supports_color", return_value=False):
        logger = get_logger("test_non_tty")
        logger.info("Non-TTY plain log line")
        captured = capsys.readouterr().out
        assert "[INFO] Non-TTY plain log line" in captured


# =========================================================================
# 5. Logger Levels and Propagation Tests
# =========================================================================

def test_logger_levels_filtering(capsys):
    with patch("src.utils.logging_utils._supports_color", return_value=False):
        logger = get_logger("test_levels")
        logger.setLevel(logging.WARNING)

        logger.info("Should not appear")
        logger.warning("Warning message appeared")
        logger.error("Error message appeared")

        captured = capsys.readouterr().out
        assert "Should not appear" not in captured
        assert "[WARNING] Warning message appeared" in captured
        assert "[ERROR] Error message appeared" in captured
