"""
Unit Tests for ApexGrid Terminal UI & Motorsport Styling
========================================================
Validates:
1. Palette and color definitions.
2. Banner, header, panel, and table rendering.
3. Prompt functions, defaults, input validation, and keyboard interrupt handling.
4. Probability bar generator.
5. Podium presentation.
6. NO_COLOR environment variable compliance.
7. Fallback behavior when Rich is unavailable or output is redirected/non-TTY.
"""

import os
import sys
from unittest.mock import patch, MagicMock
import pytest

from src.cli.terminal_ui import (
    clear_screen,
    print_banner,
    print_header,
    print_success,
    print_warning,
    print_error,
    print_info,
    print_table,
    prompt_choice,
    prompt_int,
    prompt_str,
    prompt_confirm,
    render_probability_bar,
    print_podium,
    get_console,
    Colors,
    _supports_color,
    HAS_RICH,
)


def test_supports_color_respects_no_color(monkeypatch):
    """Verifies that NO_COLOR environment variable disables color support."""
    monkeypatch.setenv("NO_COLOR", "1")
    assert _supports_color() is False


def test_print_banner_renders_cleanly(capsys):
    """Verifies print_banner executes without error and displays branding and season."""
    print_banner(season=2026, status_text="Test Status")
    out = capsys.readouterr().out
    assert "ApexGrid" in out or "A P E X G R I D" in out
    assert "Predicting the Race Beyond the Grid" in out
    assert "2026" in out
    assert "Test Status" in out


def test_print_header_renders_title_and_subtitle(capsys):
    """Verifies print_header displays title and subtitle."""
    print_header("Test Section Title", "Test Subtitle Details")
    out = capsys.readouterr().out
    assert "TEST SECTION TITLE" in out
    assert "Test Subtitle Details" in out


def test_status_messages(capsys):
    """Verifies success, warning, error, and info status lines."""
    print_success("Operation verified")
    print_warning("Potential risk")
    print_error("Critical failure")
    print_info("Diagnostic log")

    out = capsys.readouterr().out
    assert "Operation verified" in out
    assert "Potential risk" in out
    assert "Critical failure" in out
    assert "Diagnostic log" in out


def test_print_table_empty_state(capsys):
    """Verifies print_table handles empty row list gracefully."""
    print_table(["Col1", "Col2"], [])
    out = capsys.readouterr().out
    assert "(No records found)" in out


def test_print_table_with_data(capsys):
    """Verifies print_table aligns and outputs rows."""
    headers = ["Pos", "Driver", "Points"]
    rows = [["1", "VER", "25"], ["2", "HAM", "18"]]
    print_table(headers, rows, align_right=[True, False, True])
    out = capsys.readouterr().out
    assert "Pos" in out
    assert "VER" in out
    assert "HAM" in out
    assert "25" in out


def test_render_probability_bar():
    """Verifies render_probability_bar formats percentages and bar widths."""
    bar_50 = render_probability_bar(50.0, width=10)
    assert "50.0%" in bar_50
    assert "█" in bar_50

    bar_0 = render_probability_bar(0.0, width=10)
    assert "0.0%" in bar_0

    bar_100 = render_probability_bar(100.0, width=10)
    assert "100.0%" in bar_100


def test_print_podium(capsys):
    """Verifies podium showcase renders P1, P2, and P3 cards."""
    p1 = {"Abbreviation": "VER", "TeamName": "Red Bull", "WinProbability": 65.4, "Top3Probability": 92.1}
    p2 = {"Abbreviation": "NOR", "TeamName": "McLaren", "WinProbability": 21.0, "Top3Probability": 84.5}
    p3 = {"Abbreviation": "LEC", "TeamName": "Ferrari", "WinProbability": 8.5, "Top3Probability": 68.0}

    print_podium(p1, p2, p3)
    out = capsys.readouterr().out
    assert "VER" in out
    assert "NOR" in out
    assert "LEC" in out


def test_prompt_choice_with_default():
    """Verifies prompt_choice accepts default on enter."""
    with patch("builtins.input", return_value=""):
        choice = prompt_choice("Choose", ["1", "2", "3"], default="2")
        assert choice == "2"


def test_prompt_int_out_of_bounds_then_valid():
    """Verifies prompt_int reprompts when value is out of bounds."""
    inputs = iter(["5", "15", "10"])
    with patch("builtins.input", lambda _: next(inputs)):
        val = prompt_int("Enter", min_val=8, max_val=12)
        assert val == 10


def test_prompt_str_allow_empty():
    """Verifies prompt_str allows empty string when configured."""
    with patch("builtins.input", return_value=""):
        res = prompt_str("Enter text", allow_empty=True)
        assert res == ""


def test_prompt_confirm():
    """Verifies prompt_confirm handles yes/no variants."""
    with patch("builtins.input", return_value="yes"):
        assert prompt_confirm("Confirm?") is True
    with patch("builtins.input", return_value="no"):
        assert prompt_confirm("Confirm?") is False


def test_missing_rich_fallback(monkeypatch, capsys):
    """Verifies that when HAS_RICH is False, terminal_ui gracefully degrades to standard library ANSI/ASCII."""
    import src.cli.terminal_ui as tui
    monkeypatch.setattr(tui, "HAS_RICH", False)

    # get_console returns None
    assert tui.get_console() is None

    # Banner renders ASCII format
    tui.print_banner(season=2026, status_text="Standard Mode")
    banner_out = capsys.readouterr().out
    assert "ApexGrid" in banner_out
    assert "Predicting the Race Beyond the Grid" in banner_out
    assert "Standard Mode" in banner_out

    # Header renders ASCII format
    tui.print_header("Fallback Header", "Fallback Subtitle")
    hdr_out = capsys.readouterr().out
    assert "FALLBACK HEADER" in hdr_out
    assert "Fallback Subtitle" in hdr_out

    # Status messages render [OK], [WARN], [ERROR], [INFO]
    tui.print_success("Fallback OK")
    tui.print_warning("Fallback Warn")
    tui.print_error("Fallback Err")
    tui.print_info("Fallback Info")
    st_out = capsys.readouterr().out
    assert "[OK] Fallback OK" in st_out
    assert "[WARN] Fallback Warn" in st_out
    assert "[ERROR] Fallback Err" in st_out
    assert "[INFO] Fallback Info" in st_out

    # Table renders ASCII borders
    tui.print_table(["Driver", "Pos"], [["VER", "1"], ["HAM", "2"]])
    tbl_out = capsys.readouterr().out
    assert "-+-" in tbl_out
    assert "VER" in tbl_out

    # Probability bar formats plain string
    bar = tui.render_probability_bar(75.0, width=10)
    assert "75.0%" in bar
    assert "[#" not in bar  # No rich style markup

    # Podium formats plain line
    tui.print_podium({"Abbreviation": "VER"}, {"Abbreviation": "NOR"}, {"Abbreviation": "LEC"})
    pod_out = capsys.readouterr().out
    assert "[PODIUM] P1: VER | P2: NOR | P3: LEC" in pod_out


def test_redirected_stdout_non_tty_fallback(capsys):
    """Verifies table and header rendering with redirected/captured stdout."""
    headers = ["Round", "Track", "Status"]
    rows = [["1", "Australia", "Completed"], ["2", "China", "Scheduled"]]
    print_table(headers, rows)
    out = capsys.readouterr().out
    assert "Round" in out
    assert "Australia" in out
    assert "China" in out


def test_prompt_keyboard_interrupt_recovery():
    """Verifies all prompts handle KeyboardInterrupt gracefully without uncaught exceptions."""
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        assert prompt_choice("Choose", ["1", "2"]) == "0"
        assert prompt_int("Enter", default=42) is None
        assert prompt_str("Enter", default="test") == ""
        assert prompt_confirm("Confirm?") is False

