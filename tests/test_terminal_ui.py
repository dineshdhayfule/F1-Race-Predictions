import os
from unittest.mock import patch
import os
import pytest

from apexgrid.cli.terminal_ui import (
    strip_ansi,
    get_terminal_width,
    get_terminal_height,
    format_menu_divider,
    print_banner,
    print_header,
    print_table,
    paginate_text,
    render_probability_bar,
    print_podium,
    Colors,
)


class MockTerminalSize:
    def __init__(self, columns: int, lines: int):
        self.columns = columns
        self.lines = lines


# =========================================================================
# 1. Terminal Dimensions & Clamping Tests
# =========================================================================

def test_strip_ansi():
    raw = f"{Colors.BOLD}{Colors.RED}ApexGrid{Colors.RESET} Formula 1"
    clean = strip_ansi(raw)
    assert clean == "ApexGrid Formula 1"
    assert strip_ansi("") == ""
    assert strip_ansi(None) == ""


def test_get_terminal_width_clamping():
    # Narrow terminal clamp (min 40)
    with patch("shutil.get_terminal_size", return_value=MockTerminalSize(25, 20)):
        assert get_terminal_width() == 40

    # Normal terminal (80)
    with patch("shutil.get_terminal_size", return_value=MockTerminalSize(80, 24)):
        assert get_terminal_width() == 80

    # Wide terminal clamp (max 120)
    with patch("shutil.get_terminal_size", return_value=MockTerminalSize(200, 50)):
        assert get_terminal_width() == 120


def test_get_terminal_width_fallback_on_error():
    with patch("shutil.get_terminal_size", side_effect=OSError("Not a tty")):
        assert get_terminal_width(default=80) == 80


def test_get_terminal_height_clamping():
    # Very short terminal clamp (min 10)
    with patch("shutil.get_terminal_size", return_value=MockTerminalSize(80, 5)):
        assert get_terminal_height() == 10

    # 15-row terminal
    with patch("shutil.get_terminal_size", return_value=MockTerminalSize(80, 15)):
        assert get_terminal_height() == 15

    # Standard 24-row terminal
    with patch("shutil.get_terminal_size", return_value=MockTerminalSize(80, 24)):
        assert get_terminal_height() == 24


def test_dynamic_resizing_between_consecutive_calls():
    with patch("shutil.get_terminal_size", return_value=MockTerminalSize(60, 20)):
        assert get_terminal_width() == 60
    with patch("shutil.get_terminal_size", return_value=MockTerminalSize(110, 35)):
        assert get_terminal_width() == 110


# =========================================================================
# 2. Banners & Headers Output Tests
# =========================================================================

def test_banner_at_different_widths(capsys):
    from apexgrid.utils.season_utils import get_active_season
    active_yr = str(get_active_season())
    # 40 columns (dynamic default season)
    with patch("shutil.get_terminal_size", return_value=MockTerminalSize(40, 20)):
        print_banner(status_text="Ready")
        captured = capsys.readouterr().out
        assert "ApexGrid" in captured or "APEXGRID" in captured or "A P E X G R I D" in captured
        assert active_yr in captured

    # 80 columns
    with patch("shutil.get_terminal_size", return_value=MockTerminalSize(80, 24)):
        print_banner(season=get_active_season(), status_text="Ready")
        captured = capsys.readouterr().out
        assert active_yr in captured

    # 120 columns
    with patch("shutil.get_terminal_size", return_value=MockTerminalSize(120, 30)):
        print_banner(season=get_active_season(), status_text="Ready")
        captured = capsys.readouterr().out
        assert active_yr in captured


def test_header_at_different_widths(capsys):
    for width in (40, 80, 120):
        with patch("shutil.get_terminal_size", return_value=MockTerminalSize(width, 24)):
            print_header("Test Title", "Test Subtitle")
            captured = capsys.readouterr().out
            assert "TEST TITLE" in captured


def test_format_menu_divider():
    with patch("shutil.get_terminal_size", return_value=MockTerminalSize(40, 20)):
        div = format_menu_divider("Core")
        clean_div = strip_ansi(div)
        assert clean_div.startswith("── Core ──")
        assert len(clean_div) <= 40

    with patch("shutil.get_terminal_size", return_value=MockTerminalSize(80, 24)):
        div = format_menu_divider("Core")
        clean_div = strip_ansi(div)
        assert len(clean_div) <= 80


# =========================================================================
# 3. Table Output, Column Fitting & Number Preservation Tests
# =========================================================================

def test_table_preserves_numeric_values_and_fits(capsys):
    headers = ["Pos", "Driver", "Team", "Grid", "ExpFinish", "Win%"]
    rows = [
        [1, "VER", "Red Bull Racing Honda RBPT", 1, "1.25", "65.0%"],
        [2, "NOR", "McLaren Mercedes F1 Team", 2, "2.40", "22.5%"],
        [3, "LEC", "Scuderia Ferrari HP", 3, "3.10", "12.5%"],
    ]

    # Normal 80-column terminal
    with patch("shutil.get_terminal_size", return_value=MockTerminalSize(80, 24)):
        print_table(headers, rows, align_right=[True, False, False, True, True, True])
        out = capsys.readouterr().out
        # Confirm numeric values are intact
        assert "1.25" in out
        assert "2.40" in out
        assert "65.0%" in out
        assert "VER" in out

    # Narrow 40-column terminal (switches to stacked or truncated view without dropping data)
    with patch("shutil.get_terminal_size", return_value=MockTerminalSize(40, 20)):
        print_table(headers, rows, align_right=[True, False, False, True, True, True])
        out = capsys.readouterr().out
        assert "1.25" in out
        assert "VER" in out
        assert "65.0%" in out


def test_table_empty_rows(capsys):
    print_table(["Pos", "Driver"], [])
    out = capsys.readouterr().out
    assert "No records found" in out


# =========================================================================
# 4. Pagination Height Awareness Tests
# =========================================================================

def test_paginate_text_short_terminal(monkeypatch, capsys):
    lines = [f"Line {i}" for i in range(25)]
    text = "\n".join(lines)

    # In a 15-row terminal, default page_size should be max(5, 15 - 4) = 11 lines
    # Simulate user immediately typing 'q' on first prompt
    inputs = iter(["q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    with patch("shutil.get_terminal_size", return_value=MockTerminalSize(80, 15)):
        paginate_text(text)
        out = capsys.readouterr().out
        # Line 0 to 10 should be printed in the first 11-line chunk
        assert "Line 0" in out
        assert "Line 10" in out
        # Line 20 should not be printed because user quit at page 1
        assert "Line 20" not in out


def test_paginate_text_explicit_page_size(capsys):
    lines = [f"Line {i}" for i in range(5)]
    text = "\n".join(lines)
    # Total lines (5) <= page_size (10), so no pagination prompt
    paginate_text(text, page_size=10)
    out = capsys.readouterr().out
    assert "Line 0" in out
    assert "Line 4" in out


# =========================================================================
# 5. Podium & Probability Bar Responsiveness Tests
# =========================================================================

def test_render_probability_bar_responsive():
    # Normal width
    with patch("shutil.get_terminal_size", return_value=MockTerminalSize(80, 24)):
        bar80 = render_probability_bar(50.0)
        assert "50.0%" in bar80

    # Narrow width
    with patch("shutil.get_terminal_size", return_value=MockTerminalSize(40, 24)):
        bar40 = render_probability_bar(50.0)
        assert "50.0%" in bar40
        assert len(strip_ansi(bar40)) <= len(strip_ansi(bar80))


def test_podium_layout_switching(capsys):
    p1 = {"Abbreviation": "VER", "Driver": "Max Verstappen", "Team": "Red Bull", "WinProbability": 60.0, "Top3Probability": 85.0}
    p2 = {"Abbreviation": "NOR", "Driver": "Lando Norris", "Team": "McLaren", "WinProbability": 25.0, "Top3Probability": 70.0}
    p3 = {"Abbreviation": "LEC", "Driver": "Charles Leclerc", "Team": "Ferrari", "WinProbability": 15.0, "Top3Probability": 60.0}

    # Wide layout (>= 78 columns)
    with patch("shutil.get_terminal_size", return_value=MockTerminalSize(90, 24)):
        print_podium(p1, p2, p3)
        out = capsys.readouterr().out
        assert "VER" in out
        assert "NOR" in out
        assert "LEC" in out

    # Compact vertical layout (< 78 columns)
    with patch("shutil.get_terminal_size", return_value=MockTerminalSize(50, 24)):
        print_podium(p1, p2, p3)
        out = capsys.readouterr().out
        assert "VER" in out
        assert "NOR" in out
        assert "LEC" in out
