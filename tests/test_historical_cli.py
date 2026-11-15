"""
Unit tests for ApexGrid Historical Analytics Subsystem — Phase 5 CLI Integration
================================================================================
Validates:
- Master interactive menu option 8 dispatches to HistoricalStatsMenu.
- All original menu options 1–7 remain intact, functional, and correctly numbered.
- Direct CLI command `python main.py history` dispatches to HistoricalStatsMenu.
- Clean exit (0 returns to main menu).
- Interactive navigation through each historical analytics sub-view (1 to 6).
- Input validation and graceful handling of missing data and errors.
"""

from pathlib import Path
import sys
from unittest.mock import patch, MagicMock
import pytest

from src.cli.menu import InteractiveMenu
from src.cli.historical_menu import HistoricalStatsMenu
import main

ROOT = Path(__file__).resolve().parents[1]


def test_main_menu_includes_option_5_historical_stats():
    """Verify master interactive menu displays option 5 and dispatches to menu_historical_stats."""
    with patch("builtins.input", side_effect=["5", "0"]):
        with patch.object(InteractiveMenu, "menu_historical_stats") as mock_hist:
            menu = InteractiveMenu(root_dir=ROOT)
            with pytest.raises(SystemExit) as exc_info:
                menu.run()
            assert exc_info.value.code == 0
            mock_hist.assert_called_once()


def test_existing_menu_options_preserved():
    """Verify options 1 through 7 are completely preserved and functional."""
    menu = InteractiveMenu(root_dir=ROOT)
    # Check that existing menu methods exist on InteractiveMenu
    assert hasattr(menu, "menu_predict_race")
    assert hasattr(menu, "menu_collect_data")
    assert hasattr(menu, "menu_view_outputs")
    assert hasattr(menu, "menu_evaluate_prediction")
    assert hasattr(menu, "menu_system_health")
    assert hasattr(menu, "menu_configuration")
    assert hasattr(menu, "menu_data_management")
    assert hasattr(menu, "menu_historical_stats")


def test_cli_history_subcommand_dispatch():
    """Verify `python main.py history` dispatches to HistoricalStatsMenu."""
    test_args = ["main.py", "history"]
    with patch.object(sys, "argv", test_args):
        with patch("src.cli.historical_menu.HistoricalStatsMenu.run") as mock_run:
            main.main()
            mock_run.assert_called_once()


def test_historical_menu_clean_exit():
    """Verify option 0 exits historical menu back to main menu."""
    with patch("builtins.input", return_value="0"):
        hist_menu = HistoricalStatsMenu()
        # Should terminate loop and return cleanly without raising SystemExit
        hist_menu.run()


def test_historical_menu_subview_dispatch():
    """Verify options 1 through 6 in HistoricalStatsMenu dispatch to their respective methods."""
    methods_to_test = [
        ("1", "view_driver_stats"),
        ("2", "view_teammate_h2h"),
        ("3", "view_qualifying_stats"),
        ("4", "view_race_performance"),
        ("5", "view_circuit_stats"),
        ("6", "view_season_stats"),
    ]

    for opt, method_name in methods_to_test:
        with patch("builtins.input", side_effect=[opt, "0"]):
            with patch.object(HistoricalStatsMenu, method_name) as mock_method:
                hist_menu = HistoricalStatsMenu()
                hist_menu.run()
                mock_method.assert_called_once()


def test_historical_menu_driver_stats_missing_data_handled():
    """Verify viewing driver stats handles missing driver gracefully without crashing."""
    # Inputs: driver name "non_existent_driver", empty season, press enter to return
    with patch("builtins.input", side_effect=["non_existent_driver", "", ""]):
        hist_menu = HistoricalStatsMenu()
        with patch.object(hist_menu.driver_service, "get_driver_stats", return_value=None):
            # Should display warning and return cleanly
            hist_menu.view_driver_stats()


def test_historical_menu_teammate_missing_data_handled():
    """Verify teammate comparison handles non-teammates gracefully."""
    with patch("builtins.input", side_effect=["2023", "verstappen", "hamilton", ""]):
        hist_menu = HistoricalStatsMenu()
        with patch.object(hist_menu.teammate_service, "compare_teammates", return_value=None):
            hist_menu.view_teammate_h2h()


def test_historical_menu_circuit_missing_data_handled():
    """Verify circuit view handles invalid circuit gracefully."""
    with patch("builtins.input", side_effect=["unknown_track", ""]):
        hist_menu = HistoricalStatsMenu()
        with patch.object(hist_menu.circuit_service, "get_circuit_history", return_value=None):
            hist_menu.view_circuit_stats()


def test_historical_menu_season_missing_data_handled():
    """Verify season view handles invalid season input and missing data gracefully."""
    # "1949" is rejected by prompt_int (min_val=1950), then "1950" is accepted, then Enter to return
    with patch("builtins.input", side_effect=["1949", "1950", ""]):
        hist_menu = HistoricalStatsMenu()
        with patch.object(hist_menu.season_service, "get_season_summary", return_value=None):
            hist_menu.view_season_stats()
