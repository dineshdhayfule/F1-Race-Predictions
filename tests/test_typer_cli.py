"""
Unit Tests for ApexGrid Typer + Rich Standalone CLI Application
===============================================================
Tests command line parsing, validations, error handling, help outputs,
and mocked subcommand execution using typer.testing.CliRunner.
"""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from typer.testing import CliRunner

from src.cli.typer_app import app
from src.historical.analytics.models import (
    DriverCareerStats,
    DriverSeasonBreakdown,
    TeammateComparisonStats,
)

runner = CliRunner()


def test_default_no_args_launches_menu():
    """Verify default invocation with no arguments opens the interactive MAIN MENU directly."""
    with patch("src.cli.menu.InteractiveMenu.run", return_value=None) as mock_menu:
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert mock_menu.call_count == 1


def test_root_help():
    """Verify apexgrid --help displays help and registered subcommands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ApexGrid" in result.output
    assert "predict" in result.output
    assert "evaluate" in result.output
    assert "validate" in result.output
    assert "collect" in result.output
    assert "clean" in result.output
    assert "circuits" in result.output
    assert "health" in result.output
    assert "history" in result.output
    cmd_names = [c.name for c in app.registered_commands]
    assert set(cmd_names) == {"predict", "evaluate", "validate", "collect", "clean", "circuits", "health", "menu"}


def test_subcommand_helps():
    """Verify --help flags for all core subcommands."""
    subcommands = [
        ["predict", "--help"],
        ["evaluate", "--help"],
        ["validate", "--help"],
        ["collect", "--help"],
        ["clean", "--help"],
        ["circuits", "--help"],
        ["health", "--help"],
        ["menu", "--help"],
        ["history", "--help"],
        ["history", "driver", "--help"],
        ["history", "h2h", "--help"],
        ["history", "qualifying", "--help"],
        ["history", "race", "--help"],
        ["history", "circuit", "--help"],
        ["history", "standings", "--help"],
    ]
    for cmd in subcommands:
        result = runner.invoke(app, cmd)
        assert result.exit_code == 0, f"Failed for {cmd}: {result.output}"
        assert "Usage:" in result.output or "Options" in result.output


def test_circuits_command():
    """Verify apexgrid circuits lists circuits from tracks.json."""
    result = runner.invoke(app, ["circuits"])
    assert result.exit_code == 0
    assert "World Championship Calendar" in result.output
    assert "Grand" in result.output
    assert "austr" in result.output or "china" in result.output


def test_predict_validation_errors():
    """Verify argument and weight validation on apexgrid predict."""
    # Missing required --track
    res_no_track = runner.invoke(app, ["predict"])
    assert res_no_track.exit_code != 0

    # Invalid model architecture
    res_bad_model = runner.invoke(app, ["predict", "-t", "melbourne", "-m", "invalid_model"])
    assert res_bad_model.exit_code == 2
    assert "Invalid model architecture" in res_bad_model.output

    # Single weight provided
    res_one_weight = runner.invoke(app, ["predict", "-t", "melbourne", "--rf-weight", "60"])
    assert res_one_weight.exit_code == 2
    assert "Both --rf-weight and --xgb-weight must be supplied" in res_one_weight.output

    # Weights don't sum to 100
    res_bad_sum = runner.invoke(app, ["predict", "-t", "melbourne", "--rf-weight", "60", "--xgb-weight", "50"])
    assert res_bad_sum.exit_code == 2
    assert "Weight sum must equal 100%" in res_bad_sum.output

    # Weights provided for non-hybrid model
    res_non_hybrid = runner.invoke(app, ["predict", "-t", "melbourne", "-m", "rf", "--rf-weight", "50", "--xgb-weight", "50"])
    assert res_non_hybrid.exit_code == 2
    assert "Custom weights cannot be used with" in res_non_hybrid.output


def test_predict_mock_execution():
    """Verify apexgrid predict runs and renders results table when pipeline succeeds."""
    mock_df = pd.DataFrame([
        {
            "PredictedPosition": 1,
            "Abbreviation": "VER",
            "TeamName": "Red Bull Racing",
            "ActualStartingGrid": 1,
            "ExpectedFinish": 1.25,
            "WinProbability": 72.5,
            "Top3Probability": 95.0,
            "PointsProbability": 99.0,
            "P5": 1,
            "P95": 3,
            "RecommendedStrategy": "1-Stop M->H",
            "OptimalStops": 1,
            "StrategyDeltaSec": 0.0,
        },
        {
            "PredictedPosition": 2,
            "Abbreviation": "HAM",
            "TeamName": "Ferrari",
            "ActualStartingGrid": 2,
            "ExpectedFinish": 2.45,
            "WinProbability": 15.0,
            "Top3Probability": 80.0,
            "PointsProbability": 95.0,
            "P5": 1,
            "P95": 5,
            "RecommendedStrategy": "1-Stop M->H",
            "OptimalStops": 1,
            "StrategyDeltaSec": 1.2,
        },
    ])

    mock_results = {
        "predictions": mock_df,
        "track_name": "Albert Park Grand Prix Circuit",
        "selected_model": "hybrid (60% RF / 40% XGB)",
        "selected_mae": 1.84,
        "is_provisional_quali": False,
        "grid_meta": {"session_type": "qualifying"},
        "simulation_meta": {"p_safety_car": "65%", "rain_probability": "15%"},
        "output_dir": "outputs/melbourne",
    }

    with patch("src.pipeline.race_prediction_pipeline.RacePredictionPipeline.run", return_value=mock_results):
        result = runner.invoke(app, ["predict", "-t", "melbourne", "-y", "2026"])
        assert result.exit_code == 0
        assert "PREDICTION RESULTS" in result.output
        assert "VER" in result.output
        assert "HAM" in result.output
        assert "outputs/melbourne" in result.output


def test_clean_validation_and_execution():
    """Verify apexgrid clean validation and dispatch."""
    # Fails if no flag provided
    res_no_flags = runner.invoke(app, ["clean"])
    assert res_no_flags.exit_code == 2
    assert "Must specify a target to clean" in res_no_flags.output

    # Runs clean when flag provided
    mock_plan = {
        "operation": "Delete Prediction Outputs",
        "categories": {"Outputs": 1},
        "file_count": 1,
        "storage_formatted": "12 KB",
    }
    with patch("src.data.data_manager.DataManager.discover_prediction_outputs", return_value=["outputs/test.csv"]), \
         patch("src.data.data_manager.DataManager.build_deletion_preview", return_value=mock_plan), \
         patch("src.data.data_manager.DataManager.execute_deletion_plan", return_value={"files_deleted_count": 1, "storage_reclaimed_formatted": "12 KB"}):
        res = runner.invoke(app, ["clean", "--outputs", "--force"])
        assert res.exit_code == 0
        assert "Deletion complete" in res.output


def test_health_execution():
    """Verify apexgrid health command executes diagnostics."""
    result = runner.invoke(app, ["health"])
    assert result.exit_code == 0
    assert "ApexGrid System Health" in result.output
    assert "Python Interpreter" in result.output
    assert "Dependencies" in result.output


def test_history_driver_mock_execution():
    """Verify apexgrid history driver command with mocked service."""
    mock_stats = DriverCareerStats(
        driver_id="alonso",
        driver_name="Fernando Alonso",
        nationality="Spanish",
        seasons_active=[2001, 2003, 2004, 2005, 2006],
        total_entries=400,
        total_starts=398,
        classified_finishes=350,
        wins=32,
        podiums=106,
        qualifying_session_p1s=22,
        total_points=2267.0,
        dnf_count=75,
        seasons_breakdown=[
            DriverSeasonBreakdown(
                season=2005,
                constructor_name="Renault",
                races=19,
                starts=19,
                classified_finishes=17,
                wins=7,
                podiums=15,
                poles=6,
                points=133.0,
                dnfs=2,
                best_finish=1,
                avg_finish=2.1,
            )
        ],
    )
    with patch("src.historical.analytics.driver_analytics.DriverAnalyticsService.get_driver_stats", return_value=mock_stats):
        result = runner.invoke(app, ["history", "driver", "alonso"])
        assert result.exit_code == 0
        assert "ALONSO" in result.output
        assert "Spanish" in result.output
        assert "32" in result.output


def test_history_h2h_mock_execution():
    """Verify apexgrid history h2h command with mocked service."""
    mock_h2h = TeammateComparisonStats(
        driver1_id="verstappen",
        driver1_name="Max Verstappen",
        driver2_id="perez",
        driver2_name="Sergio Perez",
        constructor_name="Red Bull Racing",
        season=2023,
        races_cohabited=22,
        both_finished_count=20,
        race_h2h=(19, 3),
        qualifying_sessions_compared=22,
        qualifying_h2h=(20, 2),
        qualifying_coverage_note="Full modern coverage",
        wins_comparison=(19, 2),
        podiums_comparison=(21, 9),
        points_comparison=(575.0, 285.0),
        dnfs_comparison=(0, 2),
        notes=[],
    )
    with patch("src.historical.analytics.teammate_analytics.TeammateAnalyticsService.compare_teammates", return_value=mock_h2h):
        result = runner.invoke(app, ["history", "h2h", "verstappen", "perez", "-y", "2023"])
        assert result.exit_code == 0
        assert "Teammate Comparison" in result.output
        assert "Red Bull" in result.output
        assert "19" in result.output


def test_menu_switches():
    """Verify --menu and menu subcommand route to InteractiveMenu."""
    with patch("src.cli.menu.InteractiveMenu.run", return_value=None) as mock_menu:
        res1 = runner.invoke(app, ["--menu"])
        assert res1.exit_code == 0
        assert mock_menu.call_count == 1

        res2 = runner.invoke(app, ["menu"])
        assert res2.exit_code == 0
        assert mock_menu.call_count == 2
