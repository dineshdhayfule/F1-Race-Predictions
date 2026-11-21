from typer.testing import CliRunner

from src.cli.typer_app import app


runner = CliRunner()


def test_root_help_lists_core_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("predict", "evaluate", "validate", "collect", "clean", "circuits", "health", "menu", "history"):
        assert command in result.output


def test_history_help_lists_subcommands():
    result = runner.invoke(app, ["history", "--help"])
    assert result.exit_code == 0
    for command in ("driver", "h2h", "qualifying", "race", "circuit", "standings"):
        assert command in result.output


def test_predict_help():
    result = runner.invoke(app, ["predict", "--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_circuits_command():
    result = runner.invoke(app, ["circuits"])
    assert result.exit_code == 0
    assert "World Championship Calendar" in result.output


def test_health_command():
    result = runner.invoke(app, ["health"])
    assert result.exit_code == 0
    assert "ApexGrid System Health" in result.output


def test_clean_requires_target():
    result = runner.invoke(app, ["clean"])
    assert result.exit_code == 2
    assert "Must specify" in result.output


def test_predict_requires_track():
    result = runner.invoke(app, ["predict"])
    assert result.exit_code != 0


def test_invalid_model_is_rejected():
    result = runner.invoke(app, ["predict", "-t", "melbourne", "-m", "invalid"])
    assert result.exit_code != 0


def test_default_menu_route():
    from unittest.mock import patch
    with patch("src.cli.menu.InteractiveMenu.run", return_value=None) as mock_run:
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert mock_run.call_count == 1


def test_predict_stage_before_race_readiness(tmp_path):
    from src.cli.menu import InteractiveMenu
    menu = InteractiveMenu(root_dir=tmp_path)
    ready, details = menu._audit_prediction_readiness(
        track_key="melbourne",
        year=2026,
        round_num=1,
        stage="before_race",
        is_sprint=False,
    )
    assert isinstance(ready, bool)
    assert isinstance(details, list)
