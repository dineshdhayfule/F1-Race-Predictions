from typer.testing import CliRunner

from apexgrid.cli.typer_app import app


runner = CliRunner()


def test_root_help_lists_core_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("predict", "evaluate", "validate", "collect", "clean", "circuits", "health", "menu", "history"):
        assert command in result.output


def test_history_help_lists_subcommands():
    result = runner.invoke(app, ["history", "--help"])
    assert result.exit_code == 0
    for command in ("driver", "h2h", "qualifying", "race", "circuit", "standings", "session", "dominance", "records", "reliability", "eras"):
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
    with patch("apexgrid.cli.menu.InteractiveMenu.run", return_value=None) as mock_run:
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert mock_run.call_count == 1


def test_predict_stage_before_race_readiness(tmp_path):
    from apexgrid.cli.menu import InteractiveMenu
    menu = InteractiveMenu(root_dir=tmp_path)
    ready, details = menu._audit_prediction_readiness(
        track_key="melbourne",
        year=2026,
        round_num=1,
        stage="before_race",
        is_sprint=False,
    )
    assert ready is False
    assert len(details) >= 2
    statuses = [status for status, _ in details]
    assert "MISSING" in statuses
    assert any("Round 1" in msg or "Qualifying" in msg or "Grid" in msg for _, msg in details)


def test_forecast_mode_prompt_conditional_on_pre_quali_stages(tmp_path, monkeypatch):
    from apexgrid.cli.menu import InteractiveMenu

    menu = InteractiveMenu(root_dir=tmp_path)
    prompt_calls = []

    def fake_prompt_choice(prompt_text, valid_choices, default=None):
        prompt_calls.append(prompt_text)
        return "2"

    monkeypatch.setattr("apexgrid.cli.menu.prompt_choice", fake_prompt_choice)

    for stg in ["pre_fp1", "after_fp1", "after_fp2", "after_fp3", "after_sprint_qualifying", "after_sprint_race"]:
        prompt_calls.clear()
        mode = menu._prompt_forecast_mode(2026, round_num=5, track_display="Qatar", stage=stg)
        assert mode == "quali_only"
        assert len(prompt_calls) == 1

    for stg in ["after_qualifying", "before_race", "post_race"]:
        prompt_calls.clear()
        mode = menu._prompt_forecast_mode(2026, round_num=5, track_display="Qatar", stage=stg)
        assert mode == "full"
        assert len(prompt_calls) == 0


def test_post_prediction_actions_shows_qualifying_when_csv_exists(tmp_path, monkeypatch):
    import pandas as pd
    from unittest.mock import MagicMock
    from apexgrid.cli.menu import InteractiveMenu

    menu = InteractiveMenu(root_dir=tmp_path)
    quali_csv = tmp_path / "qatar_2026_qualifying_predictions.csv"
    pd.DataFrame([{
        "PredictedPosition": 1,
        "Abbreviation": "VER",
        "TeamName": "Red Bull Racing",
        "ExpectedQualifyingPosition": 1.85,
        "PoleProbability": 42.0,
        "Top3Probability": 78.0,
        "Top5Probability": 91.0,
        "P5": 1,
        "P95": 5,
    }]).to_csv(quali_csv, index=False)

    menu._display_qualifying_results = MagicMock()
    choices = iter(["5", "0"])
    monkeypatch.setattr("apexgrid.cli.menu.prompt_choice", lambda *args, **kwargs: next(choices))
    monkeypatch.setattr("apexgrid.cli.menu.prompt_str", lambda *args, **kwargs: "")

    results = {
        "output_dir": str(tmp_path),
        "track_key": "qatar",
        "track_name": "Qatar Grand Prix",
        "year": 2026,
        "stage": "after_fp2",
        "predictions": pd.DataFrame([{"Abbreviation": "VER"}]),
    }
    menu._handle_post_prediction_actions(results)
    menu._display_qualifying_results.assert_called_once()


def test_predict_quali_cli_rejects_post_qualifying_stage():
    for stg in ["after_qualifying", "before_race", "post_race"]:
        result = runner.invoke(app, ["predict-quali", "-t", "qatar", "-s", stg])
        assert result.exit_code == 2
        assert "not a pre-qualifying stage" in result.output


def test_predict_quali_cli_runs_pre_qualifying_stage(tmp_path):
    import pandas as pd
    from unittest.mock import patch

    mock_res = {
        "track_key": "qatar",
        "track_name": "Qatar Grand Prix",
        "year": 2026,
        "stage": "after_fp3",
        "selected_mae": None,
        "calibrated_sigma": 2.85,
        "output_dir": str(tmp_path),
        "predictions": pd.DataFrame([{
            "PredictedPosition": 1,
            "Abbreviation": "NOR",
            "TeamName": "McLaren",
            "ExpectedQualifyingPosition": 2.10,
            "PoleProbability": 38.5,
            "Top3Probability": 72.0,
            "Top5Probability": 88.0,
            "P5": 1,
            "P95": 6,
        }]),
    }
    with patch("apexgrid.pipeline.race_prediction_pipeline.RacePredictionPipeline.predict_qualifying", return_value=mock_res) as mock_pq:
        result = runner.invoke(app, ["predict-quali", "-t", "qatar", "-y", "2026", "-s", "after_fp3"])
        assert result.exit_code == 0
        assert mock_pq.call_count == 1
        assert "QUALIFYING & POLE PREDICTION" in result.output
        assert "NOR" in result.output


def test_evaluate_accepts_circuit_alias():
    from unittest.mock import patch
    # 'cota' resolves to 'austin' via resolve_circuit_key(); mock load_season_rounds to fail fast after track lookup
    with patch("apexgrid.data.data_loader.DataLoader.load_season_rounds", side_effect=RuntimeError("STOP_AFTER_LOOKUP")):
        result = runner.invoke(app, ["evaluate", "-t", "cota", "-y", "2026", "-r", "5"])
        assert "Track 'cota' not found" not in result.output
        assert "STOP_AFTER_LOOKUP" in result.output


def test_history_circuit_renders_track_spec_card():
    from unittest.mock import patch
    from apexgrid.historical.analytics.models import CircuitAnalyticsSummary

    mock_summary = CircuitAnalyticsSummary(
        circuit_id="losail",
        circuit_name="Lusail International Circuit",
        locality="Lusail",
        country="Qatar",
        first_gp_year=2021,
        last_gp_year=2025,
        total_races_hosted=4,
        top_winning_drivers=[("Max Verstappen", 2)],
        top_winning_constructors=[("Red Bull", 2)],
        top_podium_drivers=[("Max Verstappen", 3)],
        recent_winners=[(2025, "Max Verstappen", "Red Bull")],
    )
    with patch("apexgrid.historical.analytics.circuit_analytics.CircuitAnalyticsService.get_circuit_history", return_value=mock_summary):
        result = runner.invoke(app, ["history", "circuit", "qatar"])
        assert result.exit_code == 0
        assert "Technical Specs" in result.output
        assert "1:22.384" in result.output
        assert "16 turns" in result.output


def test_post_prediction_actions_shows_model_comparison_when_csv_exists(tmp_path, monkeypatch):
    import pandas as pd
    from unittest.mock import MagicMock
    from apexgrid.cli.menu import InteractiveMenu

    menu = InteractiveMenu(root_dir=tmp_path)
    comp_csv = tmp_path / "qatar_2026_model_comparison.csv"
    pd.DataFrame([{
        "model_name": "Validated Multi-Model (40% RF / 30% XGB / 30% Ridge)",
        "mae": 2.5324,
        "medae": 2.0100,
        "rmse": 3.1200,
        "spearman": 0.7420,
    }]).to_csv(comp_csv, index=False)

    menu._show_model_comparison = MagicMock()
    choices = iter(["5", "0"])
    monkeypatch.setattr("apexgrid.cli.menu.prompt_choice", lambda *args, **kwargs: next(choices))
    monkeypatch.setattr("apexgrid.cli.menu.prompt_str", lambda *args, **kwargs: "")

    results = {
        "output_dir": str(tmp_path),
        "track_key": "qatar",
        "track_name": "Qatar Grand Prix",
        "year": 2026,
        "stage": "after_qualifying",
        "predictions": pd.DataFrame([{"Abbreviation": "VER"}]),
    }
    menu._handle_post_prediction_actions(results)
    menu._show_model_comparison.assert_called_once()
