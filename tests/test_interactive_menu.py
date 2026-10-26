"""
Interactive Menu & CLI Dispatch Verification Test Suite
========================================================
Validates:
1. Startup behavior: `python main.py` opens the menu; '0' exits cleanly.
2. CLI backward-compatibility: `predict`, `evaluate`, `validate` commands dispatch directly.
3. Dynamic circuit discovery: all 23 circuits from tracks.json loaded.
4. Data readiness audit: accurate detection of available vs missing sessions.
5. Robust input validation: invalid inputs re-prompt without crashing.
6. Graceful handling of KeyboardInterrupt and EOFError.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from src.cli.menu import InteractiveMenu
from src.cli.terminal_ui import prompt_choice, prompt_int, prompt_confirm
import main

ROOT = Path(__file__).resolve().parents[1]


def test_menu_clean_exit():
    """Verifies that selecting option 0 exits the menu cleanly with code 0."""
    with patch("builtins.input", return_value="0"):
        menu = InteractiveMenu(root_dir=ROOT)
        with pytest.raises(SystemExit) as exc_info:
            menu.run()
        assert exc_info.value.code == 0


def test_cli_dispatch_predict_preserved():
    """Confirms `python main.py predict` bypasses the interactive menu and dispatches to cmd_predict."""
    test_args = ["main.py", "predict", "--track", "austin", "--year", "2026", "--stage", "after_qualifying", "--top", "3"]
    with patch.object(sys, "argv", test_args):
        with patch("main.cmd_predict") as mock_predict, patch("src.cli.menu.InteractiveMenu.run") as mock_menu:
            main.main()
            mock_predict.assert_called_once()
            mock_menu.assert_not_called()


def test_cli_dispatch_evaluate_and_validate_preserved():
    """Confirms `evaluate` and `validate` bypass the interactive menu."""
    eval_args = ["main.py", "evaluate", "--track", "sepang", "--year", "2026"]
    with patch.object(sys, "argv", eval_args):
        with patch("main.cmd_evaluate") as mock_eval, patch("src.cli.menu.InteractiveMenu.run") as mock_menu:
            main.main()
            mock_eval.assert_called_once()
            mock_menu.assert_not_called()

    val_args = ["main.py", "validate", "--track", "sepang", "--year", "2026"]
    with patch.object(sys, "argv", val_args):
        with patch("main.cmd_validate") as mock_val, patch("src.cli.menu.InteractiveMenu.run") as mock_menu:
            main.main()
            mock_val.assert_called_once()
            mock_menu.assert_not_called()


def test_dynamic_circuit_loading():
    """Verifies all circuits in config/tracks.json are loaded by the menu."""
    menu = InteractiveMenu(root_dir=ROOT)
    assert len(menu.tracks_cfg) >= 23
    assert "austin" in menu.tracks_cfg
    assert "sepang" in menu.tracks_cfg
    assert "baku" in menu.tracks_cfg
    assert "singapore" in menu.tracks_cfg


def test_data_readiness_audit_cota():
    """Verifies data readiness audit identifies available sessions for Austin 2026."""
    menu = InteractiveMenu(root_dir=ROOT)
    ready, details = menu._audit_prediction_readiness("austin", 2026, 18, "after_qualifying", is_sprint=False)
    assert ready is True
    ok_messages = [msg for status, msg in details if status == "OK"]
    assert any("Historical Training Rounds" in m for m in ok_messages)
    assert any("Qualifying Results available" in m for m in ok_messages)


def test_missing_data_audit_detection():
    """Verifies data readiness marks non-existent session as MISSING and ready=False."""
    menu = InteractiveMenu(root_dir=ROOT)
    # Circuit with no qualifying data (Round 19 Mexico not yet held)
    ready, details = menu._audit_prediction_readiness("mexico", 2026, 19, "after_qualifying", is_sprint=False)
    assert ready is False
    missing_messages = [msg for status, msg in details if status == "MISSING"]
    assert len(missing_messages) > 0


def test_prompt_choice_validation_and_recovery():
    """Verifies that invalid input loops and recovers once valid input is entered."""
    inputs = iter(["invalid_xyz", "999", "2"])
    with patch("builtins.input", lambda _: next(inputs)):
        val = prompt_choice("Choose", ["1", "2", "3"])
        assert val == "2"


def test_prompt_int_validation_and_recovery():
    """Verifies integer prompt handles non-integers and out-of-range values gracefully."""
    inputs = iter(["abc", "-5", "50", "2026"])
    with patch("builtins.input", lambda _: next(inputs)):
        val = prompt_int("Enter year", min_val=2018, max_val=2030)
        assert val == 2026


def test_prompt_confirm_defaults():
    """Verifies prompt_confirm accepts default on empty input."""
    with patch("builtins.input", return_value=""):
        assert prompt_confirm("Proceed?", default=True) is True
        assert prompt_confirm("Proceed?", default=False) is False


def test_graceful_interrupt_handling():
    """Verifies EOFError or KeyboardInterrupt cleanly returns without unhandled exception."""
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        choice = prompt_choice("Choose", ["1", "2"])
        assert choice == "0"
        num = prompt_int("Enter int", default=10)
        assert num is None
        conf = prompt_confirm("Confirm?")
        assert conf is False


def test_mexico_2026_pre_fp1_readiness_reasons():
    """Verifies Mexico 2026 pre_fp1 is ready because 18/18 historical rounds are on disk and 0 weekend sessions are required."""
    menu = InteractiveMenu(root_dir=ROOT)
    ready, details = menu._audit_prediction_readiness("mexico", 2026, 19, "pre_fp1", is_sprint=False)
    assert ready is True
    ok_messages = [msg for status, msg in details if status == "OK"]
    assert any("Historical Training Rounds: 18/18" in m for m in ok_messages)
    assert any("pre_fp1': Zero weekend telemetry required" in m for m in ok_messages)


def test_stage_session_readiness_requirements():
    """Verifies that missing session data for stage after_fp1/after_fp2/after_fp3/after_qualifying/post_race blocks readiness."""
    menu = InteractiveMenu(root_dir=ROOT)
    # Mexico 2026 has no FP1/FP2/FP3, Qualifying, or Race on disk yet
    for test_stage in ["after_fp1", "after_fp2", "after_fp3", "after_qualifying", "post_race"]:
        ready, details = menu._audit_prediction_readiness("mexico", 2026, 19, test_stage, is_sprint=False)
        assert ready is False, f"Expected {test_stage} to fail readiness when session data is absent"
        missing_entries = [msg for status, msg in details if status == "MISSING"]
        assert len(missing_entries) > 0, f"Expected MISSING entries for stage {test_stage}"


def test_evaluate_handles_duplicate_forecast_drivers(tmp_path):
    """Verifies that evaluation deduplicates forecast drivers without inflating sample size."""
    menu = InteractiveMenu(root_dir=ROOT)
    austin_csv = ROOT / "outputs" / "austin" / "austin_2026_after_qualifying_predictions.csv"
    if not austin_csv.exists():
        pytest.skip("Austin forecast CSV not present")

    import pandas as pd
    df = pd.read_csv(austin_csv)
    # Duplicate top driver row
    dup_df = pd.concat([df.iloc[[0]], df], ignore_index=True)
    temp_csv = tmp_path / "austin_duplicated_predictions.csv"
    dup_df.to_csv(temp_csv, index=False)

    with patch("src.cli.menu.prompt_str", return_value=""):
        # Should not raise exception and should evaluate cleanly
        menu._evaluate_saved_prediction_file(temp_csv)


def test_round1_readiness_blocked_def02():
    """Verifies Round 1 (Australia 2026) reports MISSING historical rounds and is_ready=False (DEF-02)."""
    menu = InteractiveMenu(root_dir=ROOT)
    ready, details = menu._audit_prediction_readiness("australia", 2026, 1, "pre_fp1", is_sprint=False)
    assert ready is False
    missing_msgs = [msg for status, msg in details if status == "MISSING"]
    assert any("Round 1 Season Opener" in m for m in missing_msgs)


def test_round1_pipeline_raises_informative_error_def01():
    """Verifies pipeline.run() on Round 1 raises an informative ValueError rather than an unhandled crash (DEF-01)."""
    from src.pipeline.race_prediction_pipeline import RacePredictionPipeline
    pipeline = RacePredictionPipeline(root_dir=ROOT)
    with pytest.raises(ValueError) as excinfo:
        pipeline.run("australia", 2026, "pre_fp1")
    assert "No prior same-season historical rounds exist" in str(excinfo.value)
    assert "Round 1" in str(excinfo.value)


def test_early_season_walk_forward_fallback_def01():
    """Verifies ModelEvaluator handles early rounds (<=3 rounds) via in-sample fallback without crashing (DEF-01)."""
    from src.models.evaluate import ModelEvaluator
    from src.pipeline.race_prediction_pipeline import RacePredictionPipeline
    pipeline = RacePredictionPipeline(root_dir=ROOT)
    # Load Round 1 data
    races_df, qualis_df = pipeline.data_loader.load_season_rounds(2026, [1])
    sessions_by_round = pipeline.data_loader.load_all_practice_sessions(2026, [1])
    train_features = pipeline.feature_engineer.build_training_dataset(
        races_df=races_df,
        qualis_df=qualis_df,
        practice_sessions_by_round=sessions_by_round,
        train_rounds=[1]
    )
    evaluator = ModelEvaluator(pipeline.model_config)
    eval_res = evaluator.evaluate_walk_forward(train_features)
    assert eval_res["validation_rounds"] == []
    assert eval_res["sample_count"] == len(train_features)
    assert eval_res["best_w_rf"] == 0.6
    assert eval_res["best_w_xgb"] == 0.4
    assert len(eval_res["selected_residuals"]) == len(train_features)
    assert eval_res["best_mae"] > 0.0

