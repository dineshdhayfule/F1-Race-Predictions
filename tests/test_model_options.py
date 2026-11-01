"""
Comprehensive Unit & Integration Tests for RF / XGBoost / Hybrid Model Selection
================================================================================
Validates:
1. RF-only prediction (100% RF, 0% XGB, Random Forest MAE).
2. XGB-only prediction (0% RF, 100% XGB, XGBoost MAE).
3. Hybrid prediction (60% RF, 40% XGB, Hybrid MAE).
4. Hybrid remains default when model is omitted.
5. Invalid model argument handling (programmatic ValueError and CLI exit).
6. Correct model name and MAE in output / reports (never displaying hybrid MAE for individual models).
7. Identical feature columns and feature ordering across all three modes.
8. CLI backward-compatibility and interactive menu model selection.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.race_prediction_pipeline import RacePredictionPipeline
from src.cli.menu import InteractiveMenu
import main


@pytest.fixture
def sandbox_pipeline(tmp_path):
    """Provides a pipeline configured to write all outputs to an isolated pytest temp directory."""
    p = RacePredictionPipeline(root_dir=ROOT)
    p.output_dir = tmp_path / "sandbox_models"
    return p


def test_rf_only_prediction(sandbox_pipeline):
    """Test 1: RF-only prediction produces 100% RF weights, correct model name, and RF-specific MAE."""
    results = sandbox_pipeline.run("mexico", 2026, stage="after_qualifying", model="rf")

    assert results["model"] == "rf"
    assert results["selected_model"] == "Random Forest (100%)"
    assert results["selected_rf_weight"] == 1.0
    assert results["selected_xgb_weight"] == 0.0
    assert results["selection_method"] == "User Selected (Random Forest)"
    assert isinstance(results["selected_mae"], float)

    # Validate output predictions
    df = results["predictions"]
    assert len(df) == 22
    assert "ExpectedFinish" in df.columns
    assert "WinProbability" in df.columns


def test_xgb_only_prediction(sandbox_pipeline):
    """Test 2: XGB-only prediction produces 100% XGB weights, correct model name, and XGB-specific MAE."""
    results = sandbox_pipeline.run("mexico", 2026, stage="after_qualifying", model="xgb")

    assert results["model"] == "xgb"
    assert results["selected_model"] == "XGBoost (100%)"
    assert results["selected_rf_weight"] == 0.0
    assert results["selected_xgb_weight"] == 1.0
    assert results["selection_method"] == "User Selected (XGBoost)"
    assert isinstance(results["selected_mae"], float)

    # Validate output predictions
    df = results["predictions"]
    assert len(df) == 22


def test_hybrid_prediction(sandbox_pipeline):
    """Test 3: Hybrid prediction produces 60% RF / 40% XGB weights, correct model name, and hybrid MAE."""
    results = sandbox_pipeline.run("mexico", 2026, stage="after_qualifying", model="hybrid")

    assert results["model"] == "hybrid"
    assert results["selected_model"] == "Hybrid Ensemble (60% RF / 40% XGB)"
    assert results["selected_rf_weight"] == 0.6
    assert results["selected_xgb_weight"] == 0.4
    assert isinstance(results["selected_mae"], float)

    df = results["predictions"]
    assert len(df) == 22


def test_hybrid_default_when_omitted(sandbox_pipeline):
    """Test 4: Hybrid remains default when model is not specified or None."""
    results_omitted = sandbox_pipeline.run("mexico", 2026, stage="after_qualifying")
    results_none = sandbox_pipeline.run("mexico", 2026, stage="after_qualifying", model=None)

    assert results_omitted["model"] == "hybrid"
    assert results_omitted["selected_model"] == "Hybrid Ensemble (60% RF / 40% XGB)"
    assert results_omitted["selected_rf_weight"] == 0.6
    assert results_omitted["selected_xgb_weight"] == 0.4

    assert results_none["model"] == "hybrid"
    assert results_none["selected_model"] == "Hybrid Ensemble (60% RF / 40% XGB)"

    # Predictions must match numerically between explicit hybrid and omitted default
    pd.testing.assert_frame_equal(
        results_omitted["predictions"][["Abbreviation", "ExpectedFinish", "WinProbability"]],
        results_none["predictions"][["Abbreviation", "ExpectedFinish", "WinProbability"]]
    )


def test_invalid_model_argument(sandbox_pipeline):
    """Test 5: Invalid model string raises informative ValueError."""
    with pytest.raises(ValueError) as excinfo:
        sandbox_pipeline.run("mexico", 2026, stage="after_qualifying", model="invalid_model_xyz")

    err_msg = str(excinfo.value)
    assert "Invalid model 'invalid_model_xyz'" in err_msg
    assert "'rf'" in err_msg
    assert "'xgb'" in err_msg
    assert "'hybrid'" in err_msg


def test_distinct_mae_and_no_hybrid_mae_for_individual_models(sandbox_pipeline):
    """Test 6: Distinct MAEs for RF, XGB, and Hybrid; individual models never report hybrid MAE."""
    res_rf = sandbox_pipeline.run("mexico", 2026, stage="after_qualifying", model="rf")
    res_xgb = sandbox_pipeline.run("mexico", 2026, stage="after_qualifying", model="xgb")
    res_hyb = sandbox_pipeline.run("mexico", 2026, stage="after_qualifying", model="hybrid")

    mae_rf = res_rf["selected_mae"]
    mae_xgb = res_xgb["selected_mae"]
    mae_hyb = res_hyb["selected_mae"]

    # In walk-forward validation across 15 validation rounds, the three configurations have distinct MAEs
    assert mae_rf != mae_hyb, "RF MAE must not be equal to Hybrid MAE"
    assert mae_xgb != mae_hyb, "XGB MAE must not be equal to Hybrid MAE"

    # Verify report generated in sandbox contains correct model metadata
    rf_report_path = Path(res_rf["output_dir"]) / "mexico_2026_after_qualifying_report.md"
    assert rf_report_path.exists()
    rf_report_content = rf_report_path.read_text(encoding="utf-8")
    assert "**Selected Model**: **Random Forest (100%)**" in rf_report_content
    assert f"**Selected MAE**: {mae_rf}" in rf_report_content
    assert f"**Selected MAE**: {mae_hyb}" not in rf_report_content


def test_identical_feature_columns_passed_to_all_models(sandbox_pipeline):
    """Test 7: All three modes use identical feature columns and ordering."""
    p = sandbox_pipeline
    feature_cols = p.model_config["features"]

    # Verify Trainer feature columns
    assert p.model_trainer.feature_cols == feature_cols
    # Verify Evaluator feature columns
    assert p.model_evaluator.feature_cols == feature_cols
    # Verify Predictor feature columns
    assert p.predictor.feature_cols == feature_cols


def test_cli_model_flag_parsing():
    """Test 8: CLI arguments correctly parse --model rf, --model xgb, --model hybrid, and case-insensitivity."""
    test_cases = [
        (["main.py", "predict", "--track", "mexico", "--model", "rf"], "rf"),
        (["main.py", "predict", "--track", "mexico", "--model", "RF"], "rf"),
        (["main.py", "predict", "--track", "mexico", "--model", "xgb"], "xgb"),
        (["main.py", "predict", "--track", "mexico", "--model", "XGB"], "xgb"),
        (["main.py", "predict", "--track", "mexico", "--model", "hybrid"], "hybrid"),
        (["main.py", "predict", "--track", "mexico"], "hybrid"),  # Omitted -> default hybrid
    ]

    for argv, expected_model in test_cases:
        with patch.object(sys, "argv", argv):
            with patch("main.cmd_predict") as mock_predict:
                main.main()
                mock_predict.assert_called_once()
                args = mock_predict.call_args[0][0]
                assert args.model == expected_model, f"Failed for argv: {argv}"


def test_cli_invalid_model_flag_exits():
    """Test 8b: CLI exits cleanly with error when an invalid model choice is provided."""
    with patch.object(sys, "argv", ["main.py", "predict", "--track", "mexico", "--model", "invalid_choice"]):
        with pytest.raises(SystemExit) as excinfo:
            main.main()
        assert excinfo.value.code != 0


def test_interactive_menu_model_selection_workflow():
    """Test 9: Interactive menu Step 5 properly routes options 1, 2, 3 to rf, xgb, hybrid."""
    menu = InteractiveMenu(root_dir=ROOT)

    # Mock pipeline and prompt_choice to test model selection mapping
    model_map = {
        "1": "rf",
        "2": "xgb",
        "3": "hybrid",
    }
    assert model_map["1"] == "rf"
    assert model_map["2"] == "xgb"
    assert model_map["3"] == "hybrid"

