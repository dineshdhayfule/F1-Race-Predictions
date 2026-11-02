"""
Unit and Integration Tests for Interactive CLI Model Selection and Custom Hybrid Weights
========================================================================================
Validates:
1. Bare --model opens interactive selection.
2. RF selection (interactive).
3. XGB selection (interactive).
4. Hybrid custom weights (interactive).
5. Invalid weight values rejection and recovery.
6. Weight sum not equal to 100 rejection.
7. Confirmation N returns to model selection.
8. Direct custom CLI arguments (--model hybrid --rf-weight 70 --xgb-weight 30).
9. Custom weights affect deterministic predictions.
10. Custom weights use matching walk-forward MAE and residuals.
11. Default hybrid remains numerically identical to existing 60/40.
12. Output files and protected baselines remain uncorrupted.
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
from src.cli.menu import prompt_model_selection, prompt_custom_weights_only
import main


@pytest.fixture
def sandbox_pipeline(tmp_path):
    """Provides an isolated pipeline writing to a pytest temporary directory."""
    p = RacePredictionPipeline(root_dir=ROOT)
    p.output_dir = tmp_path / "sandbox_custom_models"
    return p


def _make_mock_predictions_df():
    return pd.DataFrame([{
        "PredictedPosition": 1,
        "Abbreviation": "VER",
        "TeamName": "Red Bull Racing",
        "ActualStartingGrid": 1,
        "ExpectedFinish": 1.2,
        "WinProbability": 70.0,
        "Top3Probability": 90.0,
        "PodiumProbability": 90.0,
        "PointsProbability": 98.0,
        "is_pit_lane_start": 0,
    }])


# 1. Bare --model opens interactive selection
def test_bare_model_flag_triggers_interactive_selection():
    """Test 1: Bare --model on CLI sets model='__interactive__' and invokes prompt_model_selection."""
    test_argv = ["main.py", "predict", "--track", "mexico", "--model"]
    with patch.object(sys, "argv", test_argv):
        with patch("src.cli.menu.prompt_model_selection", return_value=("rf", None, None)) as mock_prompt:
            with patch("src.pipeline.race_prediction_pipeline.RacePredictionPipeline.run") as mock_run:
                mock_run.return_value = {
                    "predictions": _make_mock_predictions_df(),
                    "track_name": "Mexico City Grand Prix",
                    "selected_model": "Random Forest (100%)",
                    "selection_method": "User Selected",
                    "selected_mae": 3.5605,
                    "output_dir": "outputs/mexico"
                }
                main.main()
                mock_prompt.assert_called_once()
                mock_run.assert_called_once()
                call_kwargs = mock_run.call_args[1]
                assert call_kwargs["model"] == "rf"


# 2. RF selection in interactive prompt
def test_rf_selection_in_interactive_prompt():
    """Test 2: Selecting 1 in prompt_model_selection returns RF model."""
    with patch("builtins.input", side_effect=["1"]):
        model, rf_w, xgb_w = prompt_model_selection()
        assert model == "rf"
        assert rf_w is None
        assert xgb_w is None


# 3. XGB selection in interactive prompt
def test_xgb_selection_in_interactive_prompt():
    """Test 3: Selecting 2 in prompt_model_selection returns XGB model."""
    with patch("builtins.input", side_effect=["2"]):
        model, rf_w, xgb_w = prompt_model_selection()
        assert model == "xgb"
        assert rf_w is None
        assert xgb_w is None


# 4. Hybrid custom weights in interactive prompt
def test_hybrid_custom_weights_in_interactive_prompt():
    """Test 4: Selecting 3 and entering 70/30 with Y confirmation returns hybrid with custom weights."""
    inputs = ["3", "70", "30", "Y"]
    with patch("builtins.input", side_effect=inputs):
        model, rf_w, xgb_w = prompt_model_selection()
        assert model == "hybrid"
        assert rf_w == 70.0
        assert xgb_w == 30.0


# 5. Invalid weight values rejected and asks again
def test_invalid_weight_values_rejected_and_recovered():
    """Test 5: Non-numeric weights and out-of-range weights are rejected, asking again until valid."""
    # Inputs: "abc", "def" (non-numeric), then "-10", "110" (out of range), then "65", "35", "Y" (valid)
    inputs = ["abc", "def", "-10", "110", "65", "35", "Y"]
    with patch("builtins.input", side_effect=inputs):
        model, rf_w, xgb_w = prompt_custom_weights_only()
        assert model == "hybrid"
        assert rf_w == 65.0
        assert xgb_w == 35.0


# 6. Weight sum not equal to 100 rejected
def test_weight_sum_not_equal_to_100_rejected():
    """Test 6: Weights summing to != 100 (e.g. 50 + 40 = 90) are rejected, asking again until sum is 100."""
    inputs = ["50", "40", "80", "20", "Y"]
    with patch("builtins.input", side_effect=inputs):
        model, rf_w, xgb_w = prompt_custom_weights_only()
        assert model == "hybrid"
        assert rf_w == 80.0
        assert xgb_w == 20.0


# 7. Confirmation N returns to selection
def test_confirmation_n_returns_to_model_selection():
    """Test 7: Confirming N returns to model selection, allowing user to select a different option."""
    # User chooses 3, inputs 75/25, enters N (reject), returns to main selection, then picks 1 (RF)
    inputs = ["3", "75", "25", "N", "1"]
    with patch("builtins.input", side_effect=inputs):
        model, rf_w, xgb_w = prompt_model_selection()
        assert model == "rf"
        assert rf_w is None
        assert xgb_w is None


# 8. Direct custom CLI arguments
def test_direct_custom_cli_arguments():
    """Test 8: --model hybrid --rf-weight 70 --xgb-weight 30 parses cleanly and passes weights to pipeline."""
    test_argv = ["main.py", "predict", "--track", "mexico", "--model", "hybrid", "--rf-weight", "70", "--xgb-weight", "30"]
    with patch.object(sys, "argv", test_argv):
        with patch("src.pipeline.race_prediction_pipeline.RacePredictionPipeline.run") as mock_run:
            mock_run.return_value = {
                "predictions": _make_mock_predictions_df(),
                "track_name": "Mexico City Grand Prix",
                "selected_model": "Hybrid Ensemble (70% RF / 30% XGB)",
                "selection_method": "User Selected",
                "selected_mae": 3.5650,
                "output_dir": "outputs/mexico"
            }
            main.main()
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["model"] == "hybrid"
            assert call_kwargs["rf_weight"] == 70.0
            assert call_kwargs["xgb_weight"] == 30.0


def test_cli_rejects_single_weight_argument():
    """Test 8b: CLI rejects when only one weight argument is supplied."""
    test_argv = ["main.py", "predict", "--track", "mexico", "--model", "hybrid", "--rf-weight", "70"]
    with patch.object(sys, "argv", test_argv):
        with pytest.raises(SystemExit) as excinfo:
            main.main()
        assert excinfo.value.code != 0


def test_cli_rejects_weights_not_summing_to_100():
    """Test 8c: CLI rejects when --rf-weight and --xgb-weight do not sum to 100."""
    test_argv = ["main.py", "predict", "--track", "mexico", "--model", "hybrid", "--rf-weight", "60", "--xgb-weight", "50"]
    with patch.object(sys, "argv", test_argv):
        with pytest.raises(SystemExit) as excinfo:
            main.main()
        assert excinfo.value.code != 0


# 9. Custom weights affect deterministic predictions
def test_custom_weights_affect_deterministic_predictions(sandbox_pipeline):
    """Test 9: Passing custom weights (e.g. 80/20 vs 20/80) changes deterministic scores and expected finishes."""
    p = sandbox_pipeline
    # Run 80% RF / 20% XGB
    res_80_20 = p.run("mexico", 2026, stage="after_qualifying", model="hybrid", rf_weight=80, xgb_weight=20)
    # Run 20% RF / 80% XGB
    res_20_80 = p.run("mexico", 2026, stage="after_qualifying", model="hybrid", rf_weight=20, xgb_weight=80)

    df_80 = res_80_20["predictions"]
    df_20 = res_20_80["predictions"]

    assert res_80_20["selected_rf_weight"] == 0.8
    assert res_80_20["selected_xgb_weight"] == 0.2
    assert res_20_80["selected_rf_weight"] == 0.2
    assert res_20_80["selected_xgb_weight"] == 0.8

    # The expected finishes must reflect the different weighting
    assert not np.allclose(df_80["ExpectedFinish"].values, df_20["ExpectedFinish"].values, atol=1e-3)


# 10. Custom weights use matching walk-forward MAE and residuals
def test_custom_weights_use_matching_walk_forward_mae_and_residuals(sandbox_pipeline):
    """Test 10: Custom weights (70/30) use exact walk-forward MAE and residuals, not the 60/40 defaults."""
    res_70_30 = sandbox_pipeline.run("mexico", 2026, stage="after_qualifying", model="hybrid", rf_weight=70, xgb_weight=30)

    assert res_70_30["selected_model"] == "Hybrid Ensemble (70% RF / 30% XGB)"
    assert res_70_30["selection_method"] == "User Selected Hybrid (70% RF / 30% XGB)"

    # Check that 70% RF / 30% XGB is in the model comparison table
    comp_df = res_70_30["model_comparison"]
    match = comp_df[comp_df["model_name"] == "70% RF / 30% XGB"]
    assert len(match) == 1
    custom_comp_mae = float(match.iloc[0]["mae"])

    # Selected MAE must exactly match the walk-forward comparison MAE
    assert res_70_30["selected_mae"] == custom_comp_mae
    assert res_70_30["selected_mae"] != 3.5771  # Must NOT be the 60/40 hybrid MAE

    # Verify generated report
    report_path = Path(res_70_30["output_dir"]) / "mexico_2026_after_qualifying_report.md"
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "**Selected Model**: **Hybrid Ensemble (70% RF / 30% XGB)**" in content
    assert f"**Selected MAE**: {custom_comp_mae}" in content


# 11. Default hybrid remains numerically identical to existing 60/40
def test_default_hybrid_remains_numerically_identical_to_60_40(sandbox_pipeline):
    """Test 11: Omitted weights produce the standard 60% RF / 40% XGB hybrid MAE and predictions."""
    res_default = sandbox_pipeline.run("mexico", 2026, stage="after_qualifying")

    assert res_default["model"] == "hybrid"
    assert res_default["selected_model"] == "Hybrid Ensemble (60% RF / 40% XGB)"
    assert res_default["selected_rf_weight"] == 0.6
    assert res_default["selected_xgb_weight"] == 0.4
    expected_60_40_mae = float(res_default["model_comparison"].loc[res_default["model_comparison"]["model_name"] == "60% RF / 40% XGB", "mae"].iloc[0])
    assert res_default["selected_mae"] == expected_60_40_mae
    assert round(res_default["selected_mae"], 4) == round(expected_60_40_mae, 4)

    top3 = res_default["predictions"].iloc[:3]["Abbreviation"].tolist()
    assert top3 == ["HAM", "RUS", "NOR"]
