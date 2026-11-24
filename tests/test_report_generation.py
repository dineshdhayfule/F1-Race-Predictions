import numpy as np
import pandas as pd

from src.pipeline.race_prediction_pipeline import (
    RacePredictionPipeline,
    _format_optional_metric,
)


def _simulation_rows():
    return pd.DataFrame([
        {
            "PredictedPosition": 1,
            "Abbreviation": "VER",
            "TeamName": "Team A",
            "ActualStartingGrid": 1,
            "is_pit_lane_start": 0,
            "ExpectedFinish": 1.2,
            "MedianFinish": 1,
            "WinProbability": 55.0,
            "Top3Probability": 80.0,
            "Top5Probability": 90.0,
            "PointsProbability": 95.0,
            "P5": 1,
            "P95": 3,
        },
        {
            "PredictedPosition": 2,
            "Abbreviation": "HAM",
            "TeamName": "Team B",
            "ActualStartingGrid": 2,
            "is_pit_lane_start": 0,
            "ExpectedFinish": 2.1,
            "MedianFinish": 2,
            "WinProbability": 25.0,
            "Top3Probability": 70.0,
            "Top5Probability": 85.0,
            "PointsProbability": 90.0,
            "P5": 1,
            "P95": 4,
        },
    ])


def test_optional_metric_formatter_handles_numeric_missing_and_nonfinite_values():
    assert _format_optional_metric(1.23456) == "1.235"
    assert _format_optional_metric(None) == "N/A"
    assert _format_optional_metric(np.nan) == "N/A"
    assert _format_optional_metric(np.inf) == "N/A"
    assert _format_optional_metric("-inf") == "N/A"


def test_markdown_report_preserves_structure_with_unavailable_metrics(tmp_path):
    pipeline = RacePredictionPipeline()
    sim_results = _simulation_rows()
    original_order = sim_results[["Abbreviation", "PredictedPosition"]].copy()
    comp_df = pd.DataFrame([
        {"model_name": "Manual RF", "mae": None, "medae": np.nan, "rmse": np.inf, "spearman": None},
    ])
    report_path = tmp_path / "prediction.md"

    pipeline._write_markdown_report(
        report_path=report_path,
        track_name="Qatar",
        year=2026,
        stage="pre_fp1",
        target_round=22,
        penalties={},
        sim_results=sim_results,
        comp_df=comp_df,
        sim_meta={
            "n_simulations": 25000,
            "selected_model": "Random Forest (100%)",
            "selection_method": "User Selected (Random Forest)",
            "is_out_of_sample": False,
            "selected_mae": None,
        },
    )

    report = report_path.read_text(encoding="utf-8")
    assert "| Manual RF | N/A | N/A | N/A | N/A |" in report
    assert "**Predicted Winner**: **VER**" in report
    pd.testing.assert_frame_equal(
        sim_results[["Abbreviation", "PredictedPosition"]],
        original_order,
    )
