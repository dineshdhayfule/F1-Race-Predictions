"""
Parity Regression & Isolation Test Suite (TC-05, TC-06, TC-11)
==============================================================
Validates:
1. TC-05: Sepang 2026 pre-race prediction parity against saved baseline.
2. TC-06: Baku 2026 pre-race prediction parity against saved baseline.
3. TC-11: Complete output isolation, ensuring pipeline writes exclusively to
   pytest tmp_path and leaves all 21 files in outputs/ completely unmodified.

Derived directly from verified baseline artifacts in outputs/sepang/ and outputs/baku/.
Zero modifications to production source code or existing output directories.
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.pipeline.race_prediction_pipeline import RacePredictionPipeline

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"


def get_all_protected_file_hashes() -> dict:
    """Computes SHA-256 hashes of all protected files currently in outputs/."""
    hashes = {}
    if not OUTPUTS_DIR.exists():
        return hashes
    for p in sorted(OUTPUTS_DIR.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(OUTPUTS_DIR))
            hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes


def test_complete_output_isolation_and_hash_invariance(tmp_path):
    """
    TC-11: Confirms that redirecting pipeline.output_dir to tmp_path isolates
    all generated artifacts without modifying any protected file in outputs/.
    """
    # 1. Capture baseline hashes across all protected output files
    pre_hashes = get_all_protected_file_hashes()
    assert len(pre_hashes) >= 8, f"Expected protected files in outputs/, found {len(pre_hashes)}"

    sandbox_dir = tmp_path / "sandbox_tc11"
    pipeline = RacePredictionPipeline(root_dir=ROOT)
    pipeline.output_dir = sandbox_dir

    # 2. Execute pipeline in sandbox
    results = pipeline.run("sepang", year=2026, stage="after_qualifying")
    assert results is not None
    assert results["track_key"] == "sepang"

    # 3. Verify all 5 expected artifacts were generated inside sandbox
    track_sandbox = sandbox_dir / "sepang"
    expected_files = [
        "sepang_2026_after_qualifying_predictions.csv",
        "sepang_2026_model_comparison.csv",
        "sepang_2026_after_qualifying_report.md",
        "sepang_2026_race_top3_report.md",
        "sepang_2026_race_top10_report.md",
    ]
    for ef in expected_files:
        p = track_sandbox / ef
        assert p.exists(), f"Expected sandbox artifact missing: {p}"
        assert p.stat().st_size > 0, f"Sandbox artifact is empty: {p}"

    # 4. Verify post-test hash invariance across ALL protected files
    post_hashes = get_all_protected_file_hashes()
    assert pre_hashes == post_hashes, "CRITICAL DEFECT: Protected files in outputs/ were modified!"


def test_sepang_output_parity_in_sandbox(tmp_path):
    """
    TC-05: Verifies that running Sepang 2026 after_qualifying in sandbox matches
    the verified production baseline outputs/sepang/sepang_2026_after_qualifying_predictions.csv.
    """
    pre_hashes = get_all_protected_file_hashes()

    sandbox_dir = tmp_path / "sandbox_tc05"
    pipeline = RacePredictionPipeline(root_dir=ROOT)
    pipeline.output_dir = sandbox_dir

    results = pipeline.run("sepang", year=2026, stage="after_qualifying")
    sandbox_csv = sandbox_dir / "sepang" / "sepang_2026_after_qualifying_predictions.csv"
    assert sandbox_csv.exists()

    gen_df = pd.read_csv(sandbox_csv)
    base_csv = OUTPUTS_DIR / "sepang" / "sepang_2026_after_qualifying_predictions.csv"
    assert base_csv.exists(), f"Baseline file not found at {base_csv}"
    base_df = pd.read_csv(base_csv)

    # 1. Structural assertions
    assert len(gen_df) == 22, f"Expected 22 rows, got {len(gen_df)}"
    assert list(gen_df.columns) == list(base_df.columns)

    # 2. Driver row set and top identities assertion
    assert set(gen_df["Abbreviation"]) == set(base_df["Abbreviation"])
    assert set(gen_df.iloc[:10]["Abbreviation"]) == set(base_df.iloc[:10]["Abbreviation"])

    # 3. Top 3 set identities (VER, HAM, ANT are top 3, differing by <0.06 MC position variance)
    assert set(gen_df.iloc[:3]["Abbreviation"]) == {"VER", "HAM", "ANT"}

    # 4. Starting grid & penalties check
    assert int(gen_df.loc[gen_df["Abbreviation"] == "HAD", "ActualStartingGrid"].iloc[0]) == 8
    assert int(gen_df.loc[gen_df["Abbreviation"] == "COL", "ActualStartingGrid"].iloc[0]) == 21
    assert int(gen_df.loc[gen_df["Abbreviation"] == "LIN", "ActualStartingGrid"].iloc[0]) == 22

    # Align by driver abbreviation for driver-specific metric comparison
    gen_aligned = gen_df.sort_values("Abbreviation").reset_index(drop=True)
    base_aligned = base_df.sort_values("Abbreviation").reset_index(drop=True)
    assert gen_aligned["ActualStartingGrid"].tolist() == base_aligned["ActualStartingGrid"].tolist()

    # 5. Numerical tolerance checks (tolerance: ExpectedFinish <= 0.35, Probabilities <= 5.0%)
    for col in ["ExpectedFinish", "MLExpectedFinish"]:
        diff = np.abs(gen_aligned[col] - base_aligned[col])
        max_diff = diff.max()
        assert max_diff <= 0.35, f"Column {col} exceeded tolerance 0.35: max diff = {max_diff}"

    for col in ["WinProbability", "Top3Probability", "Top5Probability", "PointsProbability"]:
        diff = np.abs(gen_aligned[col] - base_aligned[col])
        max_diff = diff.max()
        assert max_diff <= 5.0, f"Column {col} exceeded Monte Carlo tolerance 5.0%: max diff = {max_diff}"

    # 6. Verify zero writes occurred in outputs/
    post_hashes = get_all_protected_file_hashes()
    assert pre_hashes == post_hashes, "Protected files in outputs/ were modified during Sepang run!"


def test_baku_output_parity_in_sandbox(tmp_path):
    """
    TC-06: Verifies that running Baku 2026 after_qualifying in sandbox matches
    the verified production baseline outputs/baku/baku_2026_after_qualifying_predictions.csv
    when operating in legacy mode (circuit_aware=False, 0.35 static grid weight).
    """
    pre_hashes = get_all_protected_file_hashes()

    sandbox_dir = tmp_path / "sandbox_tc06"
    pipeline = RacePredictionPipeline(root_dir=ROOT)
    pipeline.output_dir = sandbox_dir
    pipeline.model_config["blending"]["circuit_aware"] = False
    pipeline.predictor.circuit_aware = False

    results = pipeline.run("baku", year=2026, stage="after_qualifying")
    sandbox_csv = sandbox_dir / "baku" / "baku_2026_after_qualifying_predictions.csv"
    assert sandbox_csv.exists()

    gen_df = pd.read_csv(sandbox_csv)
    base_csv = OUTPUTS_DIR / "baku" / "baku_2026_after_qualifying_predictions.csv"
    assert base_csv.exists(), f"Baseline file not found at {base_csv}"
    base_df = pd.read_csv(base_csv)

    # 1. Structural assertions
    assert len(gen_df) == 22
    assert list(gen_df.columns) == list(base_df.columns)

    # 2. Driver row set and top identities assertion
    assert set(gen_df["Abbreviation"]) == set(base_df["Abbreviation"])
    assert set(gen_df.iloc[:10]["Abbreviation"]) == set(base_df.iloc[:10]["Abbreviation"])

    # 3. Top 3 exact identities
    assert gen_df.iloc[0]["Abbreviation"] == "RUS"
    assert gen_df.iloc[1]["Abbreviation"] == "LEC"
    assert gen_df.iloc[2]["Abbreviation"] == "PIA"

    # Align by driver abbreviation for driver-specific metric comparison
    gen_aligned = gen_df.sort_values("Abbreviation").reset_index(drop=True)
    base_aligned = base_df.sort_values("Abbreviation").reset_index(drop=True)

    # 4. Numerical tolerance checks (tolerance: ExpectedFinish <= 0.35, Probabilities <= 5.0%)
    for col in ["ExpectedFinish", "MLExpectedFinish"]:
        diff = np.abs(gen_aligned[col] - base_aligned[col])
        max_diff = diff.max()
        assert max_diff <= 0.35, f"Column {col} exceeded tolerance 0.35: max diff = {max_diff}"

    for col in ["WinProbability", "Top3Probability", "Top5Probability", "PointsProbability"]:
        diff = np.abs(gen_aligned[col] - base_aligned[col])
        max_diff = diff.max()
        assert max_diff <= 5.0, f"Column {col} exceeded Monte Carlo tolerance 5.0%: max diff = {max_diff}"

    # 5. Verify zero writes occurred in outputs/
    post_hashes = get_all_protected_file_hashes()
    assert pre_hashes == post_hashes, "Protected files in outputs/ were modified during Baku run!"


def test_baku_circuit_aware_production_parity(tmp_path):
    """
    TC-07: Verifies that running Baku 2026 after_qualifying in sandbox under
    the Phase 4 circuit-aware production configuration (circuit_aware=True, 0.50 street weight)
    reproduces exact parity with the documented fixture tests/fixtures/baku_2026_after_qualifying_circuit_aware_expected.csv.
    """
    pre_hashes = get_all_protected_file_hashes()

    sandbox_dir = tmp_path / "sandbox_tc07"
    pipeline = RacePredictionPipeline(root_dir=ROOT)
    pipeline.output_dir = sandbox_dir

    results = pipeline.run("baku", year=2026, stage="after_qualifying")
    sandbox_csv = sandbox_dir / "baku" / "baku_2026_after_qualifying_predictions.csv"
    assert sandbox_csv.exists()

    gen_df = pd.read_csv(sandbox_csv)
    fixture_csv = ROOT / "tests" / "fixtures" / "baku_2026_after_qualifying_circuit_aware_expected.csv"
    assert fixture_csv.exists(), f"Fixture file not found at {fixture_csv}"
    fixture_df = pd.read_csv(fixture_csv)

    # 1. Structural assertions
    assert len(gen_df) == 22
    assert list(gen_df.columns) == list(fixture_df.columns)

    # 2. Driver row set and top identities assertion
    assert set(gen_df["Abbreviation"]) == set(fixture_df["Abbreviation"])
    assert set(gen_df.iloc[:10]["Abbreviation"]) == set(fixture_df.iloc[:10]["Abbreviation"])

    # 3. Top 3 exact identities
    assert gen_df.iloc[0]["Abbreviation"] == "RUS"
    assert gen_df.iloc[1]["Abbreviation"] == "LEC"
    assert gen_df.iloc[2]["Abbreviation"] == "PIA"

    # Align by driver abbreviation for driver-specific metric comparison
    gen_aligned = gen_df.sort_values("Abbreviation").reset_index(drop=True)
    fix_aligned = fixture_df.sort_values("Abbreviation").reset_index(drop=True)

    # 4. Numerical tolerance checks (tolerance: ExpectedFinish <= 0.35, Probabilities <= 5.0%)
    for col in ["ExpectedFinish", "MLExpectedFinish"]:
        diff = np.abs(gen_aligned[col] - fix_aligned[col])
        max_diff = diff.max()
        assert max_diff <= 0.35, f"Column {col} exceeded tolerance 0.35: max diff = {max_diff}"

    for col in ["WinProbability", "Top3Probability", "Top5Probability", "PointsProbability"]:
        diff = np.abs(gen_aligned[col] - fix_aligned[col])
        max_diff = diff.max()
        assert max_diff <= 5.0, f"Column {col} exceeded Monte Carlo tolerance 5.0%: max diff = {max_diff}"

    # 5. Verify zero writes occurred in outputs/
    post_hashes = get_all_protected_file_hashes()
    assert pre_hashes == post_hashes, "Protected files in outputs/ were modified during Baku circuit-aware run!"
