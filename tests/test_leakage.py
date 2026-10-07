"""
Unit Tests for Zero Leakage Enforcement
=======================================
Verifies that target race data cannot contaminate training, validation, or target features.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.data_validator import DataValidator


def test_audit_leakage_clean():
    validator = DataValidator()

    train_df = pd.DataFrame({
        "Round": [1, 1, 2, 2],
        "Abbreviation": ["VER", "HAM", "VER", "HAM"],
        "ActualFinish": [1, 2, 2, 1],
        "races_completed": [0, 0, 1, 1],
        "GridPosition": [1, 2, 2, 1]
    })

    target_df = pd.DataFrame({
        "Round": [16, 16],
        "Abbreviation": ["VER", "HAM"],
        "GridPosition": [1, 2],
        "QualifyingPosition": [1, 2]
    })

    report = validator.audit_leakage(train_df, target_df, target_round=16)
    assert report["is_clean"] is True
    assert len(report["violations"]) == 0


def test_audit_leakage_target_has_actual_finish():
    validator = DataValidator()

    train_df = pd.DataFrame({
        "Round": [1, 2],
        "Abbreviation": ["VER", "HAM"],
        "ActualFinish": [1, 2],
        "races_completed": [0, 1],
        "GridPosition": [1, 2]
    })

    # Contaminated target containing ActualFinish
    target_df = pd.DataFrame({
        "Round": [16, 16],
        "Abbreviation": ["VER", "HAM"],
        "ActualFinish": [1, 2],
        "GridPosition": [1, 2]
    })

    report = validator.audit_leakage(train_df, target_df, target_round=16)
    assert report["is_clean"] is False
    assert any("ActualFinish" in v for v in report["violations"])


def test_audit_leakage_target_round_in_training():
    validator = DataValidator()

    # Contaminated training containing Round 16
    train_df = pd.DataFrame({
        "Round": [1, 16],
        "Abbreviation": ["VER", "HAM"],
        "ActualFinish": [1, 2],
        "races_completed": [0, 15],
        "GridPosition": [1, 2]
    })

    target_df = pd.DataFrame({
        "Round": [16, 16],
        "Abbreviation": ["VER", "HAM"],
        "GridPosition": [1, 2]
    })

    report = validator.audit_leakage(train_df, target_df, target_round=16)
    assert report["is_clean"] is False
    assert any("target round 16" in v.lower() for v in report["violations"])


def test_audit_leakage_races_completed_exceeds_cutoff():
    validator = DataValidator()

    # races_completed is 5 for Round 3 (impossible without looking forward)
    train_df = pd.DataFrame({
        "Round": [3, 3],
        "Abbreviation": ["VER", "HAM"],
        "ActualFinish": [1, 2],
        "races_completed": [5, 5],
        "GridPosition": [1, 2]
    })

    target_df = pd.DataFrame({
        "Round": [16, 16],
        "Abbreviation": ["VER", "HAM"],
        "GridPosition": [1, 2]
    })

    report = validator.audit_leakage(train_df, target_df, target_round=16)
    assert report["is_clean"] is False
    assert any("races_completed" in v for v in report["violations"])
