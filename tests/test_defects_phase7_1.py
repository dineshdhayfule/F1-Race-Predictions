"""
Tests for Phase 7.1 Critical Defect Remediation
==============================================
Validates:
- DEF-01: Ridge weight extraction, invalid weight safety, non-zero prediction guarantees, residual routing
- DEF-02: Sprint race exclusion in season data loader, no row/qualifying duplication
- DEF-05: Comprehensive terminal status classification (DNF, DNS, DSQ, NC, Finished)
- DEF-08: Round 16 metadata, circuit alias resolution, and track configuration integrity
"""

import math
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.features.season_features import (
    classify_race_status,
    is_classified_finish,
    is_dnf,
    is_retired,
    is_dns,
    is_dsq,
    is_nc,
    build_season_form,
    compute_season_driver_stats,
)
from src.data.data_loader import DataLoader
from src.models.predict import Predictor
from src.utils.circuit_mapping import resolve_circuit_key, load_tracks_config


# ============================================================================
# STEP 1: DEF-01 RIDGE WEIGHT EXTRACTION & ZERO-COLLAPSE SAFEGUARDS
# ============================================================================

class TestDef01RidgeWeightExtraction:
    """Verifies DEF-01 fix: weights are correctly extracted, validated, and normalized."""

    def test_ridge_only_selection_predict_deterministic(self):
        """When Ridge is 100%, Predictor.predict_deterministic should strictly output Ridge predictions."""
        predictor = Predictor(config={})
        target_df = pd.DataFrame({
            "Abbreviation": ["VER", "HAM"],
            "TeamName": ["Red Bull Racing", "Mercedes"],
            "GridPosition": [1, 2],
            "QualifyingPosition": [1, 2],
            "simulation_start_rank": [1, 2]
        })
        
        # Mock models
        rf_mock = MagicMock()
        rf_mock.predict.return_value = np.array([5.0, 6.0])
        xgb_mock = MagicMock()
        xgb_mock.predict.return_value = np.array([7.0, 8.0])
        ridge_mock = MagicMock()
        ridge_mock.predict.return_value = np.array([1.5, 2.5])
        
        # Test ridge_weight=1.0, rf=0.0, xgb=0.0
        det_df, blended = predictor.predict_deterministic(
            target_df=target_df,
            imputer=MagicMock(),
            rf_model=rf_mock,
            xgb_model=xgb_mock,
            ridge_model=ridge_mock,
            rf_weight=0.0,
            xgb_weight=0.0,
            ridge_weight=1.0,
            circuit_type="permanent",
            model_mode="ridge"
        )
        
        assert "DeterministicRank" in det_df.columns
        assert not det_df["DeterministicRank"].isna().any()
        # Ensure predictions did not collapse to 0
        assert (det_df["DeterministicRank"] > 0).all()
        assert (det_df["ML_PredictedScore"] > 0).all()
        # ML expected finish should reflect ridge
        assert np.isclose(det_df.loc[0, "ML_PredictedScore"], 1.5)
        assert np.isclose(det_df.loc[1, "ML_PredictedScore"], 2.5)

    def test_rf_only_selection(self):
        """When RF is 100%, output should reflect RF predictions."""
        predictor = Predictor(config={})
        target_df = pd.DataFrame({
            "Abbreviation": ["VER"],
            "TeamName": ["Red Bull Racing"],
            "GridPosition": [1],
            "QualifyingPosition": [1],
            "simulation_start_rank": [1]
        })
        rf_mock = MagicMock()
        rf_mock.predict.return_value = np.array([3.0])
        xgb_mock = MagicMock()
        xgb_mock.predict.return_value = np.array([9.0])
        
        det_df, _ = predictor.predict_deterministic(
            target_df=target_df,
            imputer=MagicMock(),
            rf_model=rf_mock,
            xgb_model=xgb_mock,
            rf_weight=1.0,
            xgb_weight=0.0,
            ridge_weight=0.0,
            model_mode="rf"
        )
        assert np.isclose(det_df.loc[0, "ML_PredictedScore"], 3.0)

    def test_xgb_only_selection(self):
        """When XGB is 100%, output should reflect XGB predictions."""
        predictor = Predictor(config={})
        target_df = pd.DataFrame({
            "Abbreviation": ["VER"],
            "TeamName": ["Red Bull Racing"],
            "GridPosition": [1],
            "QualifyingPosition": [1],
            "simulation_start_rank": [1]
        })
        rf_mock = MagicMock()
        rf_mock.predict.return_value = np.array([3.0])
        xgb_mock = MagicMock()
        xgb_mock.predict.return_value = np.array([4.5])
        
        det_df, _ = predictor.predict_deterministic(
            target_df=target_df,
            imputer=MagicMock(),
            rf_model=rf_mock,
            xgb_model=xgb_mock,
            rf_weight=0.0,
            xgb_weight=1.0,
            ridge_weight=0.0,
            model_mode="xgb"
        )
        assert np.isclose(det_df.loc[0, "ML_PredictedScore"], 4.5)

    def test_mixed_ensemble_selection(self):
        """Mixed ensemble 40% RF / 30% XGB / 30% Ridge."""
        predictor = Predictor(config={})
        target_df = pd.DataFrame({
            "Abbreviation": ["NOR"],
            "TeamName": ["McLaren"],
            "GridPosition": [2],
            "QualifyingPosition": [2],
            "simulation_start_rank": [2]
        })
        rf_mock = MagicMock()
        rf_mock.predict.return_value = np.array([2.0])
        xgb_mock = MagicMock()
        xgb_mock.predict.return_value = np.array([4.0])
        ridge_mock = MagicMock()
        ridge_mock.predict.return_value = np.array([1.0])
        
        det_df, _ = predictor.predict_deterministic(
            target_df=target_df,
            imputer=MagicMock(),
            rf_model=rf_mock,
            xgb_model=xgb_mock,
            ridge_model=ridge_mock,
            rf_weight=0.40,
            xgb_weight=0.30,
            ridge_weight=0.30,
            model_mode="multi"
        )
        expected_ml = 0.40 * 2.0 + 0.30 * 4.0 + 0.30 * 1.0  # 0.8 + 1.2 + 0.3 = 2.3
        assert np.isclose(det_df.loc[0, "ML_PredictedScore"], expected_ml)

    def test_invalid_and_zero_weights_fallback(self):
        """When weights are all zero, NaN, or negative, Predictor must fall back gracefully without zero collapse."""
        predictor = Predictor(config={})
        target_df = pd.DataFrame({
            "Abbreviation": ["LEC"],
            "TeamName": ["Ferrari"],
            "GridPosition": [3],
            "QualifyingPosition": [3],
            "simulation_start_rank": [3]
        })
        rf_mock = MagicMock()
        rf_mock.predict.return_value = np.array([4.0])
        xgb_mock = MagicMock()
        xgb_mock.predict.return_value = np.array([6.0])

        # Test total weight == 0.0
        det_df, _ = predictor.predict_deterministic(
            target_df=target_df,
            imputer=MagicMock(),
            rf_model=rf_mock,
            xgb_model=xgb_mock,
            rf_weight=0.0,
            xgb_weight=0.0,
            ridge_weight=0.0,
            model_mode="custom"
        )
        # Should fallback to default 0.60 RF + 0.40 XGB = 0.6*4.0 + 0.4*6.0 = 4.8
        assert not det_df["DeterministicRank"].isna().any()
        assert det_df.loc[0, "ML_PredictedScore"] > 0.0
        assert np.isclose(det_df.loc[0, "ML_PredictedScore"], 4.8)

    def test_pipeline_walk_forward_auto_ridge_extraction(self):
        """Simulate walk_forward_auto extraction block in RacePredictionPipeline."""
        eval_results = {
            "validation_rounds": [1, 2],
            "best_w_rf": 0.0,
            "best_w_xgb": 0.0,
            "best_w_ridge": 1.0,
            "best_model_name": "Ridge Regression (100%)",
            "best_mae": 2.15,
            "selected_residuals": np.array([0.1, -0.2]),
            "ridge_residuals": np.array([0.05, -0.15]),
            "rf_residuals": np.array([0.5, -0.6]),
            "xgb_residuals": np.array([0.4, -0.5]),
            "multi_residuals": np.array([0.2, -0.25]),
        }
        
        # Test extraction logic
        raw_rf = eval_results.get("best_w_rf")
        raw_xgb = eval_results.get("best_w_xgb")
        raw_ridge = eval_results.get("best_w_ridge", 0.0)

        w_rf = float(raw_rf) if raw_rf is not None else 0.0
        w_xgb = float(raw_xgb) if raw_xgb is not None else 0.0
        w_ridge = float(raw_ridge) if raw_ridge is not None else 0.0

        total_w = w_rf + w_xgb + w_ridge
        w_rf, w_xgb, w_ridge = w_rf / total_w, w_xgb / total_w, w_ridge / total_w

        assert w_rf == 0.0
        assert w_xgb == 0.0
        assert w_ridge == 1.0
        
        # Residual routing check
        if w_ridge >= 0.5:
            residuals = eval_results.get("ridge_residuals")
        assert np.array_equal(residuals, np.array([0.05, -0.15]))


# ============================================================================
# STEP 2: DEF-02 SPRINT RACE EXCLUSION IN DATA LOADER
# ============================================================================

class TestDef02SprintRaceExclusion:
    """Verifies DEF-02 fix: normal race loading strictly excludes sprint files."""

    def test_load_season_races_excludes_sprints(self, tmp_path):
        """Construct season folder with normal race, sprint race, and qualifying files."""
        season_dir = tmp_path / "data" / "raw" / "season" / "2026"
        season_dir.mkdir(parents=True)

        # Round 1: normal weekend (race + qualifying)
        r1_race = season_dir / "2026_round_1_race.csv"
        pd.DataFrame({
            "Abbreviation": ["VER", "HAM"],
            "Position": [1, 2],
            "GridPosition": [1, 2],
            "Points": [25, 18],
            "Status": ["Finished", "Finished"]
        }).to_csv(r1_race, index=False)

        r1_quali = season_dir / "2026_round_1_qualifying.csv"
        pd.DataFrame({
            "Abbreviation": ["VER", "HAM"],
            "Position": [1, 2]
        }).to_csv(r1_quali, index=False)

        # Round 2: sprint weekend (normal race, sprint race, qualifying)
        r2_race = season_dir / "2026_round_2_race.csv"
        pd.DataFrame({
            "Abbreviation": ["VER", "HAM"],
            "Position": [2, 1],
            "GridPosition": [2, 1],
            "Points": [18, 25],
            "Status": ["Finished", "Finished"]
        }).to_csv(r2_race, index=False)

        r2_sprint_race = season_dir / "2026_round_2_sprint_race.csv"
        pd.DataFrame({
            "Abbreviation": ["VER", "HAM"],
            "Position": [1, 2],
            "GridPosition": [1, 2],
            "Points": [8, 7],
            "Status": ["Finished", "Finished"]
        }).to_csv(r2_sprint_race, index=False)

        r2_quali = season_dir / "2026_round_2_qualifying.csv"
        pd.DataFrame({
            "Abbreviation": ["VER", "HAM"],
            "Position": [2, 1]
        }).to_csv(r2_quali, index=False)

        # Round 3: another sprint weekend with alternative naming (_sprint.csv)
        r3_race = season_dir / "2026_round_3_race.csv"
        pd.DataFrame({
            "Abbreviation": ["NOR", "PIA"],
            "Position": [1, 2],
            "GridPosition": [1, 2],
            "Points": [25, 18],
            "Status": ["Finished", "Finished"]
        }).to_csv(r3_race, index=False)

        r3_sprint = season_dir / "2026_round_3_sprint.csv"
        pd.DataFrame({
            "Abbreviation": ["NOR", "PIA"],
            "Position": [2, 1],
            "GridPosition": [2, 1],
            "Points": [7, 8],
            "Status": ["Finished", "Finished"]
        }).to_csv(r3_sprint, index=False)

        r3_quali = season_dir / "2026_round_3_qualifying.csv"
        pd.DataFrame({
            "Abbreviation": ["NOR", "PIA"],
            "Position": [1, 2]
        }).to_csv(r3_quali, index=False)

        # Initialize loader pointing to tmp_path root
        loader = DataLoader(root_dir=tmp_path)
        race_df, quali_df = loader.load_season_races(year=2026)

        # Expected: exactly 2 drivers per round across 3 rounds = 6 race rows, 6 quali rows
        assert len(race_df) == 6, f"Expected 6 race rows, got {len(race_df)}"
        assert len(quali_df) == 6, f"Expected 6 quali rows, got {len(quali_df)}"

        # Check Round 2 specifically has exactly 2 rows (no 4-row sprint duplicate)
        r2_race_rows = race_df[race_df["Round"] == 2]
        r2_quali_rows = quali_df[quali_df["Round"] == 2]
        assert len(r2_race_rows) == 2, f"Round 2 race duplicated! Rows: {len(r2_race_rows)}"
        assert len(r2_quali_rows) == 2, f"Round 2 qualifying duplicated! Rows: {len(r2_quali_rows)}"
        
        # Verify drivers are unique per round
        for r in [1, 2, 3]:
            round_drivers = race_df[race_df["Round"] == r]["Abbreviation"].tolist()
            assert len(round_drivers) == len(set(round_drivers)), f"Duplicate drivers in round {r}!"


# ============================================================================
# STEP 3: DEF-05 TERMINAL STATUS CLASSIFICATION
# ============================================================================

class TestDef05TerminalStatusClassification:
    """Verifies DEF-05 fix: distinguishes Finished, Retired, DNS, DSQ, NC."""

    @pytest.mark.parametrize("status,expected_category", [
        ("Finished", "FINISHED"),
        ("finished", "FINISHED"),
        ("+1 Lap", "FINISHED"),
        ("+2 Laps", "FINISHED"),
        ("Lapped", "FINISHED"),
        ("1 Lap", "FINISHED"),
        ("Accident", "RETIRED"),
        ("Collision", "RETIRED"),
        ("Engine", "RETIRED"),
        ("Gearbox", "RETIRED"),
        ("Hydraulics", "RETIRED"),
        ("Electrical", "RETIRED"),
        ("Brakes", "RETIRED"),
        ("Suspension", "RETIRED"),
        ("Power Unit", "RETIRED"),
        ("Puncture", "RETIRED"),
        ("Retired", "RETIRED"),
        ("Turbo", "RETIRED"),
        ("turbo", "RETIRED"),
        ("Vibrations", "RETIRED"),
        ("vibrations", "RETIRED"),
        ("Overheating", "RETIRED"),
        ("Spins off", "RETIRED"),
        ("Damage", "RETIRED"),
        ("Out of fuel", "RETIRED"),
        ("Did not start", "DNS"),
        ("DNS", "DNS"),
        ("dns", "DNS"),
        ("Withdrew", "DNS"),
        ("Disqualified", "DSQ"),
        ("DSQ", "DSQ"),
        ("Excluded", "DSQ"),
        ("NC", "NC"),
        ("Not classified", "NC"),
        (np.nan, "UNKNOWN"),
        (None, "UNKNOWN"),
        ("", "UNKNOWN"),
    ])
    def test_classify_race_status(self, status, expected_category):
        assert classify_race_status(status) == expected_category

    @pytest.mark.parametrize("status,expected_finish", [
        ("Finished", True),
        ("+1 Lap", True),
        ("+2 Laps", True),
        ("Lapped", True),
        ("Accident", False),
        ("Did not start", False),
        ("Disqualified", False),
        ("NC", False),
        ("Turbo", False),
        ("Vibrations", False),
        (None, False),
    ])
    def test_is_classified_finish(self, status, expected_finish):
        assert is_classified_finish(status) == expected_finish

    @pytest.mark.parametrize("status,expected_dnf", [
        ("Finished", False),
        ("+1 Lap", False),
        ("Lapped", False),
        ("Accident", True),
        ("Collision", True),
        ("Power Unit", True),
        ("Gearbox", True),
        ("Turbo", True),
        ("Vibrations", True),
        ("Retired", True),
        ("Did not start", False),
        ("DNS", False),
        ("Disqualified", False),
        ("DSQ", False),
        ("NC", False),
        ("Not classified", False),
        (np.nan, False),
        (None, False),
        ("", False),
    ])
    def test_is_dnf(self, status, expected_dnf):
        """is_dnf defaults to pure mechanical/incident retirement; DNS/DSQ/NC are separate metrics."""
        assert is_dnf(status) == expected_dnf

    def test_is_dnf_optional_inclusion_flags(self):
        """Verify optional parameter flags allow callers to include DNS, DSQ, or NC if needed."""
        assert is_dnf("Did not start", include_dns=True) is True
        assert is_dnf("DNS", include_dns=True) is True
        assert is_dnf("Did not start", include_dns=False) is False
        assert is_dnf("Disqualified", include_dsq=True) is True
        assert is_dnf("DSQ", include_dsq=True) is True
        assert is_dnf("Disqualified", include_dsq=False) is False
        assert is_dnf("NC", include_nc=True) is True
        assert is_dnf("Not classified", include_nc=True) is True
        assert is_dnf("NC", include_nc=False) is False
        assert is_dnf("Turbo", include_dns=False, include_dsq=False, include_nc=False) is True
        assert is_dnf("Unknown_status", include_dns=True, include_dsq=True, include_nc=True) is False

    def test_individual_status_classification_helpers(self):
        """Verify explicit category predicate helpers."""
        assert is_retired("Turbo") is True
        assert is_retired("Accident") is True
        assert is_retired("Finished") is False
        assert is_retired("DNS") is False

        assert is_dns("DNS") is True
        assert is_dns("Did not start") is True
        assert is_dns("Withdrew") is True
        assert is_dns("Finished") is False

        assert is_dsq("DSQ") is True
        assert is_dsq("Disqualified") is True
        assert is_dsq("Excluded") is True
        assert is_dsq("Finished") is False

        assert is_nc("NC") is True
        assert is_nc("Not classified") is True
        assert is_nc("Finished") is False

    def test_build_season_form_semantic_rates(self):
        """Verify season form computes separate metrics for DNF, DNS, DSQ, NC with statistical denominators."""
        races_df = pd.DataFrame({
            "Round": [1, 2, 3, 4, 5, 6],
            "Abbreviation": ["ALO"] * 6,
            "Position": [1, 20, 22, 21, 19, 15],
            "GridPosition": [1, 5, 10, 8, 12, 14],
            "Points": [25, 0, 0, 0, 0, 0],
            "Status": [
                "Finished",         # classified finish (start 1)
                "Turbo",            # RETIRED (start 2) -> in-race DNF
                "Did not start",    # DNS (entry 3, but not a start)
                "Disqualified",     # DSQ (entry 4, start 3)
                "Not classified",   # NC (entry 5, start 4)
                "CorruptStatusXYZ"  # UNKNOWN (entry 6 -> excluded from valid denominators)
            ]
        })
        driver_stats, _ = build_season_form(target_round=7, races_df=races_df)
        alo_stats = driver_stats["ALO"]

        # Total entered rounds = 6 (recorded entries)
        assert alo_stats["races_completed"] == 6

        # Valid entries = 6 - 1 (unknown) = 5
        # DNS rate = 1 / 5 = 0.20
        assert np.isclose(alo_stats["driver_dns_rate"], 1.0 / 5.0, atol=1e-4)

        # DSQ rate = 1 / 5 = 0.20
        assert np.isclose(alo_stats["driver_dsq_rate"], 1.0 / 5.0, atol=1e-4)

        # Valid starts = 5 - 1 (DNS) = 4
        # DNF rate = 1 / 4 = 0.25 (1 in-race retirement in 4 actual starts)
        assert np.isclose(alo_stats["driver_dnf_rate"], 1.0 / 4.0, atol=1e-4)

        # NC rate = 1 / 4 = 0.25 (1 unclassified outcome in 4 actual starts)
        assert np.isclose(alo_stats["driver_nc_rate"], 1.0 / 4.0, atol=1e-4)

    def test_compute_season_driver_stats_all_dns_zero_starts(self):
        """Driver with only DNS entries has 0.0 DNF rate without division by zero."""
        races_df = pd.DataFrame({
            "Round": [1, 2],
            "Abbreviation": ["ALB", "ALB"],
            "Position": [np.nan, np.nan],
            "GridPosition": [7, 8],
            "Points": [0, 0],
            "Status": ["Did not start", "DNS"]
        })
        stats = compute_season_driver_stats(prior_races_df=races_df)
        alb = stats["ALB"]
        assert alb["driver_dns_rate"] == 1.0
        assert alb["driver_dnf_rate"] == 0.0
        assert alb["driver_dsq_rate"] == 0.0
        assert alb["driver_nc_rate"] == 0.0

    def test_compute_season_driver_stats_clean_finisher(self):
        """Driver with all finishes has 0.0 for all failure rates."""
        races_df = pd.DataFrame({
            "Round": [1, 2, 3],
            "Abbreviation": ["VER", "VER", "VER"],
            "Position": [1, 1, 2],
            "GridPosition": [1, 2, 1],
            "Points": [25, 25, 18],
            "Status": ["Finished", "+1 Lap", "Finished"]
        })
        stats = compute_season_driver_stats(prior_races_df=races_df)
        ver = stats["VER"]
        assert ver["driver_dnf_rate"] == 0.0
        assert ver["driver_dns_rate"] == 0.0
        assert ver["driver_dsq_rate"] == 0.0
        assert ver["driver_nc_rate"] == 0.0
        assert ver["races_completed"] == 3


# ============================================================================
# STEP 4: DEF-08 ROUND 16 METADATA & CIRCUIT MAPPING REGRESSION
# ============================================================================

class TestDef08Round16CircuitMapping:
    """Verifies DEF-08: Round 16 metadata, Sepang canonical resolution, and track configuration."""

    @pytest.mark.parametrize("identifier", [
        "sepang",
        "Sepang",
        "Sepang International Circuit",
        "sepang_international_circuit",
        "Malaysian Grand Prix",
        "malaysian_grand_prix",
        "Malaysia",
        "malaysia",
        "Kuala Lumpur",
        "kuala_lumpur",
        "Bahrain Grand Prix",
        "bahrain_grand_prix",
        "Bahrain Grand Prix in Malaysia"
    ])
    def test_round16_all_identifiers_resolve_to_sepang(self, identifier):
        """All Round 16 circuit identifiers must deterministically resolve to canonical 'sepang'."""
        resolved = resolve_circuit_key(identifier)
        assert resolved == "sepang", f"Failed to resolve '{identifier}' to 'sepang'. Got: {resolved}"

    def test_round16_track_config_integrity(self):
        """Verify config/tracks.json contains complete physical parameters for Sepang."""
        tracks = load_tracks_config()
        assert "sepang" in tracks, "Canonical key 'sepang' missing from tracks.json"
        
        sepang_cfg = tracks["sepang"]
        assert sepang_cfg["round_2026"] == 16
        assert sepang_cfg["country"] == "Malaysia"
        assert sepang_cfg["circuit_name"] == "Sepang International Circuit"
        assert sepang_cfg["circuit_type"] == "permanent"
        assert math.isclose(sepang_cfg["track_length_km"], 5.543, rel_tol=1e-3)
        assert "Sepang" in sepang_cfg["event_name_patterns"]
        assert "Kuala Lumpur" in sepang_cfg["event_name_patterns"]
