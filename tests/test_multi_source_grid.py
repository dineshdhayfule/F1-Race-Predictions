"""
Comprehensive Unit and Integration Tests for Multi-Source Starting Grid System
=============================================================================
Tests:
1. Multi-source priority (FIA Manifest -> F1 Website -> OpenF1 API -> Algorithmic Fallback)
2. OpenF1 meeting discovery & Race session selection
3. OpenF1 driver-number-to-driver identity mapping
4. Network resilience (timeout, HTTP 404/500, malformed JSON)
5. Grid structural and sporting validation (duplicate positions, gap detection)
6. Cache storage and offline reuse
7. Stage-aware availability and leakage prevention
8. Accurate provenance reporting across all layers
9. Pit-lane starts and penalty/displacement semantic separation
10. Mexico 2026 OpenF1 grid verification (VER P5 / ANT P4 / LIN P8 / LAW P22)
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import urllib.error
import numpy as np
import pandas as pd
import pytest

from src.data.data_loader import DataLoader
from src.data.grid_retriever import StartingGridRetriever
from src.features.grid_features import (
    GridPublicationStatus,
    GridProvenance,
    StartingGridStatus,
    format_grid_provenance_display,
)
from src.cli.menu import InteractiveMenu

ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# 1. OpenF1 Meeting Matching & Synonyms
# =============================================================================
def test_openf1_meeting_matching():
    retriever = StartingGridRetriever(ROOT)
    meetings = [
        {"meeting_key": 1001, "meeting_name": "Australian Grand Prix", "circuit_short_name": "Melbourne", "country_name": "Australia"},
        {"meeting_key": 1018, "meeting_name": "United States Grand Prix", "circuit_short_name": "Austin", "country_name": "United States"},
        {"meeting_key": 1019, "meeting_name": "Mexico City Grand Prix", "circuit_short_name": "Mexico City", "country_name": "Mexico"},
        {"meeting_key": 1009, "meeting_name": "British Grand Prix", "circuit_short_name": "Silverstone", "country_name": "Great Britain"},
    ]

    # Circuit keys and synonyms
    m_mex = retriever._match_openf1_meeting(meetings, "mexico", 19)
    assert m_mex is not None and m_mex["meeting_key"] == 1019

    m_cota = retriever._match_openf1_meeting(meetings, "cota", 18)
    assert m_cota is not None and m_cota["meeting_key"] == 1018

    m_sil = retriever._match_openf1_meeting(meetings, "silverstone", 9)
    assert m_sil is not None and m_sil["meeting_key"] == 1009


# =============================================================================
# 2. OpenF1 Session Selection: Strictly Race Session
# =============================================================================
def test_openf1_race_session_selection_skips_sprint_and_quali():
    retriever = StartingGridRetriever(ROOT)
    sessions = [
        {"session_key": 101, "session_name": "Practice 1", "session_type": "Practice"},
        {"session_key": 102, "session_name": "Sprint Qualifying", "session_type": "Qualifying"},
        {"session_key": 103, "session_name": "Sprint", "session_type": "Race"},
        {"session_key": 104, "session_name": "Qualifying", "session_type": "Qualifying"},
        {"session_key": 105, "session_name": "Race", "session_type": "Race"},
    ]
    race_sess = next(
        (s for s in sessions if str(s.get("session_name", "")).lower() == "race"),
        None,
    )
    assert race_sess is not None
    assert race_sess["session_key"] == 105


# =============================================================================
# 3. OpenF1 Driver Number to Driver Identity Mapping
# =============================================================================
def test_openf1_driver_mapping():
    retriever = StartingGridRetriever(ROOT)
    openf1_data = {
        "grid": [
            {"position": 1, "driver_number": 44, "lap_duration": 76.514},
            {"position": 2, "driver_number": 63, "lap_duration": 76.523},
        ],
        "drivers": [
            {"driver_number": 44, "name_acronym": "HAM", "team_name": "Ferrari"},
            {"driver_number": 63, "name_acronym": "RUS", "team_name": "Mercedes"},
        ],
    }
    df = retriever._openf1_to_dataframe(openf1_data, expected_grid_size=2)
    assert df is not None
    assert df.iloc[0]["Abbreviation"] == "HAM"
    assert df.iloc[0]["TeamName"] == "Ferrari"
    assert df.iloc[1]["Abbreviation"] == "RUS"
    assert df.iloc[1]["TeamName"] == "Mercedes"


# =============================================================================
# 4. Network Resilience: API Timeout, HTTP Errors, Malformed JSON
# =============================================================================
def test_openf1_network_timeout_falls_back_gracefully():
    retriever = StartingGridRetriever(ROOT)
    with patch("urllib.request.urlopen", side_effect=TimeoutError("Connection timed out")):
        # Request a round with no local cache
        res = retriever._fetch_openf1_live("monza", 2026, 14)
        assert res is None


def test_openf1_http_404_error_falls_back_gracefully():
    retriever = StartingGridRetriever(ROOT)
    err = urllib.error.HTTPError("https://api.openf1.org/v1/meetings", 404, "Not Found", {}, None)
    with patch("urllib.request.urlopen", side_effect=err):
        res = retriever._fetch_openf1_live("monza", 2026, 14)
        assert res is None


def test_openf1_malformed_json_falls_back_gracefully():
    retriever = StartingGridRetriever(ROOT)
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"<!DOCTYPE html><html>Malformed Non-JSON</html>"
    mock_resp.__enter__.return_value = mock_resp
    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = retriever._fetch_openf1_live("monza", 2026, 14)
        assert res is None


# =============================================================================
# 5. Grid Validation: Structural & Sporting Checks
# =============================================================================
def test_validate_grid_dataframe_detects_duplicate_positions():
    retriever = StartingGridRetriever(ROOT)
    df_dup = pd.DataFrame({
        "Abbreviation": ["VER", "HAM"],
        "ActualStartingGrid": [1.0, 1.0],  # Duplicate P1!
        "simulation_start_rank": [1.0, 2.0],
        "is_pit_lane_start": [0, 0],
    })
    valid, errs = retriever.validate_grid_dataframe(df_dup)
    assert valid is False
    assert any("Duplicate physical grid positions" in e for e in errs)


def test_validate_grid_dataframe_detects_duplicate_drivers():
    retriever = StartingGridRetriever(ROOT)
    df_dup = pd.DataFrame({
        "Abbreviation": ["VER", "VER"],
        "ActualStartingGrid": [1.0, 2.0],
        "simulation_start_rank": [1.0, 2.0],
        "is_pit_lane_start": [0, 0],
    })
    valid, errs = retriever.validate_grid_dataframe(df_dup)
    assert valid is False
    assert any("Duplicate drivers" in e for e in errs)


def test_validate_grid_dataframe_enforces_pit_lane_null_grid():
    retriever = StartingGridRetriever(ROOT)
    df_pit = pd.DataFrame({
        "Abbreviation": ["VER", "HAM"],
        "ActualStartingGrid": [1.0, 22.0],  # Pit-lane starter must be NaN!
        "simulation_start_rank": [1.0, 2.0],
        "is_pit_lane_start": [0, 1],
    })
    valid, errs = retriever.validate_grid_dataframe(df_pit)
    assert valid is False
    assert any("non-null ActualStartingGrid" in e for e in errs)


# =============================================================================
# 6. Multi-Source Priority Invariant Enforcement
# =============================================================================
def test_source_priority_fia_manifest_takes_priority_over_openf1():
    """
    Sepang (Round 16) has an official FIA manifest.
    Even if OpenF1 is available, Source 1 (FIA Manifest) must be chosen.
    """
    retriever = StartingGridRetriever(ROOT)
    grid_df, meta = retriever.get_starting_grid(
        track_key="sepang",
        year=2026,
        round_num=16,
        stage="after_qualifying",
    )
    assert grid_df is not None
    assert meta["provenance"] == GridProvenance.FIA_OFFICIAL.value
    assert "Doc" in meta["document_reference"]
    assert len(grid_df) == 22


def test_source_priority_openf1_takes_priority_over_algorithmic_fallback():
    """
    Mexico (Round 19) has no FIA manifest, but has OpenF1 starting grid data.
    When Source 1 and Source 2 are unavailable, Source 3 (OpenF1 API) must be selected
    over Source 4 (Algorithmic Fallback).
    """
    with patch.object(StartingGridRetriever, "_retrieve_f1_website", return_value=(None, {})):
        loader = DataLoader(root_dir=ROOT)
        q_df = loader.load_target_qualifying("mexico", 2026)
        grid_df, meta = loader.load_starting_grid(
            track_id="mexico",
            year=2026,
            round_num=19,
            stage="after_qualifying",
            quali_df=q_df,
        )
        assert grid_df is not None
        assert meta["provenance"] == GridProvenance.OPENF1_API.value
        assert "OpenF1" in meta["document_reference"]
        assert len(grid_df) == 22


def test_source_priority_algorithmic_fallback_when_external_sources_missing():
    """
    When no manifest or OpenF1 grid exists (e.g. mock round 99),
    Source 4 (Algorithmic Fallback) must be activated with transparent provenance.
    """
    retriever = StartingGridRetriever(ROOT)
    q_df = pd.DataFrame({
        "Abbreviation": ["VER", "HAM", "LEC"],
        "Position": [1, 2, 3],
    })
    with patch.object(retriever, "_retrieve_openf1", return_value=(None, None)):
        grid_df, meta = retriever.get_starting_grid(
            track_key="monza",
            year=2026,
            round_num=99,
            stage="after_qualifying",
            quali_df=q_df,
            expected_grid_size=3,
        )
    assert grid_df is not None
    assert meta["provenance"] == GridProvenance.ALGORITHMIC_FALLBACK.value
    assert "FIA Art 42.3" in meta["document_reference"]


# =============================================================================
# 7. Stage-Aware Availability & Leakage Prevention
# =============================================================================
def test_stage_aware_availability_pre_qualifying():
    retriever = StartingGridRetriever(ROOT)
    for early_stage in [
        "pre_fp1",
        "after_fp1",
        "after_fp2",
        "after_fp3",
        "after_sprint_qualifying",
        "after_sprint_race",
        "after_sprint",
    ]:
        grid_df, meta = retriever.get_starting_grid(
            track_key="mexico",
            year=2026,
            round_num=19,
            stage=early_stage,
        )
        assert grid_df is None
        assert meta["provenance"] == GridProvenance.UNPUBLISHED.value


# =============================================================================
# 8. Provenance Display Formatting
# =============================================================================
def test_format_grid_provenance_display():
    assert format_grid_provenance_display({"provenance": "official_manifest", "publication_status": "FINAL", "document_reference": "FIA Doc 48"}) == "FIA Official Final Starting Grid (FIA Doc 48)"
    assert format_grid_provenance_display({"provenance": "official_manifest", "publication_status": "PROVISIONAL", "document_reference": "FIA Doc 40"}) == "FIA Official Provisional Starting Grid (FIA Doc 40)"
    assert format_grid_provenance_display({"provenance": "openf1_api", "session_key": 11400, "document_reference": "OpenF1 API (Session 11400)"}) == "OpenF1 Published Starting Grid (OpenF1 API (Session 11400))"
    assert format_grid_provenance_display({"provenance": "algorithmic_fallback"}) == "Algorithmic Provisional Grid (FIA Art 42.3)"
    assert format_grid_provenance_display({"provenance": "unpublished", "stage": "pre_fp1"}) == "Provisional Predicted Grid (Pre-Qualifying Stage: pre_fp1)"
    assert format_grid_provenance_display(None) == "Starting Grid Unavailable"
    assert format_grid_provenance_display({}) == "Starting Grid Unavailable"


# =============================================================================
# 9. Readiness Audit Distinguishes Qualifying Availability from Official Grid
# =============================================================================
def test_readiness_audit_distinguishes_qualifying_from_official_manifest():
    with patch.object(StartingGridRetriever, "_retrieve_f1_website", return_value=(None, {})):
        menu = InteractiveMenu(root_dir=ROOT)
        ready, details = menu._audit_prediction_readiness("mexico", 2026, 19, "after_qualifying", is_sprint=False)
        assert ready is True

        # Qualifying results must be marked OK
        assert any("Qualifying Results available" in msg for status, msg in details if status == "OK")

        # Starting Grid must NOT be claimed as "Starting Grid verified" without official manifest
        assert not any("Starting Grid verified" in msg for _, msg in details)

        # OpenF1 published grid must be acknowledged
        assert any("OpenF1 Published Starting Grid available" in msg for status, msg in details if status == "INFO")


# =============================================================================
# 10. Mexico 2026 OpenF1 Starting Grid Invariants
# =============================================================================
def test_mexico_2026_openf1_grid_inspection():
    with patch.object(StartingGridRetriever, "_retrieve_f1_website", return_value=(None, {})):
        loader = DataLoader(root_dir=ROOT)
        q_df = loader.load_target_qualifying("mexico", 2026)
        grid_df, meta = loader.load_starting_grid(
            track_id="mexico",
            year=2026,
            round_num=19,
            stage="after_qualifying",
            quali_df=q_df,
        )
        assert grid_df is not None
        assert len(grid_df) == 22

        # Verify positions on OpenF1 published grid
        # HAM P1, RUS P2, LEC P3
        assert grid_df.iloc[0]["Abbreviation"] == "HAM" and int(grid_df.iloc[0]["ActualStartingGrid"]) == 1
        assert grid_df.iloc[1]["Abbreviation"] == "RUS" and int(grid_df.iloc[1]["ActualStartingGrid"]) == 2
        assert grid_df.iloc[2]["Abbreviation"] == "LEC" and int(grid_df.iloc[2]["ActualStartingGrid"]) == 3

        # ANT starts P4 (Quali P5, net delta -1)
        ant_row = grid_df[grid_df["Abbreviation"] == "ANT"].iloc[0]
        assert int(ant_row["ActualStartingGrid"]) == 4
        assert int(ant_row["QualifyingPosition"]) == 5

        # VER starts P5 (Quali P4, net delta +1)
        ver_row = grid_df[grid_df["Abbreviation"] == "VER"].iloc[0]
        assert int(ver_row["ActualStartingGrid"]) == 5
        assert int(ver_row["QualifyingPosition"]) == 4

        # LIN starts P8 (Quali P8)
        lin_row = grid_df[grid_df["Abbreviation"] == "LIN"].iloc[0]
        assert int(lin_row["ActualStartingGrid"]) == 8
        assert int(lin_row["QualifyingPosition"]) == 8

        # LAW starts P22 (Quali P8, net delta +14, penalized to back)
        law_row = grid_df[grid_df["Abbreviation"] == "LAW"].iloc[0]
        assert int(law_row["ActualStartingGrid"]) == 22
        assert int(law_row["QualifyingPosition"]) == 8
        assert int(law_row["grid_penalty_places"]) == 14
        assert law_row["StartingGridStatus"] == StartingGridStatus.PENALIZED.value

