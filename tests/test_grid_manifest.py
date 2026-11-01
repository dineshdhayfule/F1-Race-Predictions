"""
Unit and Integration Tests for Starting-Grid Manifest and Retrieval System
==========================================================================
Verifies:
- Canonical manifest schema validation.
- Invariant enforcement: physical contiguity (1..N), pit-lane nullness, simulation rank.
- Provenance metadata preservation (official manifest vs algorithmic fallback).
- Stage-aware cutoff enforcement (no published grid before qualifying).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.grid_retriever import StartingGridRetriever
from src.features.grid_features import StartingGridStatus, GridPublicationStatus

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_schema_validation_success():
    retriever = StartingGridRetriever(ROOT)
    # Valid manifest
    data = {
        "season": 2026,
        "round": 16,
        "circuit_key": "sepang",
        "publication_status": "FINAL",
        "grid": [
            {
                "driver": "VER",
                "grid_pos": 1,
                "simulation_start_rank": 1,
                "is_pit_lane_start": 0,
            },
            {
                "driver": "HAM",
                "grid_pos": 2,
                "simulation_start_rank": 2,
                "is_pit_lane_start": 0,
            },
        ],
    }
    valid, errs = retriever.validate_manifest_json(data)
    assert valid is True
    assert len(errs) == 0


def test_manifest_schema_validation_detects_duplicate_driver():
    retriever = StartingGridRetriever(ROOT)
    data = {
        "season": 2026,
        "round": 16,
        "circuit_key": "sepang",
        "publication_status": "FINAL",
        "grid": [
            {"driver": "VER", "grid_pos": 1, "simulation_start_rank": 1, "is_pit_lane_start": 0},
            {"driver": "VER", "grid_pos": 2, "simulation_start_rank": 2, "is_pit_lane_start": 0},
        ],
    }
    valid, errs = retriever.validate_manifest_json(data)
    assert valid is False
    assert any("Duplicate driver" in e for e in errs)


def test_manifest_schema_validation_detects_pit_lane_non_null():
    retriever = StartingGridRetriever(ROOT)
    # Pit-lane starter has physical grid position 22 instead of null
    data = {
        "season": 2026,
        "round": 16,
        "circuit_key": "sepang",
        "publication_status": "FINAL",
        "grid": [
            {"driver": "VER", "grid_pos": 1, "simulation_start_rank": 1, "is_pit_lane_start": 0},
            {"driver": "HAM", "grid_pos": 22, "simulation_start_rank": 2, "is_pit_lane_start": 1},
        ],
    }
    valid, errs = retriever.validate_manifest_json(data)
    assert valid is False
    assert any("Pit-lane starter" in e and "null" in e for e in errs)


def test_load_sepang_official_manifest():
    retriever = StartingGridRetriever(ROOT)
    grid_df, meta = retriever.get_starting_grid(
        track_key="sepang",
        year=2026,
        round_num=16,
        stage="after_qualifying"
    )
    assert grid_df is not None
    assert len(grid_df) == 22
    assert meta["provenance"] == "official_manifest"
    assert "Doc" in meta["document_reference"]

    # Verify key driver starting positions
    assert int(grid_df.loc[grid_df["Abbreviation"] == "VER", "ActualStartingGrid"].iloc[0]) == 1
    assert int(grid_df.loc[grid_df["Abbreviation"] == "HAM", "ActualStartingGrid"].iloc[0]) == 2
    assert int(grid_df.loc[grid_df["Abbreviation"] == "HAD", "ActualStartingGrid"].iloc[0]) == 8
    assert int(grid_df.loc[grid_df["Abbreviation"] == "COL", "ActualStartingGrid"].iloc[0]) == 21
    assert int(grid_df.loc[grid_df["Abbreviation"] == "LIN", "ActualStartingGrid"].iloc[0]) == 22

    # Physical grid contiguity 1..22
    assert grid_df["ActualStartingGrid"].tolist() == list(range(1, 23))


def test_load_baku_official_manifest():
    retriever = StartingGridRetriever(ROOT)
    grid_df, meta = retriever.get_starting_grid(
        track_key="baku",
        year=2026,
        round_num=15,
        stage="after_qualifying"
    )
    assert grid_df is not None
    assert len(grid_df) == 22
    assert meta["provenance"] == "official_manifest"
    assert grid_df.iloc[0]["Abbreviation"] == "RUS"
    assert grid_df.iloc[1]["Abbreviation"] == "LEC"
    assert grid_df.iloc[2]["Abbreviation"] == "PIA"


def test_load_silverstone_manifest_with_ten_place_penalty():
    retriever = StartingGridRetriever(ROOT)
    grid_df, meta = retriever.get_starting_grid(
        track_key="silverstone",
        year=2026,
        round_num=9,
        stage="after_qualifying"
    )
    assert grid_df is not None
    assert len(grid_df) == 22
    # NOR qualified P3, has 10 place penalty -> Starts P13
    nor_row = grid_df[grid_df["Abbreviation"] == "NOR"].iloc[0]
    assert int(nor_row["QualifyingPosition"]) == 3
    assert int(nor_row["ActualStartingGrid"]) == 13
    assert int(nor_row["grid_penalty_places"]) == 10


def test_pre_qualifying_stage_returns_unpublished():
    retriever = StartingGridRetriever(ROOT)
    for early_stage in ["pre_fp1", "after_fp1", "after_fp2", "after_fp3"]:
        grid_df, meta = retriever.get_starting_grid(
            track_key="sepang",
            year=2026,
            round_num=16,
            stage=early_stage
        )
        assert grid_df is None
        assert meta["provenance"] == "unpublished"


def test_algorithmic_fallback_when_manifest_missing():
    retriever = StartingGridRetriever(ROOT)
    # Circuit without manifest file (e.g. Round 99)
    q_df = pd.DataFrame({
        "Abbreviation": ["VER", "HAM", "LEC"],
        "Position": [1, 2, 3]
    })
    grid_df, meta = retriever.get_starting_grid(
        track_key="monza",
        year=2026,
        round_num=99,
        stage="after_qualifying",
        quali_df=q_df,
        expected_grid_size=3
    )
    assert grid_df is not None
    assert len(grid_df) == 3
    assert meta["provenance"] == "algorithmic_fallback"
    assert meta["publication_status"] == GridPublicationStatus.PROVISIONAL.value


def test_dataloader_load_starting_grid_sepang():
    from src.data.data_loader import DataLoader
    loader = DataLoader(root_dir=ROOT)
    grid_df, meta = loader.load_starting_grid(
        track_id="sepang",
        year=2026,
        round_num=16,
        stage="after_qualifying"
    )
    assert grid_df is not None
    assert meta["provenance"] == "official_manifest"
    assert len(grid_df) == 22
    assert grid_df.iloc[0]["Abbreviation"] == "VER"


def test_dataloader_load_starting_grid_silverstone_round_9():
    from src.data.data_loader import DataLoader
    loader = DataLoader(root_dir=ROOT)
    grid_df, meta = loader.load_starting_grid(
        track_id="silverstone",
        year=2026,
        round_num=9,
        stage="after_qualifying"
    )
    assert grid_df is not None
    assert meta["provenance"] == "official_manifest"
    assert len(grid_df) == 22
    nor = grid_df[grid_df["Abbreviation"] == "NOR"].iloc[0]
    assert int(nor["ActualStartingGrid"]) == 13


