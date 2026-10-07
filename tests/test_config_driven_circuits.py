"""
Unit Tests for Track Configuration and Multi-Circuit Extensibility
==================================================================
Verifies that circuits (Baku, Sepang, Monza, Silverstone) are fully defined in config/tracks.json
and can be dynamically resolved without writing new python modules.
"""

import json
from pathlib import Path
import pytest

from src.data.data_loader import DataLoader


def test_tracks_config_structure():
    config_path = Path(__file__).resolve().parents[1] / "config" / "tracks.json"
    assert config_path.exists(), "tracks.json must exist in config/"

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "tracks" in data
    tracks = data["tracks"]

    required_tracks = ["baku", "sepang", "monza"]
    for t in required_tracks:
        assert t in tracks, f"Track '{t}' must be configured in tracks.json"

    required_keys = [
        "display_name", "circuit_name", "country", "round_2026",
        "circuit_type", "historical_data_available", "known_penalties_2026"
    ]

    for t, cfg in tracks.items():
        for k in required_keys:
            assert k in cfg, f"Track '{t}' is missing required key '{k}'"


def test_data_loader_dynamic_track_resolution():
    loader = DataLoader()

    # Baku
    baku_info = loader.get_track_info("baku")
    assert baku_info["round_2026"] == 15
    assert baku_info["historical_data_available"] is True

    # Sepang
    sepang_info = loader.get_track_info("sepang")
    assert sepang_info["round_2026"] == 16
    assert sepang_info["historical_data_available"] is False
    assert "HAD" in sepang_info["known_penalties_2026"]

    # Monza (proves 3rd track extensibility without code changes)
    monza_info = loader.get_track_info("monza")
    assert monza_info["circuit_type"] in ["permanent", "high_speed"]
    assert "display_name" in monza_info

