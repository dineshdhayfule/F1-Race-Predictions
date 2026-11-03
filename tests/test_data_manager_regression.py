"""
Regression tests for DataManager path safety, historical protection, and cache discovery.
"""

from pathlib import Path
import json
import os
import sys
import pytest

from src.data.data_manager import DataManager
from src.data.deletion_audit import DeletionAuditLogger


@pytest.fixture
def isolated_env(tmp_path: Path):
    ws = tmp_path / "sandbox_repo"
    ws.mkdir(parents=True, exist_ok=True)

    # Config
    (ws / "config").mkdir(parents=True, exist_ok=True)
    (ws / "config" / "tracks.json").write_text(json.dumps({
        "tracks": {
            "mexico": {
                "round_2026": 19,
                "display_name": "Mexico City Grand Prix",
                "event_name_patterns": ["mexico", "mexico_city"]
            },
            "monza": {
                "round_2026": 16,
                "display_name": "Italian Grand Prix",
                "event_name_patterns": ["monza", "italian"]
            }
        }
    }), encoding="utf-8")

    # Data structure
    (ws / "data" / "cache" / "2026").mkdir(parents=True, exist_ok=True)
    (ws / "data" / "raw" / "season" / "2026").mkdir(parents=True, exist_ok=True)
    (ws / "data" / "raw" / "season" / "2024").mkdir(parents=True, exist_ok=True)
    (ws / "data" / "raw" / "tracks" / "mexico" / "2026").mkdir(parents=True, exist_ok=True)
    (ws / "data" / "raw" / "tracks" / "mexico" / "history").mkdir(parents=True, exist_ok=True)
    (ws / "outputs" / "mexico").mkdir(parents=True, exist_ok=True)
    (ws / "logs").mkdir(parents=True, exist_ok=True)

    audit_logger = DeletionAuditLogger(root_dir=ws, log_file=ws / "logs" / "deletion_audit.jsonl")
    dm = DataManager(root_dir=ws, audit_logger=audit_logger)
    return ws, dm


def test_multiple_cache_files_nested_and_empty_dirs(isolated_env):
    """Test 1: Cache discovery over multiple files, nested subdirectories, and empty dirs."""
    ws, dm = isolated_env
    c_base = ws / "data" / "cache" / "2026" / "2026-10-25_Mexico"
    nested_dir = c_base / "telemetry" / "laps"
    nested_dir.mkdir(parents=True, exist_ok=True)
    empty_dir = c_base / "empty_session"
    empty_dir.mkdir(parents=True, exist_ok=True)

    f1 = c_base / "session.ff1"
    f2 = nested_dir / "lap_times.ff1"
    f3 = nested_dir / "car_data.ff1"
    f1.write_text("s1", encoding="utf-8")
    f2.write_text("l1", encoding="utf-8")
    f3.write_text("c1", encoding="utf-8")

    gp_data = dm.discover_grand_prix_data(track_key="mexico", year=2026)
    cached = gp_data["cached_data"]

    assert len(cached) == 3
    assert f1.resolve() in [p.resolve() for p in cached]
    assert f2.resolve() in [p.resolve() for p in cached]
    assert f3.resolve() in [p.resolve() for p in cached]
    # Ensure zero duplicates
    assert len(cached) == len(set(cached))


def test_duplicate_discovery_prevention(isolated_env):
    """Test 2: Verifies each file is discovered exactly once without duplicates."""
    ws, dm = isolated_env
    c_dir = ws / "data" / "cache" / "2026" / "2026_mexico_event"
    c_dir.mkdir(parents=True, exist_ok=True)
    f = c_dir / "telemetry.ff1"
    f.write_text("data", encoding="utf-8")

    gp_data = dm.discover_grand_prix_data(track_key="mexico", year=2026)
    cached = gp_data["cached_data"]

    assert len(cached) == 1
    assert cached[0].resolve() == f.resolve()


def test_historical_protected_by_default(isolated_env):
    """Test 3: Historical datasets are strictly protected by default."""
    ws, dm = isolated_env
    h_season = ws / "data" / "raw" / "season" / "2024" / "2024_round_1_race.csv"
    h_season.write_text("hist", encoding="utf-8")

    h_track = ws / "data" / "raw" / "tracks" / "mexico" / "history" / "mexico_2022_race.csv"
    h_track.write_text("hist_track", encoding="utf-8")

    # Default: protected
    prot1, r1 = dm.is_protected(h_season, include_historical=False)
    assert prot1 is True
    assert "historical" in r1.lower()

    prot2, r2 = dm.is_protected(h_track, include_historical=False)
    assert prot2 is True
    assert "historical" in r2.lower()

    # Explicit opt-in: eligible
    prot1_opt, _ = dm.is_protected(h_season, include_historical=True)
    assert prot1_opt is False

    prot2_opt, _ = dm.is_protected(h_track, include_historical=True)
    assert prot2_opt is False


def test_explicit_historical_deletion_opt_in(isolated_env):
    """Test 4: Explicit historical opt-in allows historical files into preview and deletion."""
    ws, dm = isolated_env
    h_file = ws / "data" / "raw" / "season" / "2024" / "2024_round_1_race.csv"
    h_file.write_text("hist_data", encoding="utf-8")

    # Preview with include_historical=False -> excluded
    prev_no = dm.build_deletion_preview([h_file], operation_name="Delete", include_historical=False)
    assert prev_no["file_count"] == 0
    assert prev_no["excluded_count"] == 1

    # Preview with include_historical=True -> targeted
    prev_yes = dm.build_deletion_preview([h_file], operation_name="Delete", include_historical=True)
    assert prev_yes["file_count"] == 1
    assert prev_yes["excluded_count"] == 0
    assert prev_yes["required_confirmation_phrase"] == "DELETE HISTORICAL DATA"

    # Confirmed execution deletes historical file
    res = dm.execute_deletion_plan(
        prev_yes,
        confirmed=True,
        confirmation_phrase="DELETE HISTORICAL DATA"
    )
    assert res["status"] == "COMPLETED"
    assert res["files_deleted_count"] == 1
    assert not h_file.exists()


def test_unsafe_path_rejection_discover_prediction_outputs(isolated_env):
    """Test 5: discover_prediction_outputs strictly rejects path traversal attempts."""
    ws, dm = isolated_env
    traversal_payloads = [
        "../data/raw/season/2026",
        "..\\data\\raw\\season\\2026",
        "../../src",
        "..\\..",
        "/etc/passwd",
        "C:\\Windows\\System32",
        "mexico/../monza",
    ]
    for payload in traversal_payloads:
        with pytest.raises(ValueError) as excinfo:
            dm.discover_prediction_outputs(track_key=payload)
        assert "unsafe" in str(excinfo.value).lower() or "invalid" in str(excinfo.value).lower()


def test_unsafe_path_rejection_discover_grand_prix_data(isolated_env):
    """Test 6: discover_grand_prix_data strictly rejects path traversal attempts."""
    ws, dm = isolated_env
    traversal_payloads = [
        "../season",
        "..\\tracks",
        "../../config",
        "/var/log",
        "D:\\data",
    ]
    for payload in traversal_payloads:
        with pytest.raises(ValueError) as excinfo:
            dm.discover_grand_prix_data(track=payload)
        assert "unsafe" in str(excinfo.value).lower() or "invalid" in str(excinfo.value).lower()


def test_unknown_circuit_key_handling(isolated_env):
    """Test 7: Nonexistent or unknown circuit identifiers return empty cleanly without errors."""
    ws, dm = isolated_env
    res = dm.discover_prediction_outputs(track_key="atlantis")
    assert res == []

    gp_res = dm.discover_grand_prix_data(track_key="atlantis", year=2026)
    assert gp_res["session_data"] == []
    assert gp_res["track_data"] == []
    assert gp_res["prediction_outputs"] == []
    assert gp_res["cached_data"] == []


def test_symlink_escape_rejection(isolated_env, tmp_path: Path):
    """Test 8: Symlinks pointing outside project root are rejected."""
    ws, dm = isolated_env
    outside_file = tmp_path / "outside_target.csv"
    outside_file.write_text("sensitive outside data", encoding="utf-8")

    link_path = ws / "outputs" / "mexico" / "outside_link.csv"
    try:
        os.symlink(outside_file, link_path)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation not supported in this environment.")

    assert link_path.is_symlink()

    # validate_path must raise ValueError
    with pytest.raises(ValueError) as excinfo:
        dm.validate_path(link_path)
    assert "outside project root" in str(excinfo.value).lower()

    # is_protected must treat it as protected
    prot, reason = dm.is_protected(link_path)
    assert prot is True


def test_preview_to_execution_path_tampering(isolated_env):
    """Test 9: Tampering with preview target_files rejects unauthorized un-previewed files."""
    ws, dm = isolated_env

    # Legitimate previewed file
    legit_file = ws / "outputs" / "mexico" / "mexico_2026_after_qualifying_predictions.csv"
    legit_file.write_text("legit", encoding="utf-8")

    # Sensitive file not part of preview
    unpreviewed_file = ws / "outputs" / "mexico" / "mexico_2026_after_qualifying_report.md"
    unpreviewed_file.write_text("unpreviewed sensitive", encoding="utf-8")

    preview = dm.build_deletion_preview([legit_file], operation_name="Delete Legit File")
    assert preview["file_count"] == 1
    assert preview["target_paths"] == ["outputs/mexico/mexico_2026_after_qualifying_predictions.csv"]

    # Tampering: attacker injects unpreviewed_file into target_files
    preview["target_files"].append(unpreviewed_file)

    res = dm.execute_deletion_plan(preview, confirmed=True)

    # Legit file is deleted
    assert not legit_file.exists()

    # Tampered file MUST NOT be deleted
    assert unpreviewed_file.exists()
    assert res["status"] == "PARTIAL_FAILURE"
    assert len(res["failed_deletions"]) == 1
    assert "not included in approved deletion preview" in res["failed_deletions"][0]["error"].lower()
