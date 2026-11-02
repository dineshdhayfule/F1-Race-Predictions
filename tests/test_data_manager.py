"""
Comprehensive Unit & Safety Tests for DataManager and DeletionAuditLogger
========================================================================
Runs strictly against isolated synthetic temporary directory fixtures (tmp_path).
NEVER touches or modifies real project data.
"""

import json
import os
from pathlib import Path
import pytest

from src.data.data_manager import DataManager
from src.data.deletion_audit import DeletionAuditLogger


@pytest.fixture
def synthetic_workspace(tmp_path: Path):
    """
    Creates an isolated synthetic workspace mimicking the F1 project structure:
    - config/
    - src/
    - tests/
    - logs/
    - data/cache/
    - data/raw/season/2024, 2026
    - data/raw/tracks/mexico/2026, data/raw/tracks/singapore/2022
    - data/grids/2026
    - outputs/mexico/, outputs/sao_paulo/
    """
    ws = tmp_path / "f1_synthetic_workspace"
    ws.mkdir(parents=True, exist_ok=True)

    # 1. Protected directories
    (ws / "src").mkdir(parents=True, exist_ok=True)
    (ws / "src" / "main_code.py").write_text("print('core')", encoding="utf-8")

    (ws / "config").mkdir(parents=True, exist_ok=True)
    (ws / "config" / "tracks.json").write_text(json.dumps({
        "tracks": {
            "mexico": {"round_2026": 19, "display_name": "Mexico City Grand Prix"},
            "sao_paulo": {"round_2026": 20, "display_name": "São Paulo Grand Prix"}
        }
    }), encoding="utf-8")

    (ws / "tests").mkdir(parents=True, exist_ok=True)
    (ws / "tests" / "test_fake.py").write_text("def test_ok(): pass", encoding="utf-8")

    (ws / "README.md").write_text("# Project Docs", encoding="utf-8")
    (ws / ".gitignore").write_text("*.tmp", encoding="utf-8")

    # 2. Cache data
    (ws / "data" / "cache" / "2026" / "2026-11-01_Mexico").mkdir(parents=True, exist_ok=True)
    (ws / "data" / "cache" / "2026" / "2026-11-01_Mexico" / "cache_file1.ff1").write_text("cache1", encoding="utf-8")
    (ws / "data" / "cache" / "2026" / "2026-11-01_Mexico" / "cache_file2.ff1").write_text("cache2", encoding="utf-8")

    # 3. Collected datasets (2026 current season)
    s2026 = ws / "data" / "raw" / "season" / "2026"
    s2026.mkdir(parents=True, exist_ok=True)
    (s2026 / "2026_round_19_fp1.csv").write_text("fp1 data", encoding="utf-8")
    (s2026 / "2026_round_19_qualifying.csv").write_text("quali data", encoding="utf-8")
    (s2026 / "2026_round_19_race.csv").write_text("race data", encoding="utf-8")
    (s2026 / "2026_round_20_fp1.csv").write_text("fp1 data round 20", encoding="utf-8")

    # 4. Historical training datasets (Protected by default!)
    s2024 = ws / "data" / "raw" / "season" / "2024"
    s2024.mkdir(parents=True, exist_ok=True)
    (s2024 / "2024_round_1_race.csv").write_text("historical race 2024", encoding="utf-8")

    t_hist = ws / "data" / "raw" / "tracks" / "singapore" / "2022"
    t_hist.mkdir(parents=True, exist_ok=True)
    (t_hist / "singapore_2022_race.csv").write_text("historical singapore 2022", encoding="utf-8")

    # 5. Protected official manifests
    (ws / "data" / "grids" / "2026").mkdir(parents=True, exist_ok=True)
    (ws / "data" / "grids" / "2026" / "mexico_manifest.json").write_text("{}", encoding="utf-8")

    # 6. Prediction outputs
    out_mex = ws / "outputs" / "mexico"
    out_mex.mkdir(parents=True, exist_ok=True)
    (out_mex / "mexico_2026_pre_fp1_predictions.csv").write_text("preds", encoding="utf-8")
    (out_mex / "mexico_2026_pre_fp1_report.md").write_text("# report", encoding="utf-8")

    out_sp = ws / "outputs" / "sao_paulo"
    out_sp.mkdir(parents=True, exist_ok=True)
    (out_sp / "sao_paulo_2026_pre_fp1_predictions.csv").write_text("sp preds", encoding="utf-8")

    (ws / "outputs" / "EXECUTIVE_REPORT_F1_PREDICTION_ENGINE.md").write_text("# Exec Report", encoding="utf-8")

    # 7. Audit log
    audit_file = ws / "logs" / "deletion_audit.jsonl"
    audit_logger = DeletionAuditLogger(root_dir=ws, log_file=audit_file)

    manager = DataManager(root_dir=ws, audit_logger=audit_logger)
    return ws, manager, audit_file


# =============================================================================
# 1. SAFETY & PROTECTION TESTS
# =============================================================================

def test_project_root_boundary_and_traversal(synthetic_workspace):
    """Verify rejection of paths outside root, path traversal, and targeting root itself."""
    ws, manager, _ = synthetic_workspace

    # Target is root itself
    with pytest.raises(ValueError, match="cannot be project root"):
        manager.validate_path(ws)

    # Path traversal outside root
    outside_target = ws / ".." / "external_file.txt"
    with pytest.raises(ValueError, match="resolves outside project root"):
        manager.validate_path(outside_target)


def test_protected_files_exclusion(synthetic_workspace):
    """Verify source code, config, tests, documentation, and historical data are protected."""
    ws, manager, _ = synthetic_workspace

    # Python source files protected
    prot, reason = manager.is_protected(ws / "src" / "main_code.py")
    assert prot is True
    assert "protected" in reason.lower()

    # Config protected
    prot, _ = manager.is_protected(ws / "config" / "tracks.json")
    assert prot is True

    # Tests protected
    prot, _ = manager.is_protected(ws / "tests" / "test_fake.py")
    assert prot is True

    # Docs protected
    prot, _ = manager.is_protected(ws / "README.md")
    assert prot is True

    # Executive doc in outputs protected
    prot, _ = manager.is_protected(ws / "outputs" / "EXECUTIVE_REPORT_F1_PREDICTION_ENGINE.md")
    assert prot is True

    # Audit log protected
    prot, _ = manager.is_protected(ws / "logs" / "deletion_audit.jsonl")
    assert prot is True

    # Official FIA grid manifest protected
    prot, _ = manager.is_protected(ws / "data" / "grids" / "2026" / "mexico_manifest.json")
    assert prot is True

    # Historical data protected by default
    prot_hist, _ = manager.is_protected(ws / "data" / "raw" / "season" / "2024" / "2024_round_1_race.csv")
    assert prot_hist is True

    # Historical data eligible ONLY when explicitly opted-in
    prot_optin, _ = manager.is_protected(
        ws / "data" / "raw" / "season" / "2024" / "2024_round_1_race.csv",
        include_historical=True
    )
    assert prot_optin is False


# =============================================================================
# 2. DISCOVERY TESTS
# =============================================================================

def test_cache_discovery(synthetic_workspace):
    """Verify discovery of cache data in data/cache."""
    ws, manager, _ = synthetic_workspace
    caches = manager.discover_cache_data()
    assert len(caches) == 2
    assert all("cache" in p.as_posix() for p in caches)


def test_prediction_output_discovery(synthetic_workspace):
    """Verify discovery of generated prediction files, excluding root docs."""
    ws, manager, _ = synthetic_workspace
    outputs = manager.discover_prediction_outputs()
    assert len(outputs) == 3
    # Executive doc must NOT be discovered
    assert all("EXECUTIVE_REPORT" not in p.name for p in outputs)


def test_collected_datasets_discovery(synthetic_workspace):
    """Verify categorized discovery and historical protection by default."""
    ws, manager, _ = synthetic_workspace

    # Default: historical 2024 data excluded
    discovered = manager.discover_collected_datasets(include_historical=False)
    all_files = [f for flist in discovered.values() for f in flist]
    assert all("2024" not in f.as_posix() for f in all_files)

    # Opt-in: historical 2024 included
    discovered_hist = manager.discover_collected_datasets(include_historical=True)
    all_hist_files = [f for flist in discovered_hist.values() for f in flist]
    assert any("2024" in f.as_posix() for f in all_hist_files)


def test_grand_prix_specific_discovery(synthetic_workspace):
    """Verify Grand Prix discovery finds only selected event artifacts."""
    ws, manager, _ = synthetic_workspace

    # Discover Mexico 2026 (Round 19)
    gp_data = manager.discover_grand_prix_data("mexico", year=2026, round_num=19)
    all_mex_files = [f for flist in gp_data.values() for f in flist]

    # Mexico files found
    assert len(all_mex_files) >= 4
    # Round 20 (São Paulo) files MUST NOT be included!
    assert all("round_20" not in f.name for f in all_mex_files)
    assert all("sao_paulo" not in f.as_posix() for f in all_mex_files)


# =============================================================================
# 3. CONFIRMATION & EXECUTION TESTS
# =============================================================================

def test_cancellation_leaves_files_untouched(synthetic_workspace):
    """Verify cancellation leaves all candidate files completely untouched."""
    ws, manager, _ = synthetic_workspace

    caches = manager.discover_cache_data()
    assert len(caches) > 0

    preview = manager.build_deletion_preview(caches, "Delete Cached Data")
    res = manager.execute_deletion_plan(preview, confirmed=False)

    assert res["status"] == "CANCELLED"
    assert res["files_deleted_count"] == 0

    # Every cache file must still exist
    for c in caches:
        assert c.exists()


def test_confirmation_phrase_mismatch_prevents_deletion(synthetic_workspace):
    """Verify operation is rejected if confirmation phrase mismatch occurs."""
    ws, manager, _ = synthetic_workspace

    all_data = manager.discover_all_project_generated_data()
    flat_targets = [f for flist in all_data.values() for f in flist]

    preview = manager.build_deletion_preview(flat_targets, "Delete all project-generated data")
    assert preview["required_confirmation_phrase"] == "DELETE ALL GENERATED DATA"

    # Supply wrong phrase
    res = manager.execute_deletion_plan(preview, confirmed=True, confirmation_phrase="WRONG PHRASE")
    assert res["status"] == "REJECTED"
    assert res["files_deleted_count"] == 0

    # Files must still exist
    for f in flat_targets:
        assert f.exists()


def test_confirmed_deletion_executes_and_logs(synthetic_workspace):
    """Verify confirmed deletion removes target files, preserves unselected, and appends audit log."""
    ws, manager, audit_file = synthetic_workspace

    outputs_before = manager.discover_prediction_outputs()
    assert len(outputs_before) == 3

    # Delete only Mexico outputs
    mex_outputs = manager.discover_prediction_outputs(track_key="mexico")
    assert len(mex_outputs) == 2

    preview = manager.build_deletion_preview(mex_outputs, "Delete Prediction Outputs")
    res = manager.execute_deletion_plan(preview, confirmed=True)

    assert res["status"] == "COMPLETED"
    assert res["files_deleted_count"] == 2
    assert res["storage_reclaimed_bytes"] > 0

    # Mexico output files must be deleted
    for f in mex_outputs:
        assert not f.exists()

    # São Paulo output must still exist untouched
    sp_out = ws / "outputs" / "sao_paulo" / "sao_paulo_2026_pre_fp1_predictions.csv"
    assert sp_out.exists()

    # Audit log was written and contains structured entry
    assert audit_file.exists()
    entries = manager.audit_logger.get_audit_history()
    assert len(entries) >= 1
    last = entries[-1]
    assert last["status"] == "COMPLETED"
    assert last["files_deleted_count"] == 2
    assert "outputs/mexico" in last["files_deleted"][0]


# =============================================================================
# 4. CLI COMMAND & INTEGRATION TESTS
# =============================================================================

def test_cli_clean_preview(synthetic_workspace, monkeypatch, capsys):
    """Verify CLI clean with --preview prints preview and deletes no files."""
    ws, manager, _ = synthetic_workspace
    import main as main_module

    monkeypatch.setattr(main_module, "ROOT", ws)

    args = type("Args", (), {
        "command": "clean",
        "clean_all": False,
        "cache": True,
        "outputs": False,
        "collected": False,
        "track": None,
        "year": None,
        "round": None,
        "categories": None,
        "include_historical": False,
        "preview": True,
        "force": False,
        "audit_log": False,
    })()

    main_module.cmd_clean(args)
    captured = capsys.readouterr().out

    assert "DELETION PLAN PREVIEW" in captured
    assert "Total files targeted : 2" in captured
    assert "[PREVIEW MODE] Dry run complete. No files were deleted." in captured

    # Files must still exist
    caches = manager.discover_cache_data()
    assert len(caches) == 2


def test_cli_clean_outputs_with_force(synthetic_workspace, monkeypatch, capsys):
    """Verify CLI clean with --outputs --force executes deletion and logs audit."""
    ws, manager, audit_file = synthetic_workspace
    import main as main_module

    monkeypatch.setattr(main_module, "ROOT", ws)

    args = type("Args", (), {
        "command": "clean",
        "clean_all": False,
        "cache": False,
        "outputs": True,
        "collected": False,
        "track": "mexico",
        "year": None,
        "round": None,
        "categories": None,
        "include_historical": False,
        "preview": False,
        "force": True,
        "audit_log": False,
    })()

    main_module.cmd_clean(args)
    captured = capsys.readouterr().out

    assert "DELETION RESULT: COMPLETED" in captured
    assert "Files deleted        : 2" in captured

    # Outputs for Mexico gone
    mex_outputs = manager.discover_prediction_outputs(track_key="mexico")
    assert len(mex_outputs) == 0


def test_cli_clean_audit_log_display(synthetic_workspace, monkeypatch, capsys):
    """Verify CLI clean --audit-log displays logged operations."""
    ws, manager, audit_file = synthetic_workspace
    import main as main_module

    monkeypatch.setattr(main_module, "ROOT", ws)

    # Log a dummy operation first
    manager.audit_logger.log_operation(
        operation_type="Test Cleanup",
        files_identified=["outputs/dummy.csv"],
        files_deleted=["outputs/dummy.csv"],
        files_excluded=[],
        failed_deletions=[],
        storage_reclaimed_bytes=1024,
        confirmed=True,
        status="COMPLETED"
    )

    args = type("Args", (), {
        "command": "clean",
        "clean_all": False,
        "cache": False,
        "outputs": False,
        "collected": False,
        "track": None,
        "year": None,
        "round": None,
        "categories": None,
        "include_historical": False,
        "preview": False,
        "force": False,
        "audit_log": True,
    })()

    main_module.cmd_clean(args)
    captured = capsys.readouterr().out

    assert "DELETION AUDIT TRAIL" in captured
    assert "Test Cleanup" in captured
    assert "COMPLETED" in captured


def test_interactive_menu_has_data_manager(synthetic_workspace):
    """Verify InteractiveMenu initializes DataManager without side effects."""
    ws, _, _ = synthetic_workspace
    from src.cli.menu import InteractiveMenu

    menu = InteractiveMenu(root_dir=ws)
    assert hasattr(menu, "data_manager")
    assert menu.data_manager.root_dir == ws

