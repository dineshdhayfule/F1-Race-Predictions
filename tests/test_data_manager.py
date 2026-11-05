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


# =============================================================================
# 5. REGRESSION TESTS: CACHE CONFIRMATION & ISOLATION
# =============================================================================

def test_regression_test_a_all_season_cache_deletion(synthetic_workspace):
    """Test A: All-season cache deletion requires no typed phrase and deletes with confirmed=True."""
    ws, manager, _ = synthetic_workspace

    caches = manager.discover_cache_data()
    assert len(caches) >= 2

    # In menu, op_label was "Delete Cached Data (All Seasons)"
    preview = manager.build_deletion_preview(
        caches,
        operation_name="Delete Cached Data (All Seasons)",
        operation_type="cache"
    )

    # Must NOT require typing "DELETE ALL GENERATED DATA"
    assert preview["required_confirmation_phrase"] is None
    assert preview["categories"] == {"Cache data": len(caches)}
    assert preview["file_count"] == len(caches)

    # Simple confirmation (Yes/No) executes successfully
    res = manager.execute_deletion_plan(preview, confirmed=True)
    assert res["status"] == "COMPLETED"
    assert res["files_deleted_count"] == len(caches)

    # All cache files are gone
    for f in caches:
        assert not f.exists()

    # Core source and config remain
    assert (ws / "src" / "main_code.py").exists()
    assert (ws / "config" / "tracks.json").exists()


def test_regression_test_b_single_season_cache_deletion(synthetic_workspace):
    """Test B: Single-season cache deletion only targets selected season; preserves other seasons."""
    ws, manager, _ = synthetic_workspace

    # Add 2024 cache file alongside 2026
    c2024 = ws / "data" / "cache" / "2024" / "session_2024.ff1"
    c2024.parent.mkdir(parents=True, exist_ok=True)
    c2024.write_text("cache 2024", encoding="utf-8")

    caches_2026 = manager.discover_cache_data(year=2026)
    assert len(caches_2026) == 2
    assert all("2026" in p.as_posix() for p in caches_2026)

    preview = manager.build_deletion_preview(
        caches_2026,
        operation_name="Delete Cached Data (2026)",
        operation_type="cache"
    )

    assert preview["required_confirmation_phrase"] is None
    assert preview["file_count"] == 2

    res = manager.execute_deletion_plan(preview, confirmed=True)
    assert res["status"] == "COMPLETED"
    assert res["files_deleted_count"] == 2

    # 2026 cache files removed
    for f in caches_2026:
        assert not f.exists()

    # 2024 cache file preserved
    assert c2024.exists()


def test_regression_test_c_full_project_generated_data_deletion(synthetic_workspace):
    """Test C: Full project deletion requires 'DELETE ALL GENERATED DATA' and rejects missing/wrong phrase."""
    ws, manager, _ = synthetic_workspace

    all_data = manager.discover_all_project_generated_data()
    flat_targets = [f for flist in all_data.values() for f in flist]
    assert len(flat_targets) > 0

    preview = manager.build_deletion_preview(
        flat_targets,
        operation_name="Delete all project-generated data",
        operation_type="all_generated_data"
    )

    # Must require exact confirmation phrase
    assert preview["required_confirmation_phrase"] == "DELETE ALL GENERATED DATA"

    # 1. Rejection without confirmation phrase
    res_none = manager.execute_deletion_plan(preview, confirmed=True, confirmation_phrase=None)
    assert res_none["status"] == "REJECTED"
    assert res_none["files_deleted_count"] == 0

    # 2. Rejection with incorrect confirmation phrase
    res_wrong = manager.execute_deletion_plan(preview, confirmed=True, confirmation_phrase="yes")
    assert res_wrong["status"] == "REJECTED"
    assert res_wrong["files_deleted_count"] == 0

    # Files must still exist
    for f in flat_targets:
        assert f.exists()

    # 3. Successful execution with correct phrase
    res_ok = manager.execute_deletion_plan(preview, confirmed=True, confirmation_phrase="DELETE ALL GENERATED DATA")
    assert res_ok["status"] == "COMPLETED"
    assert res_ok["files_deleted_count"] == len(flat_targets)


def test_regression_test_d_historical_data_protection(synthetic_workspace):
    """Test D: Historical training data is protected unless opted-in, which requires 'DELETE HISTORICAL DATA'."""
    ws, manager, _ = synthetic_workspace

    # Default discovery excludes historical seasons (2024)
    default_datasets = manager.discover_collected_datasets(include_historical=False)
    flat_default = [f for flist in default_datasets.values() for f in flist]
    assert all("2024" not in f.as_posix() for f in flat_default)

    # Opted-in discovery includes historical
    hist_datasets = manager.discover_collected_datasets(include_historical=True)
    flat_hist = [f for flist in hist_datasets.values() for f in flist]
    assert any("2024" in f.as_posix() for f in flat_hist)

    preview = manager.build_deletion_preview(
        flat_hist,
        operation_name="Delete Collected Datasets (All Categories)",
        include_historical=True,
        operation_type="collected"
    )

    assert preview["required_confirmation_phrase"] == "DELETE HISTORICAL DATA"

    # Rejection without phrase
    res_fail = manager.execute_deletion_plan(preview, confirmed=True)
    assert res_fail["status"] == "REJECTED"

    # Execution with exact phrase
    res_ok = manager.execute_deletion_plan(preview, confirmed=True, confirmation_phrase="DELETE HISTORICAL DATA")
    assert res_ok["status"] == "COMPLETED"


def test_regression_test_e_cache_cleanup_isolation(synthetic_workspace):
    """Test E: Scratch CSVs, raw datasets, prediction outputs, and audit logs are untouched by cache deletion."""
    ws, manager, audit_file = synthetic_workspace

    # Populate scratch directory with CSVs and temporary files
    scratch_dir = ws / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    scratch_csv1 = scratch_dir / "similarity_drivers.csv"
    scratch_csv2 = scratch_dir / "similarity_pairwise.csv"
    scratch_tmp = scratch_dir / "temp_cache.tmp"
    scratch_csv1.write_text("driver_id,score\n1,0.95", encoding="utf-8")
    scratch_csv2.write_text("pair_id,sim\n1_2,0.88", encoding="utf-8")
    scratch_tmp.write_text("temp data", encoding="utf-8")

    # Discover cache data
    cache_files = manager.discover_cache_data()

    # Scratch CSVs must NOT be in discovered cache files
    assert scratch_csv1 not in cache_files
    assert scratch_csv2 not in cache_files

    # Preview cache deletion
    preview = manager.build_deletion_preview(
        cache_files,
        operation_name="Delete Cached Data (All Seasons)",
        operation_type="cache"
    )

    # All files in preview must be categorized as 'Cache data'
    assert list(preview["categories"].keys()) == ["Cache data"]
    assert preview["required_confirmation_phrase"] is None

    # Execute deletion
    res = manager.execute_deletion_plan(preview, confirmed=True)
    assert res["status"] == "COMPLETED"

    # Verify scratch CSVs still exist completely untouched
    assert scratch_csv1.exists()
    assert scratch_csv2.exists()
    assert scratch_csv1.read_text(encoding="utf-8") == "driver_id,score\n1,0.95"

    # Verify raw datasets, outputs, source code, and audit logs are untouched
    assert (ws / "data" / "raw" / "season" / "2026" / "2026_round_19_race.csv").exists()
    assert (ws / "data" / "raw" / "season" / "2024" / "2024_round_1_race.csv").exists()
    assert (ws / "outputs" / "mexico" / "mexico_2026_pre_fp1_predictions.csv").exists()
    assert (ws / "src" / "main_code.py").exists()
    assert audit_file.exists()


def test_regression_test_f_cancellation_preserves_everything(synthetic_workspace):
    """Test F: Choosing confirmed=False cancels the operation with status CANCELLED and 0 deletions."""
    ws, manager, _ = synthetic_workspace

    cache_files = manager.discover_cache_data()
    assert len(cache_files) > 0

    preview = manager.build_deletion_preview(
        cache_files,
        operation_name="Delete Cached Data (All Seasons)",
        operation_type="cache"
    )

    res = manager.execute_deletion_plan(preview, confirmed=False)
    assert res["status"] == "CANCELLED"
    assert res["files_deleted_count"] == 0

    for f in cache_files:
        assert f.exists()


