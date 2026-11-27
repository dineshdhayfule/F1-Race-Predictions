import json
from pathlib import Path

from apexgrid.data.data_manager import DataManager


def test_synthetic_workspace_discovery(tmp_path):
    ws = tmp_path / "repo"
    (ws / "data" / "cache").mkdir(parents=True)
    (ws / "outputs").mkdir(parents=True)
    (ws / "config").mkdir(parents=True)
    (ws / "config" / "tracks.json").write_text(json.dumps({"tracks": {}}), encoding="utf-8")
    manager = DataManager(root_dir=ws)
    cache = manager.discover_cache_data()
    assert isinstance(cache, list)


def test_historical_data_is_not_selected_by_default(tmp_path):
    manager = DataManager(root_dir=tmp_path)
    result = manager.discover_collected_datasets(include_historical=False)
    assert isinstance(result, dict)


def test_preview_does_not_delete_files(tmp_path):
    manager = DataManager(root_dir=tmp_path)
    plan = manager.build_deletion_preview([], operation_name="Preview", include_historical=False)
    assert plan["file_count"] == 0


def test_delete_preview_requires_historical_opt_in(tmp_path):
    manager = DataManager(root_dir=tmp_path)
    plan = manager.build_deletion_preview([], operation_name="Preview", include_historical=False)
    assert plan["historical_included"] is False
