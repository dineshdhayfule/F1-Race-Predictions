"""Tests for dynamic Step 3 stage availability tracking and Sprint/Standard branching."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from apexgrid.cli.menu import InteractiveMenu
from apexgrid.utils.network import is_internet_available


def test_standard_vs_sprint_stage_metadata_and_slugs(tmp_path):
    menu = InteractiveMenu(root_dir=tmp_path)
    past_dt = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()

    with patch.object(menu, "_check_session_dataset_availability", return_value="YES") as mock_avail:
        std_event = SimpleNamespace(
            event_name="Italian Grand Prix",
            circuit_name="Autodromo Nazionale Monza",
            event_date=past_dt,
            sessions={"Practice 1": past_dt, "Practice 2": past_dt, "Practice 3": past_dt, "Qualifying": past_dt, "Race": past_dt},
        )
        std_stages = menu._get_stage_availability_metadata(std_event, 2026, 13, "monza", is_sprint=False, is_online=True)
        std_keys = [s["key"] for s in std_stages]
        assert "after_fp2" in std_keys
        assert "after_fp3" in std_keys
        assert "after_sprint_qualifying" not in std_keys

        sprint_event = SimpleNamespace(
            event_name="Chinese Grand Prix",
            circuit_name="Shanghai International Circuit",
            event_date=past_dt,
            sessions={"Practice 1": past_dt, "Sprint Qualifying": past_dt, "Sprint": past_dt, "Qualifying": past_dt, "Race": past_dt},
        )
        sprint_stages = menu._get_stage_availability_metadata(sprint_event, 2026, 2, "china", is_sprint=True, is_online=True)
        sprint_map = {s["key"]: s for s in sprint_stages}
        assert "after_fp2" not in sprint_map
        assert "after_fp3" not in sprint_map
        assert sprint_map["after_sprint_qualifying"]["session_slug"] == "sprint_qualifying"
        sprint_race_key = "after_sprint_race" if "after_sprint_race" in sprint_map else "after_sprint"
        assert sprint_map[sprint_race_key]["session_slug"] == "sprint"
        assert mock_avail.call_count > 0


def test_pre_fp1_always_selectable_across_offline_and_future(tmp_path):
    menu = InteractiveMenu(root_dir=tmp_path)
    future_dt = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
    future_event = SimpleNamespace(
        event_name="Qatar Grand Prix",
        circuit_name="Lusail International Circuit",
        event_date=future_dt,
        sessions={"Practice 1": future_dt, "Qualifying": future_dt, "Race": future_dt},
    )

    with patch.object(menu, "_check_session_dataset_availability", return_value="FUTURE"):
        for online in (True, False):
            stages = menu._get_stage_availability_metadata(future_event, 2026, 22, "qatar", is_sprint=False, is_online=online)
            by_key = {s["key"]: s for s in stages}
            assert by_key["pre_fp1"]["selectable"] is True
            assert "[READY]" in by_key["pre_fp1"]["badge"]
            assert by_key["after_fp1"]["selectable"] is False
            assert "[FUTURE]" in by_key["after_fp1"]["badge"]
            assert "Session has not started yet" in by_key["after_fp1"]["block_reason"]


def test_past_missing_session_online_vs_offline_guard(tmp_path):
    menu = InteractiveMenu(root_dir=tmp_path)
    past_dt = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    past_event = SimpleNamespace(
        event_name="Qatar Grand Prix",
        circuit_name="Lusail International Circuit",
        event_date=past_dt,
        sessions={"Practice 1": past_dt, "Qualifying": past_dt, "Race": past_dt},
    )

    with patch.object(menu, "_check_session_dataset_availability", return_value="NO"):
        online_stages = {s["key"]: s for s in menu._get_stage_availability_metadata(past_event, 2026, 22, "qatar", is_sprint=False, is_online=True)}
        assert online_stages["after_fp1"]["selectable"] is True
        assert "[ONLINE]" in online_stages["after_fp1"]["badge"]

        offline_stages = {s["key"]: s for s in menu._get_stage_availability_metadata(past_event, 2026, 22, "qatar", is_sprint=False, is_online=False)}
        assert offline_stages["after_fp1"]["selectable"] is False
        assert "[OFFLINE - NO DATA]" in offline_stages["after_fp1"]["badge"]
        assert "offline" in offline_stages["after_fp1"]["block_reason"].lower()


def test_is_internet_available_handles_offline():
    with patch("apexgrid.utils.network.socket.create_connection", side_effect=OSError("Network unreachable")):
        assert is_internet_available(timeout=0.1) is False
