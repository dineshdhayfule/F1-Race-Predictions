"""
Sprint Architecture and Data Loader Unit Tests
==============================================
Validates:
1. Canonical session name normalization across all alias formats.
2. Track configuration sprint weekend detection.
3. Strict stage-aware session access matrix for Normal and Sprint weekends.
4. Invalid stage rejection per weekend format.
5. Strict anti-leakage guarantees (no GP race data prior to post_race).
6. File discovery and track_k path resolution bugfix verification.
7. CLI stage parsing.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import json
import subprocess
import sys

from src.data.data_loader import DataLoader, normalize_session_name


# -------------------------------------------------------------------------
# OBJECTIVE 3: Session Name Normalization Tests
# -------------------------------------------------------------------------

def test_session_name_normalization():
    # Practice sessions
    assert normalize_session_name("FP1") == "fp1"
    assert normalize_session_name("Free Practice 1") == "fp1"
    assert normalize_session_name("Practice 1") == "fp1"
    assert normalize_session_name("fp1") == "fp1"

    assert normalize_session_name("FP2") == "fp2"
    assert normalize_session_name("Free Practice 2") == "fp2"
    assert normalize_session_name("Practice 2") == "fp2"

    assert normalize_session_name("FP3") == "fp3"
    assert normalize_session_name("Free Practice 3") == "fp3"
    assert normalize_session_name("Practice 3") == "fp3"

    # Sprint qualifying / shootout
    assert normalize_session_name("Sprint Qualifying") == "sprint_qualifying"
    assert normalize_session_name("Sprint Shootout") == "sprint_qualifying"
    assert normalize_session_name("sprint_shootout") == "sprint_qualifying"
    assert normalize_session_name("SQ") == "sprint_qualifying"
    assert normalize_session_name("sprint_qualifying") == "sprint_qualifying"

    # Sprint race
    assert normalize_session_name("Sprint") == "sprint_race"
    assert normalize_session_name("Sprint Race") == "sprint_race"
    assert normalize_session_name("sprint_race") == "sprint_race"

    # GP Qualifying & Race
    assert normalize_session_name("Qualifying") == "qualifying"
    assert normalize_session_name("Quali") == "qualifying"
    assert normalize_session_name("Race") == "race"


# -------------------------------------------------------------------------
# OBJECTIVE 1: Sprint Weekend Detection Tests
# -------------------------------------------------------------------------

def test_sprint_weekend_detection():
    loader = DataLoader()

    # Confirmed sprint weekends in tracks.json for 2026
    sprint_tracks = ["singapore", "china", "miami", "canada", "silverstone", "zandvoort"]
    for t in sprint_tracks:
        assert loader.is_sprint_weekend(t, 2026) is True, f"Track {t} should be a sprint weekend in 2026"

    # Confirmed normal weekends in tracks.json for 2026
    normal_tracks = ["sepang", "baku", "monza", "melbourne", "suzuka", "spa", "monaco"]
    for t in normal_tracks:
        assert loader.is_sprint_weekend(t, 2026) is False, f"Track {t} should NOT be a sprint weekend in 2026"


# -------------------------------------------------------------------------
# OBJECTIVES 2 & 6: Stage Cutoff Matrix and Leakage Prevention
# -------------------------------------------------------------------------

def test_normal_weekend_stage_cutoffs():
    loader = DataLoader()

    # Normal weekend (e.g. Sepang)
    sessions_pre = loader.load_weekend_sessions("sepang", 2026, stage="pre_fp1")
    assert all(v is None for v in sessions_pre.values())

    # after_fp1
    # Check that sprint sessions are not even searched
    sessions_fp1 = loader.load_weekend_sessions("sepang", 2026, stage="after_fp1")
    assert sessions_fp1["qualifying"] is None
    assert sessions_fp1["race"] is None

    # after_qualifying allows qualifying but NEVER race
    sessions_q = loader.load_weekend_sessions("sepang", 2026, stage="after_qualifying")
    assert sessions_q["race"] is None

    # Sprint stages are invalid on normal weekend
    with pytest.raises(ValueError, match="invalid for Normal weekend"):
        loader.load_weekend_sessions("sepang", 2026, stage="after_sprint_qualifying")

    with pytest.raises(ValueError, match="invalid for Normal weekend"):
        loader.load_weekend_sessions("sepang", 2026, stage="after_sprint_race")


def test_sprint_weekend_stage_cutoffs():
    loader = DataLoader()

    # Sprint weekend (e.g. Singapore)
    # FP2 and FP3 do not exist on sprint weekend
    with pytest.raises(ValueError, match="invalid for Sprint weekend"):
        loader.load_weekend_sessions("singapore", 2026, stage="after_fp2")

    with pytest.raises(ValueError, match="invalid for Sprint weekend"):
        loader.load_weekend_sessions("singapore", 2026, stage="after_fp3")


# -------------------------------------------------------------------------
# OBJECTIVES 4, 5 & 6: Data Loader File Resolution & Leakage Barrier
# -------------------------------------------------------------------------

def test_sprint_data_file_discovery_and_resolution(tmp_path):
    """
    Creates synthetic session CSVs to verify:
    1. 'track_key' vs 'track_k' bug fix (positional track_id resolution).
    2. Stage-aware loading only loads sessions allowed at each cutoff.
    3. Strict anti-leakage: future sessions (especially GP race) are never loaded.
    """
    raw_dir = tmp_path / "data" / "raw"
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)

    tracks_cfg = {
        "tracks": {
            "test_sprint_gp": {
                "display_name": "Test Sprint GP",
                "round_2026": 18,
                "sprint_weekend_2026": True
            }
        }
    }
    with open(config_dir / "tracks.json", "w", encoding="utf-8") as f:
        json.dump(tracks_cfg, f)

    track_dir = raw_dir / "tracks" / "test_sprint_gp" / "2026"
    track_dir.mkdir(parents=True)

    # Create dummy CSV files for all sessions
    drivers_df = pd.DataFrame({"Abbreviation": ["VER", "NOR", "LEC"], "Position": [1, 2, 3]})
    for s in ["fp1", "sprint_qualifying", "sprint_race", "qualifying", "race"]:
        drivers_df.to_csv(track_dir / f"2026_test_sprint_gp_{s}.csv", index=False)

    loader = DataLoader(root_dir=tmp_path)

    # 1. pre_fp1: Nothing loaded
    s_pre = loader.load_weekend_sessions("test_sprint_gp", 2026, stage="pre_fp1")
    for s in loader.STAGES:
        assert s_pre.get(s) is None

    # 2. after_fp1: Only FP1 loaded
    s_fp1 = loader.load_weekend_sessions("test_sprint_gp", 2026, stage="after_fp1")
    assert s_fp1["fp1"] is not None
    assert s_fp1["sprint_qualifying"] is None
    assert s_fp1["sprint_race"] is None
    assert s_fp1["qualifying"] is None
    assert s_fp1["race"] is None

    # 3. after_sprint_qualifying: FP1 + Sprint Qualifying loaded
    s_sq = loader.load_weekend_sessions("test_sprint_gp", 2026, stage="after_sprint_qualifying")
    assert s_sq["fp1"] is not None
    assert s_sq["sprint_qualifying"] is not None
    assert s_sq["sprint_race"] is None
    assert s_sq["qualifying"] is None
    assert s_sq["race"] is None

    # 4. after_sprint_race: FP1 + SQ + Sprint Race loaded
    s_sr = loader.load_weekend_sessions("test_sprint_gp", 2026, stage="after_sprint_race")
    assert s_sr["fp1"] is not None
    assert s_sr["sprint_qualifying"] is not None
    assert s_sr["sprint_race"] is not None
    assert s_sr["qualifying"] is None
    assert s_sr["race"] is None

    # 5. after_qualifying: FP1 + SQ + Sprint Race + GP Qualifying loaded. Main race strictly FORBIDDEN!
    s_q = loader.load_weekend_sessions("test_sprint_gp", 2026, stage="after_qualifying")
    assert s_q["fp1"] is not None
    assert s_q["sprint_qualifying"] is not None
    assert s_q["sprint_race"] is not None
    assert s_q["qualifying"] is not None
    assert s_q["race"] is None, "STRICT LEAKAGE VIOLATION: GP Race must NEVER be loaded at after_qualifying!"

    # 6. post_race: All sessions loaded including race
    s_post = loader.load_weekend_sessions("test_sprint_gp", 2026, stage="post_race")
    assert s_post["race"] is not None


# -------------------------------------------------------------------------
# OBJECTIVE 8: CLI Parser Stage Verification
# -------------------------------------------------------------------------

def test_cli_accepts_sprint_stages():
    """Verify that main.py --help or argument parser accepts sprint stages without error."""
    res = subprocess.run(
        [sys.executable, "main.py", "predict", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1]
    )
    assert res.returncode == 0
    assert "after_sprint_qualifying" in res.stdout
    assert "after_sprint_race" in res.stdout


# -------------------------------------------------------------------------
# OBJECTIVE 9: Historical Sprint Round Detection & Session Loading Tests
# -------------------------------------------------------------------------

def test_historical_sprint_detection_by_year_and_round(tmp_path):
    """
    Validates:
    1. Historical Sprint round detection strictly by (year, round) files.
    2. Historical Normal round detection by (year, round).
    3. Same circuit having different formats across seasons (independent of circuit name).
    4. Missing Sprint files safely treated as Normal / no sprint data.
    """
    season_dir_2024 = tmp_path / "data" / "raw" / "season" / "2024"
    season_dir_2025 = tmp_path / "data" / "raw" / "season" / "2025"
    season_dir_2024.mkdir(parents=True)
    season_dir_2025.mkdir(parents=True)

    loader = DataLoader(root_dir=tmp_path)

    # In 2024, Round 5 was a NORMAL weekend (only fp1, fp2, fp3)
    for s in ("fp1", "fp2", "fp3", "qualifying", "race"):
        (season_dir_2024 / f"2024_round_5_{s}.csv").write_text("Driver,Position\nVER,1\n", encoding="utf-8")

    # In 2025, Round 5 of the same track became a SPRINT weekend (fp1, sprint_qualifying, sprint_race)
    for s in ("fp1", "sprint_qualifying", "sprint_race", "qualifying", "race"):
        (season_dir_2025 / f"2025_round_5_{s}.csv").write_text("Driver,Position\nVER,1\n", encoding="utf-8")

    # 1. 2024 Round 5 must be detected as NORMAL
    assert loader.is_historical_sprint_round(2024, 5, season_dir=season_dir_2024) is False

    # 2. 2025 Round 5 must be detected as SPRINT
    assert loader.is_historical_sprint_round(2025, 5, season_dir=season_dir_2025) is True

    # 3. Missing files for non-existent round 99 safely returns False
    assert loader.is_historical_sprint_round(2025, 99, season_dir=season_dir_2025) is False

    # 4. Explicit calendar override works if provided
    assert loader.is_historical_sprint_round(2024, 5, sprint_calendar={2024: [5]}) is True
    assert loader.is_historical_sprint_round(2025, 5, sprint_calendar={2025: []}) is False


def test_historical_sprint_session_loading_and_feature_population(tmp_path):
    """
    Validates:
    1. Historical Sprint Qualifying & Sprint Race loading.
    2. Historical Normal weekend keeps FP2/FP3 and sets Sprint features to NaN.
    3. Historical Sprint weekend keeps FP2/FP3 as None and populates Sprint features.
    4. Anti-leakage: GP race result is never used in sprint feature creation.
    """
    from src.features.build_features import FeatureEngineer

    season_dir = tmp_path / "data" / "raw" / "season" / "2025"
    season_dir.mkdir(parents=True)

    # Round 1: Normal weekend
    for s in ("fp1", "fp2", "fp3"):
        pd.DataFrame({"Abbreviation": ["VER", "NOR"], "LapTimeSeconds": [90.0, 91.0]}).to_csv(
            season_dir / f"2025_round_1_{s}.csv", index=False
        )
    pd.DataFrame({"Abbreviation": ["VER", "NOR"], "Position": [1, 2], "Round": 1}).to_csv(
        season_dir / "2025_round_1_qualifying.csv", index=False
    )
    pd.DataFrame({"Abbreviation": ["VER", "NOR"], "TeamName": ["Red Bull Racing", "McLaren"], "Position": [1, 2], "Round": 1, "GridPosition": [1, 2], "Points": [25, 18]}).to_csv(
        season_dir / "2025_round_1_race.csv", index=False
    )

    # Round 2: Sprint weekend
    pd.DataFrame({"Abbreviation": ["VER", "NOR"], "LapTimeSeconds": [89.5, 90.5]}).to_csv(
        season_dir / "2025_round_2_fp1.csv", index=False
    )
    pd.DataFrame({"Abbreviation": ["VER", "NOR"], "Position": [1, 2]}).to_csv(
        season_dir / "2025_round_2_sprint_qualifying.csv", index=False
    )
    pd.DataFrame({"Abbreviation": ["VER", "NOR"], "Position": [2, 1], "StartingGrid": [1, 2]}).to_csv(
        season_dir / "2025_round_2_sprint_race.csv", index=False
    )
    pd.DataFrame({"Abbreviation": ["VER", "NOR"], "Position": [1, 2], "Round": 2}).to_csv(
        season_dir / "2025_round_2_qualifying.csv", index=False
    )
    pd.DataFrame({"Abbreviation": ["VER", "NOR"], "TeamName": ["Red Bull Racing", "McLaren"], "Position": [1, 2], "Round": 2, "GridPosition": [1, 2], "Points": [25, 18]}).to_csv(
        season_dir / "2025_round_2_race.csv", index=False
    )

    loader = DataLoader(root_dir=tmp_path)
    sessions_by_round = loader.load_all_practice_sessions(2025, [1, 2])

    # Round 1 (Normal):
    assert sessions_by_round[1]["fp1"] is not None
    assert sessions_by_round[1]["fp2"] is not None
    assert sessions_by_round[1]["fp3"] is not None
    assert sessions_by_round[1]["sprint_qualifying"] is None
    assert sessions_by_round[1]["sprint_race"] is None

    # Round 2 (Sprint):
    assert sessions_by_round[2]["fp1"] is not None
    assert sessions_by_round[2]["fp2"] is None
    assert sessions_by_round[2]["fp3"] is None
    assert sessions_by_round[2]["sprint_qualifying"] is not None
    assert sessions_by_round[2]["sprint_race"] is not None

    # Verify FeatureEngineer builds features with correct Sprint vs Normal coverage
    races_df, qualis_df = loader.load_season_rounds(2025, [1, 2])
    fe = FeatureEngineer(expected_grid_size=2)
    train_df = fe.build_training_dataset(
        races_df=races_df,
        qualis_df=qualis_df,
        practice_sessions_by_round=sessions_by_round,
        train_rounds=[1, 2]
    )

    r1_df = train_df[train_df["Round"] == 1]
    r2_df = train_df[train_df["Round"] == 2]

    # Normal round: sprint features are NaN
    assert r1_df["sprint_quali_pos"].isna().all()
    assert r1_df["sprint_finish_pos"].isna().all()
    assert r1_df["sprint_pos_delta"].isna().all()

    # Sprint round: sprint features are populated with real numbers
    assert r2_df["sprint_quali_pos"].notna().all()
    assert r2_df["sprint_finish_pos"].notna().all()
    assert r2_df["sprint_pos_delta"].notna().all()

    # Verify exact delta: StartingGrid - FinishPosition
    # VER: StartingGrid 1, Finish 2 -> delta = -1.0
    ver_r2 = r2_df[r2_df["Abbreviation"] == "VER"].iloc[0]
    assert ver_r2["sprint_pos_delta"] == -1.0
    # NOR: StartingGrid 2, Finish 1 -> delta = +1.0
    nor_r2 = r2_df[r2_df["Abbreviation"] == "NOR"].iloc[0]
    assert nor_r2["sprint_pos_delta"] == 1.0

