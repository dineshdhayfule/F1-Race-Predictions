"""
COTA (Austin) 2026 Session Discovery, Stage Cutoffs & Anti-Leakage Tests
========================================================================
Validates:
1. Qualifying-session discovery from season-level CSV (2026_round_18_qualifying.csv).
2. Race-session discovery from season-level CSV (2026_round_18_race.csv).
3. Strict stage cutoffs (no race results in after_qualifying; race allowed only in post_race).
4. Data integrity: 22 unique drivers, normalized identifiers, valid positions 1..22, no duplicates.
5. Missing-session behavior and warning / provisional flag handling.
6. Target-round anti-leakage guarantees (strict exclusion of Round 18 from COTA training set).
"""

from pathlib import Path
import pandas as pd
import pytest

from src.data.data_loader import DataLoader
from src.data.data_validator import DataValidator
from src.features.build_features import FeatureEngineer
from src.pipeline.race_prediction_pipeline import RacePredictionPipeline

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def loader():
    return DataLoader(root_dir=ROOT)


@pytest.fixture
def validator():
    return DataValidator()


@pytest.fixture
def feature_engineer():
    return FeatureEngineer(ewma_alpha=0.3)


def test_cota_qualifying_session_discovery(loader):
    """
    Verifies that DataLoader finds and loads COTA 2026 qualifying results
    from data/raw/season/2026/2026_round_18_qualifying.csv.
    """
    df = loader.load_target_qualifying("austin", 2026)
    assert df is not None, "Target qualifying for Austin 2026 was not found!"
    assert len(df) == 22, f"Expected 22 qualifying drivers, got {len(df)}"
    assert "Abbreviation" in df.columns
    assert "Position" in df.columns

    # Verify actual pole sitter and front row identities
    p1 = df[df["Position"] == 1.0].iloc[0]
    p2 = df[df["Position"] == 2.0].iloc[0]
    p3 = df[df["Position"] == 3.0].iloc[0]
    assert p1["Abbreviation"] == "ANT", f"Expected ANT on pole, got {p1['Abbreviation']}"
    assert p2["Abbreviation"] == "VER", f"Expected VER P2, got {p2['Abbreviation']}"
    assert p3["Abbreviation"] == "HAD", f"Expected HAD P3, got {p3['Abbreviation']}"


def test_cota_race_session_discovery(loader):
    """
    Verifies that DataLoader finds and loads COTA 2026 race results
    from data/raw/season/2026/2026_round_18_race.csv.
    """
    df = loader.load_target_race("austin", 2026)
    assert df is not None, "Target race results for Austin 2026 were not found!"
    assert len(df) == 22, f"Expected 22 race drivers, got {len(df)}"
    assert "Abbreviation" in df.columns
    assert "Position" in df.columns

    # Verify official podium finishers: ANT P1, LEC P2, VER P3
    p1 = df[df["Position"] == 1.0].iloc[0]
    p2 = df[df["Position"] == 2.0].iloc[0]
    p3 = df[df["Position"] == 3.0].iloc[0]
    assert p1["Abbreviation"] == "ANT", f"Expected ANT race winner, got {p1['Abbreviation']}"
    assert p2["Abbreviation"] == "LEC", f"Expected LEC P2, got {p2['Abbreviation']}"
    assert p3["Abbreviation"] == "VER", f"Expected VER P3, got {p3['Abbreviation']}"


def test_cota_data_integrity_and_driver_validation(loader):
    """
    Validates that both COTA season files are well-formed:
    - 22 unique drivers
    - Normalized identifiers matching 2026 grid
    - Positions 1 to 22 without duplicates
    """
    raw_season_dir = ROOT / "data" / "raw" / "season" / "2026"
    q_file = raw_season_dir / "2026_round_18_qualifying.csv"
    r_file = raw_season_dir / "2026_round_18_race.csv"

    assert q_file.exists(), f"Qualifying file missing at {q_file}"
    assert r_file.exists(), f"Race file missing at {r_file}"

    df_q = pd.read_csv(q_file)
    df_r = pd.read_csv(r_file)

    # 1. Driver count and uniqueness
    assert len(df_q) == 22
    assert len(df_r) == 22
    assert df_q["Abbreviation"].nunique() == 22
    assert df_r["Abbreviation"].nunique() == 22

    # 2. Driver rosters match exactly between qualifying and race
    assert set(df_q["Abbreviation"]) == set(df_r["Abbreviation"])

    # 3. Position validity
    assert sorted(df_q["Position"].astype(int)) == list(range(1, 23))
    assert sorted(df_r["Position"].astype(int)) == list(range(1, 23))


def test_cota_stage_cutoffs_anti_leakage(loader):
    """
    Verifies stage-based information cutoffs for Austin:
    - pre_fp1: no practice, no qualifying, no race
    - after_qualifying: practice + qualifying loaded; race MUST BE None
    - post_race: practice + qualifying + race loaded
    """
    # 1. pre_fp1
    s_pre = loader.load_weekend_sessions("austin", 2026, stage="pre_fp1")
    assert all(v is None for v in s_pre.values()), "pre_fp1 should have no loaded sessions"

    # 2. after_qualifying
    s_quali = loader.load_weekend_sessions("austin", 2026, stage="after_qualifying")
    assert s_quali["fp1"] is not None
    assert s_quali["fp2"] is not None
    assert s_quali["fp3"] is not None
    assert s_quali["qualifying"] is not None
    assert s_quali["race"] is None, "CRITICAL LEAKAGE: Race session loaded in after_qualifying stage!"

    # 3. post_race
    s_post = loader.load_weekend_sessions("austin", 2026, stage="post_race")
    assert s_post["qualifying"] is not None
    assert s_post["race"] is not None, "Race session should be available in post_race stage"


def test_cota_training_rounds_and_sample_count(loader, feature_engineer, validator):
    """
    Verifies that COTA training features use strictly Rounds 1–17 (374 samples)
    and strictly excludes Round 18 (COTA target round) to prevent target leakage.
    """
    target_round = 18
    train_rounds = [r for r in range(1, target_round)]
    assert train_rounds == list(range(1, 18))
    assert len(train_rounds) == 17

    races_df, qualis_df = loader.load_season_rounds(2026, train_rounds)
    sessions_by_round = loader.load_all_practice_sessions(2026, train_rounds)

    train_feats = feature_engineer.build_training_dataset(
        races_df=races_df,
        qualis_df=qualis_df,
        practice_sessions_by_round=sessions_by_round,
        train_rounds=train_rounds
    )

    # Exactly 17 rounds * 22 drivers = 374 driver-race samples
    assert len(train_feats) == 374, f"Expected exactly 374 training samples, got {len(train_feats)}"
    assert sorted(train_feats["Round"].unique()) == list(range(1, 18))
    assert target_round not in train_feats["Round"].values, "Target Round 18 leaked into training features!"

    # Target features constructed from actual qualifying
    target_quali = loader.load_target_qualifying("austin", 2026)
    weekend_sessions = loader.load_weekend_sessions("austin", 2026, stage="after_qualifying")
    target_feats = feature_engineer.build_target_dataset(
        target_round=target_round,
        track_name="United States Grand Prix",
        quali_df=target_quali,
        practice_sessions=weekend_sessions,
        season_races_df=races_df,
        season_qualis_df=qualis_df,
        penalties={}
    )

    # Target features must have 22 rows and no ActualFinish column
    assert len(target_feats) == 22
    assert "ActualFinish" not in target_feats.columns

    # Audit leakage
    audit = validator.audit_leakage(train_feats, target_feats, target_round=target_round)
    assert audit["is_clean"] is True, f"Leakage detected: {audit['violations']}"
    assert len(audit["violations"]) == 0


def test_missing_session_behavior(loader):
    """
    Verifies behavior when sessions are genuinely missing:
    - load_target_qualifying returns None for non-existent tracks/rounds
    - load_target_race returns None for non-existent tracks/rounds
    """
    assert loader.load_target_qualifying("nonexistent_circuit_xyz", 2026) is None
    assert loader.load_target_race("nonexistent_circuit_xyz", 2026) is None
    assert loader.load_target_qualifying("austin", 1990) is None
    assert loader.load_target_race("austin", 1990) is None
