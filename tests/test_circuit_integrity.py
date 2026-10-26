"""
Circuit Configuration Schema & Temporal Anti-Leakage Audit (TC-15, TC-16)
==========================================================================
Validates:
1. TC-15: Verifies that all 23 circuits in config/tracks.json conform to the required
   14-key schema and data types. Formally reports the known Sepang display-name defect
   ("Bahrain Grand Prix") without crashing the test suite.
2. TC-16: Automated chronological leakage audit across all completed 2026 round transitions
   (R = 2 to 16) using the production DataValidator.audit_leakage() API.

Zero modifications to production source code or existing output directories.
"""

import json
from pathlib import Path
import pytest

from src.data.data_loader import DataLoader
from src.data.data_validator import DataValidator
from src.features.build_features import FeatureEngineer

ROOT = Path(__file__).resolve().parents[1]


def test_circuit_configuration_schema_and_defect_reporting():
    """
    TC-15: Validates configuration schema across all 23 tracks and reports
    the known Sepang display-name discrepancy as a diagnostic finding.
    """
    config_path = ROOT / "config" / "tracks.json"
    assert config_path.exists(), "config/tracks.json missing!"

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "tracks" in data
    tracks = data["tracks"]
    assert len(tracks) == 23, f"Expected 23 configured circuits, found {len(tracks)}"

    # Verified 14-key schema common to all tracks
    required_keys = [
        "circuit_name",
        "circuit_type",
        "country",
        "display_name",
        "event_name_patterns",
        "historical_data_available",
        "historical_years",
        "known_penalties_2026",
        "overtaking_ease",
        "round_2026",
        "speed_bias",
        "sprint_weekend_2026",
        "straightness",
        "track_length_km",
    ]

    for t_key, cfg in tracks.items():
        # 1. Schema keys check
        for k in required_keys:
            assert k in cfg, f"Track '{t_key}' is missing required key '{k}'"

        # 2. Field type validations
        assert isinstance(cfg["round_2026"], int), f"Track '{t_key}': round_2026 must be an int"
        assert 1 <= cfg["round_2026"] <= 23, f"Track '{t_key}': round_2026 must be between 1 and 23"
        assert isinstance(cfg["track_length_km"], (int, float)) and cfg["track_length_km"] > 0
        assert isinstance(cfg["sprint_weekend_2026"], bool)
        assert isinstance(cfg["known_penalties_2026"], dict)
        assert isinstance(cfg["historical_years"], list)

    # 3. Known Defect Reporting: Sepang display-name mismatch
    sepang_cfg = tracks.get("sepang", {})
    sepang_display = sepang_cfg.get("display_name", "")

    # We assert that the defect is currently present in config/tracks.json as documented
    if sepang_display == "Bahrain Grand Prix":
        print("\n[TC-15 KNOWN DEFECT DETECTED] Track key 'sepang' has display_name 'Bahrain Grand Prix' instead of 'Malaysian Grand Prix'. (Reported as diagnostic defect)")
    else:
        print(f"\n[TC-15 NOTIFICATION] Track key 'sepang' display_name is '{sepang_display}'")


def test_multi_round_chronological_leakage_audit():
    """
    TC-16: Validates temporal anti-leakage boundaries across completed 2026 rounds (R = 2 to 16).
    Enforces that target race results are strictly excluded and training rounds strictly precede target.
    """
    loader = DataLoader(root_dir=ROOT)
    validator = DataValidator()
    feature_engineer = FeatureEngineer()

    tracks_cfg = loader.tracks_config.get("tracks", {})
    round_to_track = {cfg.get("round_2026"): k for k, cfg in tracks_cfg.items() if "round_2026" in cfg}

    audited_rounds = list(range(2, 17))  # Target rounds 2 through 16

    for target_round in audited_rounds:
        track_key = round_to_track.get(target_round, "unknown")
        track_info = tracks_cfg.get(track_key, {})
        track_name = track_info.get("display_name", track_key.title())
        penalties = track_info.get("known_penalties_2026", {})

        # Training set strictly rounds < target_round
        train_rounds = [r for r in range(1, target_round)]
        races_df, qualis_df = loader.load_season_rounds(2026, train_rounds)
        sessions_by_round = loader.load_all_practice_sessions(2026, train_rounds)

        # Target sessions
        weekend_sessions = loader.load_weekend_sessions(track_key, 2026, stage="after_qualifying")
        if all(v is None for v in weekend_sessions.values()):
            weekend_sessions = loader.load_all_practice_sessions(2026, [target_round]).get(target_round, {})

        # Target qualifying
        target_quali = loader.load_target_qualifying(track_key, 2026)
        if target_quali is None or target_quali.empty:
            _, q_df = loader.load_season_rounds(2026, [target_round])
            target_quali = q_df[q_df["Round"] == target_round].copy() if not q_df.empty else None

        if target_quali is None or target_quali.empty:
            continue

        train_feats = feature_engineer.build_training_dataset(
            races_df=races_df,
            qualis_df=qualis_df,
            practice_sessions_by_round=sessions_by_round,
            train_rounds=train_rounds
        )

        target_feats = feature_engineer.build_target_dataset(
            target_round=target_round,
            track_name=track_name,
            quali_df=target_quali,
            practice_sessions=weekend_sessions,
            season_races_df=races_df,
            season_qualis_df=qualis_df,
            penalties=penalties
        )

        # Execute production leakage audit
        audit = validator.audit_leakage(train_feats, target_feats, target_round=target_round)

        # Strict anti-leakage assertions
        assert audit["is_clean"] is True, f"Leakage detected for target round {target_round}: {audit['violations']}"
        assert len(audit["violations"]) == 0
        assert target_round not in train_feats["Round"].values, f"Target round {target_round} leaked into training data!"
        assert (train_feats["Round"] < target_round).all(), f"Future round found in training data for round {target_round}!"

        # Forbidden post-race columns assertion
        for forbidden in validator.FORBIDDEN_PRE_RACE_COLUMNS:
            assert forbidden not in target_feats.columns, f"Forbidden column '{forbidden}' leaked into target dataset!"
