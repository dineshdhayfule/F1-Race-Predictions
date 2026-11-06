"""
Tests for Historical FastF1 Compatibility, Driver Preservation, Grid Extraction,
and Readiness Logic (Phase 2).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from src.data.collect_season import (
    validate_and_format_session_results,
    validate_and_format_practice_laps,
    collect_round_data,
)
from src.data.grid_retriever import StartingGridRetriever
from src.data.data_loader import DataLoader
from src.cli.menu import InteractiveMenu
from src.features.grid_features import GridProvenance, GridPublicationStatus
from src.utils.driver_mapping import normalize_driver, normalize_driver_series


# =============================================================================
# Helper Fixtures
# =============================================================================

def _make_historical_results_df(n_drivers: int = 21) -> pd.DataFrame:
    """Creates a mock historical FastF1 session results DataFrame like 1950 Ergast."""
    names = [
        ("Nino Farina", "farina", 2, "Alfa Romeo"),
        ("Luigi Fagioli", "fagioli", 3, "Alfa Romeo"),
        ("Reg Parnell", "reg_parnell", 4, "Alfa Romeo"),
        ("Yves Cabantous", "cabantous", 14, "Talbot-Lago"),
        ("Louis Rosier", "rosier", 15, "Talbot-Lago"),
        ("Bob Gerard", "gerard", 12, "ERA"),
        ("Cuth Harrison", "harrison", 11, "ERA"),
        ("Philippe Étancelin", "etancelin", 16, "Talbot-Lago"),
        ("David Hampshire", "hampshire", 6, "Maserati"),
        ("Joe Fry", "fry", 10, "Maserati"),
        ("Brian Shawe-Taylor", "shawe_taylor", 10, "Maserati"),
        ("Johnny Claes", "claes", 18, "Talbot-Lago"),
        ("Eugene Martin", "martin", 1, "Talbot-Lago"),
        ("Prince Bira", "bira", 23, "Maserati"),
        ("David Murray", "murray", 21, "Maserati"),
        ("Toulo de Graffenried", "de_graffenried", 5, "Maserati"),
        ("Geoff Crossley", "crossley", 24, "Alta"),
        ("Peter Walker", "walker", 9, "ERA"),
        ("Tony Rolt", "rolt", 9, "ERA"),
        ("Leslie Johnson", "johnson", 20, "ERA"),
        ("Juan Manuel Fangio", "fangio", 19, "Alfa Romeo"),
    ]
    rows = []
    for i in range(min(n_drivers, len(names))):
        full_name, drv_id, num, team = names[i]
        rows.append({
            "DriverNumber": str(num),
            "BroadcastName": "",
            "Abbreviation": "",  # Empty strings as in Ergast archive
            "DriverId": drv_id,
            "TeamName": team,
            "FullName": full_name,
            "Position": float(i + 1),
            "ClassifiedPosition": str(i + 1),
            "GridPosition": float(i + 1),
            "Status": "Finished" if i < 10 else "Retired",
            "Points": 9.0 if i == 0 else (6.0 if i == 1 else (4.0 if i == 2 else 0.0)),
            "Time": "0 days 02:13:23.600" if i == 0 else pd.NaT,
        })
    return pd.DataFrame(rows)


def _make_mock_session(df: pd.DataFrame, year: int = 1950, round_num: int = 1, f1_api_support: bool = False):
    sess = MagicMock()
    sess.results = df.copy()
    sess.f1_api_support = f1_api_support
    sess.event = MagicMock()
    sess.event.RoundNumber = round_num
    sess.event.EventName = "British Grand Prix"
    sess.event.Location = "Silverstone"
    sess.event.Country = "UK"
    sess.event.EventDate = "1950-05-13"
    sess.event.F1ApiSupport = f1_api_support
    sess.date = "1950-05-13"
    return sess


# =============================================================================
# Test A — Historical Driver Preservation
# =============================================================================

def test_historical_driver_preservation_empty_abbreviations():
    """
    Test A: Verifies that a historical race with 21 drivers and empty Abbreviation values
    retains all 21 drivers without accidental deduplication, and normalizes identifiers stably.
    """
    raw_df = _make_historical_results_df(21)
    assert (raw_df["Abbreviation"] == "").all()
    assert len(raw_df) == 21

    sess = _make_mock_session(raw_df, year=1950, round_num=1)
    valid, msg, df = validate_and_format_session_results(
        session=sess,
        expected_year=1950,
        expected_round=1,
        event_name="British Grand Prix",
        circuit_name="Silverstone",
        country="UK",
        canonical_type="Race",
        original_session_name="Race",
    )
    assert valid is True
    assert msg == "VALID"
    # All 21 drivers must be retained
    assert len(df) == 21
    # Abbreviation must be non-empty and non-duplicated
    assert (df["Abbreviation"] != "").all()
    assert df["Abbreviation"].nunique() == 21
    assert "NINO FARINA" in df["Abbreviation"].values
    assert "LUIGI FAGIOLI" in df["Abbreviation"].values


def test_normalize_driver_fallback_order():
    """Verifies normalize_driver falls back to FullName / DriverId when Abbreviation is empty."""
    df = pd.DataFrame({
        "Abbreviation": ["", "  ", None],
        "FullName": ["Nino Farina", "Luigi Fagioli", "Reg Parnell"],
        "DriverId": ["farina", "fagioli", "reg_parnell"],
        "Position": [1.0, 2.0, 3.0]
    })
    norm = normalize_driver(df)
    assert norm["Abbreviation"].tolist() == ["NINO FARINA", "LUIGI FAGIOLI", "REG PARNELL"]


# =============================================================================
# Test B — Historical Missing Lap Data
# =============================================================================

def test_historical_missing_lap_data_data_not_loaded_error():
    """
    Test B: Simulates a session whose laps property raises DataNotLoadedError.
    Verifies that collection does not crash, no fake laps are generated,
    and returns unavailable lap telemetry cleanly.
    """
    import fastf1.exceptions

    sess = MagicMock()
    sess.f1_api_support = False
    sess.results = pd.DataFrame()
    # Mock laps property to raise DataNotLoadedError
    type(sess).laps = property(lambda self: (_ for _ in ()).throw(
        fastf1.exceptions.DataNotLoadedError("The data you are trying to access has not been loaded yet.")
    ))

    valid, msg, df = validate_and_format_practice_laps(
        session=sess,
        expected_year=1950,
        expected_round=1,
        event_name="British Grand Prix",
        circuit_name="Silverstone",
        country="UK",
        canonical_type="Practice 1",
    )
    # Does not crash; reports unavailable session results
    assert valid is False
    assert df is None
    assert "Session results table is empty" in msg or "not available" in msg.lower()


# =============================================================================
# Test C — Historical Loading Behavior
# =============================================================================

def test_historical_loading_behavior_disables_unsupported_laps():
    """
    Test C: Verifies that for pre-2018 or F1ApiSupport=False sessions,
    collect_round_data calls sess.load(laps=False, ...).
    """
    mock_sess = MagicMock()
    mock_sess.f1_api_support = False
    mock_sess.results = _make_historical_results_df(21)
    mock_sess.event.F1ApiSupport = False

    event_row = pd.Series({
        "EventName": "British Grand Prix",
        "Location": "Silverstone",
        "Country": "UK",
        "EventFormat": "conventional",
        "Session1": "Practice 1",
        "Session2": "Practice 2",
        "Session3": "Practice 3",
        "Session4": "Qualifying",
        "Session5": "Race",
    })

    with patch("fastf1.get_session", return_value=mock_sess):
        # Run with dummy tmp_path output
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir)
            summary = collect_round_data(
                year=1950,
                round_num=1,
                event_row=event_row,
                output_dir=out_path,
                force=True,
                session_mode="prediction_only",
            )
            # Verify that load was called with laps=False
            calls = mock_sess.load.call_args_list
            assert len(calls) > 0
            for call in calls:
                kwargs = call.kwargs
                assert kwargs.get("laps") is False
                assert kwargs.get("telemetry") is False
                assert kwargs.get("weather") is False


# =============================================================================
# Test D — Historical Grid Extraction & Zero Data Leakage
# =============================================================================

def test_historical_grid_extraction_no_leakage(tmp_path):
    """
    Test D: Verifies that starting grid extracted from historical race results
    contains only legitimate pre-race grid information, with ZERO post-race outcome fields.
    """
    race_df = _make_historical_results_df(21)
    # Save as historical race CSV
    season_dir = tmp_path / "data" / "raw" / "season" / "1950"
    season_dir.mkdir(parents=True, exist_ok=True)
    race_file = season_dir / "1950_round_1_race.csv"
    race_df.to_csv(race_file, index=False)

    retriever = StartingGridRetriever(root_dir=tmp_path)
    grid_df, grid_meta = retriever._retrieve_historical_race_grid(
        track_key="silverstone",
        year=1950,
        round_num=1,
        stage="after_qualifying",
    )

    assert grid_df is not None
    assert len(grid_df) == 21
    assert grid_meta["provenance"] == GridProvenance.HISTORICAL_RACE_RESULTS.value
    assert grid_meta["publication_status"] == GridPublicationStatus.FINAL.value

    # STRICT ZERO DATA LEAKAGE ASSERTIONS:
    # Outcome fields MUST NEVER be present in starting grid DataFrame
    forbidden_columns = [
        "Position", "ClassifiedPosition", "Points", "Status",
        "Time", "Laps", "FastestLap", "ActualFinish"
    ]
    for col in forbidden_columns:
        assert col not in grid_df.columns, f"LEAKAGE DEFECT: '{col}' leaked into starting grid DataFrame!"

    # Allowed pre-race fields
    assert "ActualStartingGrid" in grid_df.columns
    assert "simulation_start_rank" in grid_df.columns
    assert "Abbreviation" in grid_df.columns
    assert "TeamName" in grid_df.columns

    # Verify grid positions match input pre-race grid
    assert grid_df.loc[grid_df["Abbreviation"] == "NINO FARINA", "ActualStartingGrid"].iloc[0] == 1.0
    assert grid_df.loc[grid_df["Abbreviation"] == "LUIGI FAGIOLI", "ActualStartingGrid"].iloc[0] == 2.0


# =============================================================================
# Test E — Missing Historical Qualifying
# =============================================================================

def test_missing_historical_qualifying_uses_historical_grid_fallback(tmp_path):
    """
    Test E: Verifies that when a separate qualifying file does not exist,
    the loader does not fabricate qualifying results, and StartingGridRetriever
    successfully resolves the starting grid via historical race fallback.
    """
    # Create race file only, no qualifying file
    season_dir = tmp_path / "data" / "raw" / "season" / "1950"
    season_dir.mkdir(parents=True, exist_ok=True)
    race_file = season_dir / "1950_round_1_race.csv"
    _make_historical_results_df(21).to_csv(race_file, index=False)

    loader = DataLoader(root_dir=tmp_path)
    q_df = loader.load_target_qualifying("silverstone", year=1950, round_num=1)
    # Qualifying must NOT be fabricated
    assert q_df is None

    # Starting grid must resolve via historical race results fallback
    grid_df, grid_meta = loader.load_starting_grid(
        track_id="silverstone",
        year=1950,
        round_num=1,
        stage="after_qualifying",
        quali_df=None,
    )
    assert grid_df is not None
    assert len(grid_df) == 21
    assert grid_meta["provenance"] == GridProvenance.HISTORICAL_RACE_RESULTS.value


# =============================================================================
# Test F — Round 1 Training Readiness
# =============================================================================

def test_round1_training_readiness_reports_limitation_safely(tmp_path):
    """
    Test F: Verifies Round 1 readiness reports an explicit limitation about no prior
    championship rounds without mislabeling verified session data as missing.
    """
    # Create 1950 Round 1 race CSV
    season_dir = tmp_path / "data" / "raw" / "season" / "1950"
    season_dir.mkdir(parents=True, exist_ok=True)
    race_file = season_dir / "1950_round_1_race.csv"
    _make_historical_results_df(21).to_csv(race_file, index=False)

    menu = InteractiveMenu(root_dir=tmp_path)
    ready, details = menu._audit_prediction_readiness("silverstone", 1950, 1, "after_qualifying", is_sprint=False)

    # Round 1 has no prior rounds in 1950, so is_ready must be False
    assert ready is False

    # Check that starting grid was verified
    ok_msgs = [msg for status, msg in details if status == "OK"]
    assert any("Historical Starting Grid verified" in m for m in ok_msgs)

    # Check that Round 1 missing priors is explicitly reported
    missing_msgs = [msg for status, msg in details if status == "MISSING"]
    assert any("Round 1" in m for m in missing_msgs)
    assert any("No prior" in m for m in missing_msgs)


# =============================================================================
# Phase 2.2 Tests — Shared Drives, Provenance, Readiness & Feature Handling
# =============================================================================

def test_historical_shared_drives_separated_from_physical_cars(tmp_path):
    """
    Test G.1: Verifies that when a race contains shared drives (e.g. 1950 British GP
    with 23 driver records across 21 cars), StartingGridRetriever produces exactly
    21 physical car grid entries while preserving all 23 driver participation records.
    """
    # 23 driver records with Car 9 and Car 10 shared
    race_df = _make_historical_results_df(21)
    # Ensure car 9 and 10 have duplicate rows
    assert len(race_df) == 21
    # Add second driver for car 9 (Tony Rolt) and car 10 (Brian Shawe Taylor)
    # In _make_historical_results_df, Car 9 has Peter Walker and Car 10 has Joe Fry
    # Let's ensure car numbers 9 and 10 are duplicated
    c9_row = race_df[race_df["DriverNumber"] == "9"].iloc[0].to_dict()
    c9_row["FullName"] = "Tony Rolt"
    c9_row["DriverId"] = "rolt"
    c9_row["Abbreviation"] = ""

    c10_row = race_df[race_df["DriverNumber"] == "10"].iloc[0].to_dict()
    c10_row["FullName"] = "Brian Shawe Taylor"
    c10_row["DriverId"] = "shawe_taylor"
    c10_row["Abbreviation"] = ""

    race_23 = pd.concat([race_df, pd.DataFrame([c9_row, c10_row])], ignore_index=True)
    assert len(race_23) == 23

    season_dir = tmp_path / "data" / "raw" / "season" / "1950"
    season_dir.mkdir(parents=True, exist_ok=True)
    race_23.to_csv(season_dir / "1950_round_1_race.csv", index=False)

    retriever = StartingGridRetriever(root_dir=tmp_path)
    grid_df, meta = retriever.get_starting_grid("silverstone", year=1950, round_num=1, stage="after_qualifying")

    assert grid_df is not None
    # 1. Exactly 21 physical car entries
    assert len(grid_df) == 21
    assert meta["total_physical_cars"] == 21

    # 2. Exactly 23 driver participation records preserved
    assert len(meta["driver_participations"]) == 23
    assert meta["total_driver_participations"] == 23

    # 3. Grid positions are unique and contiguous 1..21
    actual_positions = grid_df["ActualStartingGrid"].dropna().tolist()
    assert len(actual_positions) == 21
    assert len(set(actual_positions)) == 21
    assert sorted(actual_positions) == list(range(1, 22))

    # 4. Simulation ranks are contiguous 1..21
    assert list(grid_df["simulation_start_rank"]) == list(range(1, 22))

    # 5. Grid normalization uses 21 physical cars
    # Rank 1: norm = 0.0, Rank 21: norm = 1.0
    assert grid_df.loc[grid_df["simulation_start_rank"] == 1.0, "grid_norm"].iloc[0] == 0.0
    assert grid_df.loc[grid_df["simulation_start_rank"] == 21.0, "grid_norm"].iloc[0] == 1.0

    # 6. Starting driver identity is not fabricated for shared cars
    shared_cars = grid_df[grid_df["is_shared_drive"]]
    assert len(shared_cars) == 2
    for _, car in shared_cars.iterrows():
        assert car["is_starting_driver_resolved"] is False
        assert car["StartingDriver"] is None

    # 7. Physical grid passes validate_grid_dataframe
    valid, errors = retriever.validate_grid_dataframe(grid_df, expected_grid_size=21)
    assert valid is True
    assert len(errors) == 0


def test_pipeline_gridprovenance_imported():
    """
    Test G.2: Verifies that GridProvenance is imported and accessible in the pipeline orchestrator.
    """
    import src.pipeline.race_prediction_pipeline as rpp
    assert hasattr(rpp, "GridProvenance")
    assert rpp.GridProvenance.HISTORICAL_RACE_RESULTS.value == "historical_race_results"


def test_historical_qualifying_absence_retains_all_entrants(tmp_path):
    """
    Test G.3: Verifies that when qualifying is absent, build_target_dataset retains all
    physical starting grid entrants (no rows dropped due to NaN qualifying Position),
    and does not manufacture fake qualifying performance.
    """
    from src.features.build_features import FeatureEngineer

    # 21 physical cars
    grid_data = []
    for pos in range(1, 22):
        grid_data.append({
            "CarNumber": str(pos),
            "Abbreviation": f"DRV_{pos}",
            "TeamName": f"Team_{pos % 5}",
            "QualifyingPosition": np.nan,
            "ActualStartingGrid": float(pos),
            "simulation_start_rank": float(pos),
            "is_pit_lane_start": 0,
            "StartingGridStatus": "CLASSIFIED",
            "GridPenalty": 0,
            "grid_penalty_places": 0,
            "grid_vs_quali_delta": 0,
            "grid_norm": (pos - 1) / 20.0,
            "is_shared_drive": False,
            "is_starting_driver_resolved": True,
            "StartingDriver": f"DRV_{pos}",
            "shared_drivers": [f"DRV_{pos}"],
        })
    grid_df = pd.DataFrame(grid_data)

    mock_races = pd.DataFrame({
        "Year": [1950],
        "Round": [0],
        "Abbreviation": ["DRV_1"],
        "GridPosition": [1.0],
        "Position": [1.0],
        "Team": ["Team_1"],
        "TeamName": ["Team_1"],
        "Status": ["Finished"]
    })

    fe = FeatureEngineer()
    target_df = fe.build_target_dataset(
        target_round=1,
        track_name="Silverstone",
        quali_df=None,  # No qualifying session
        practice_sessions={},
        season_races_df=mock_races,
        season_qualis_df=pd.DataFrame(),
        starting_grid_df=grid_df,
        stage="after_qualifying",
        track_key="silverstone",
        expected_grid_size=21
    )

    # All 21 drivers must be retained
    assert len(target_df) == 21
    # Check indicators
    assert "QualifyingDataAvailable" in target_df.columns
    assert (target_df["QualifyingDataAvailable"] == 0).all()
    assert (target_df["GridBasedComparisonAvailable"] == 1).all()


def test_readiness_classification_blocked_experimental_validated(tmp_path):
    """
    Test G.4: Verifies the 3-state readiness classification:
    - BLOCKED when 0 rounds or missing grid
    - EXPERIMENTAL when 1..2 rounds available (< 3 rounds for walk-forward)
    - VALIDATED when >= 3 rounds available
    """
    menu = InteractiveMenu(root_dir=tmp_path)
    season_dir = tmp_path / "data" / "raw" / "season" / "1950"
    season_dir.mkdir(parents=True, exist_ok=True)

    # Case 1: Round 1 (0 prior rounds) -> BLOCKED
    audit_r1 = menu._audit_prediction_readiness("silverstone", 1950, 1, "after_qualifying", is_sprint=False)
    assert audit_r1.is_ready is False
    assert audit_r1.classification == "BLOCKED"

    # Add Round 1 race
    _make_historical_results_df(21).to_csv(season_dir / "1950_round_1_race.csv", index=False)
    # Add Round 2 race so starting grid exists for Monaco
    _make_historical_results_df(21).to_csv(season_dir / "1950_round_2_race.csv", index=False)

    # Case 2: Round 2 (1 prior round available) -> EXPERIMENTAL
    audit_r2 = menu._audit_prediction_readiness("monaco", 1950, 2, "after_qualifying", is_sprint=False)
    assert audit_r2.is_ready is True
    assert audit_r2.classification == "EXPERIMENTAL"
    assert audit_r2.can_walk_forward is False
    assert audit_r2.completed_rounds == 1

    # Add Round 2 and Round 3 prior races
    _make_historical_results_df(21).to_csv(season_dir / "1950_round_3_race.csv", index=False)
    _make_historical_results_df(21).to_csv(season_dir / "1950_round_4_race.csv", index=False)

    # Case 3: Round 4 (3 prior rounds available: 1, 2, 3) -> VALIDATED
    audit_r4 = menu._audit_prediction_readiness("spa", 1950, 4, "after_qualifying", is_sprint=False)
    assert audit_r4.is_ready is True
    assert audit_r4.classification == "VALIDATED"
    assert audit_r4.can_walk_forward is True
    assert audit_r4.completed_rounds == 3
