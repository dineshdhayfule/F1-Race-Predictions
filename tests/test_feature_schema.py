import pandas as pd

from src.features.build_features import FeatureEngineer
from src.features.tire_degradation import TIRE_FEATURE_COLUMNS, build_tire_degradation_features


def _race_data():
    return pd.DataFrame([
        {
            "Round": 1,
            "Abbreviation": "VER",
            "TeamName": "Team A",
            "GridPosition": 1,
            "Position": 1,
            "ActualFinish": 1,
            "Status": "Finished",
            "Race": "Test GP",
            "Points": 25,
        },
        {
            "Round": 1,
            "Abbreviation": "HAM",
            "TeamName": "Team B",
            "GridPosition": 2,
            "Position": 2,
            "ActualFinish": 2,
            "Status": "Finished",
            "Race": "Test GP",
            "Points": 18,
        },
    ])


def test_empty_tire_input_preserves_schema():
    features = build_tire_degradation_features({}, ["VER", "HAM"], circuit_key="las_vegas")

    assert list(features.columns) == TIRE_FEATURE_COLUMNS
    assert features["Abbreviation"].tolist() == ["VER", "HAM"]


def test_missing_driver_telemetry_keeps_other_driver_rows():
    telemetry = {
        "fp1": pd.DataFrame([
            {"Abbreviation": "VER", "LapTimeSeconds": 90.0, "Stint": 1, "Compound": "MEDIUM"},
        ])
    }
    features = build_tire_degradation_features(
        telemetry, ["VER", "HAM"], circuit_key="las_vegas"
    )

    assert features["Abbreviation"].tolist() == ["VER", "HAM"]


def test_training_and_target_builders_tolerate_disabled_empty_subsystems():
    races = _race_data()
    qualis = races[["Abbreviation", "TeamName", "Position"]].assign(Round=1)
    engineer = FeatureEngineer(
        expected_grid_size=2,
        capabilities={
            "tire_degradation_modeling": False,
            "weather_aware_prediction": False,
        },
    )

    training = engineer.build_training_dataset(races, qualis, {1: {}}, [1])
    target = engineer.build_target_dataset(
        target_round=2,
        track_name="Test GP",
        quali_df=qualis,
        practice_sessions={},
        season_races_df=races,
        expected_grid_size=2,
    )

    assert training["Abbreviation"].tolist() == ["VER", "HAM"]
    assert target["Abbreviation"].tolist() == ["VER", "HAM"]
