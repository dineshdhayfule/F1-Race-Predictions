import numpy as np
import pandas as pd

from src.features.grid_conversion import compute_grid_conversion_map
from src.features.grid_features import apply_grid_penalties
from src.features.season_features import classify_race_status, is_dnf, is_classified_finish
from src.simulation.monte_carlo import MonteCarloSimulator


def test_finished_status_is_classified():
    assert classify_race_status("Finished") == "FINISHED"
    assert is_classified_finish("Finished") is True
    assert is_dnf("Finished") is False


def test_retired_status_is_dnf_but_not_classified():
    assert classify_race_status("Engine") == "RETIRED"
    assert is_classified_finish("Engine") is False
    assert is_dnf("Engine") is True


def test_dns_status_is_not_finish():
    assert classify_race_status("Did not start") == "DNS"
    assert is_classified_finish("Did not start") is False
    assert is_dnf("Did not start") is False


def test_dsq_status_is_not_finish():
    assert classify_race_status("Disqualified") == "DSQ"
    assert is_classified_finish("Disqualified") is False


def test_nc_status_is_not_finish():
    assert classify_race_status("Not classified") == "NC"
    assert is_classified_finish("Not classified") is False


def test_grid_conversion_maps_classified_finishes_and_excludes_dnfs():
    df = pd.DataFrame([
        {"Abbreviation": "VER", "GridPosition": 1, "Position": 1, "Status": "Finished"},
        {"Abbreviation": "NOR", "GridPosition": 2, "Position": 2, "Status": "Retired"},
    ])
    converted = compute_grid_conversion_map(df, expected_grid_size=2)
    assert converted[1]["expected_grid_finish"] == 1.0
    assert converted[2]["expected_grid_finish"] == 1.714


def test_missing_status_uses_safe_fallback():
    df = pd.DataFrame([{"Abbreviation": "VER", "GridPosition": 1, "Position": 1}])
    converted = compute_grid_conversion_map(df, expected_grid_size=2)
    assert set(converted[1]) == {"expected_grid_finish", "grid_retention_prob"}


def test_penalty_shifts_qualifying_grid_position():
    df = pd.DataFrame([{
        "Abbreviation": "VER",
        "TeamName": "Red Bull",
        "QualifyingPosition": 1,
        "GridPosition": 1,
    }, {
        "Abbreviation": "NOR",
        "TeamName": "McLaren",
        "QualifyingPosition": 2,
        "GridPosition": 2,
    }])
    result = apply_grid_penalties(df, {"VER": 1}, expected_grid_size=2)
    assert float(result.loc[result["Abbreviation"] == "VER", "ActualStartingGrid"].iloc[0]) == 2


def test_pit_lane_start_does_not_replace_physical_grid():
    df = pd.DataFrame([{
        "Abbreviation": "NOR",
        "TeamName": "McLaren",
        "QualifyingPosition": 3,
        "GridPosition": 3,
        "is_pit_lane_start": 1,
    }, {
        "Abbreviation": "VER",
        "TeamName": "Red Bull",
        "QualifyingPosition": 1,
        "GridPosition": 1,
    }])
    result = apply_grid_penalties(df, {}, expected_grid_size=2)
    assert int(result.loc[result["Abbreviation"] == "NOR", "is_pit_lane_start"].iloc[0]) == 1


def test_duplicate_grid_positions_are_rejected():
    df = pd.DataFrame([
        {"Abbreviation": "VER", "GridPosition": 1, "ActualStartingGrid": 1,
         "simulation_start_rank": 1, "is_pit_lane_start": 0},
        {"Abbreviation": "NOR", "GridPosition": 1, "ActualStartingGrid": 1,
         "simulation_start_rank": 2, "is_pit_lane_start": 0},
    ])
    from src.data.grid_retriever import StartingGridRetriever
    valid, error = StartingGridRetriever().validate_grid_dataframe(df)
    assert valid is False
    assert error


def test_duplicate_driver_entries_are_rejected():
    df = pd.DataFrame([
        {"Abbreviation": "VER", "GridPosition": 1, "ActualStartingGrid": 1,
         "simulation_start_rank": 1, "is_pit_lane_start": 0},
        {"Abbreviation": "VER", "GridPosition": 2, "ActualStartingGrid": 2,
         "simulation_start_rank": 2, "is_pit_lane_start": 0},
    ])
    from src.data.grid_retriever import StartingGridRetriever
    valid, error = StartingGridRetriever().validate_grid_dataframe(df)
    assert valid is False
    assert error


def test_monte_carlo_preserves_grid_size():
    config = {
        "monte_carlo": {"default_simulations": 100},
        "random_seed": 42,
    }
    simulator = MonteCarloSimulator(config)
    target = pd.DataFrame({
        "Abbreviation": [f"D{i}" for i in range(1, 23)],
        "TeamName": [f"T{i//2}" for i in range(1, 23)],
        "GridPosition": list(range(1, 23)),
        "StartingGridStatus": ["CLASSIFIED"] * 22,
        "is_pit_lane_start": [0] * 22,
    })
    result, _ = simulator.simulate(target, np.arange(1.0, 23.0), n_simulations=100)
    assert len(result) == 22
