import numpy as np
import pandas as pd
import pytest

from apexgrid.simulation.monte_carlo import MonteCarloSimulator


def _config():
    return {
        "random_seed": 42,
        "monte_carlo": {
            "default_simulations": 10000,
            "precision_simulations": 25000,
            "fallback_sigma": 2.85,
            "fallback_dnf_prob": 0.158,
        },
    }


def _target():
    return pd.DataFrame({
        "Abbreviation": [f"D{i}" for i in range(1, 23)],
        "TeamName": [f"T{i//2}" for i in range(1, 23)],
        "GridPosition": list(range(1, 23)),
        "StartingGridStatus": ["CLASSIFIED"] * 22,
        "is_pit_lane_start": [0] * 22,
    })


def test_default_draw_count():
    simulator = MonteCarloSimulator(_config())
    assert simulator.n_simulations == 10000


@pytest.mark.slow
def test_10k_and_25k_draw_depth():
    simulator = MonteCarloSimulator(_config())
    target = _target()
    scores = np.arange(1.0, 23.0)
    r10, m10 = simulator.simulate(target, scores, n_simulations=10000)
    r25, m25 = simulator.simulate(target, scores, n_simulations=25000)
    assert m10["n_simulations"] == 10000
    assert m25["n_simulations"] == 25000
    assert len(r10) == len(r25) == 22


def test_fixed_seed_is_deterministic():
    target = _target()
    scores = np.linspace(1.0, 22.0, 22)
    simulator_a = MonteCarloSimulator(_config())
    simulator_b = MonteCarloSimulator(_config())
    result_a, _ = simulator_a.simulate(target, scores, n_simulations=2000)
    result_b, _ = simulator_b.simulate(target, scores, n_simulations=2000)
    pd.testing.assert_frame_equal(result_a, result_b)


def test_probability_monotonicity():
    simulator = MonteCarloSimulator(_config())
    result, _ = simulator.simulate(_target(), np.linspace(1.5, 21.5, 22), n_simulations=5000)
    for _, row in result.iterrows():
        assert row["WinProbability"] <= row["Top3Probability"] + 1e-4
        assert row["Top3Probability"] <= row["Top5Probability"] + 1e-4
        assert row["Top5Probability"] <= row["PointsProbability"] + 1e-4


def test_nan_grid_and_pit_lane_are_handled():
    target = _target()
    target["GridPosition"] = np.nan
    target["is_pit_lane_start"] = np.nan
    simulator = MonteCarloSimulator(_config())
    result, _ = simulator.simulate(target, np.arange(1.0, 23.0), n_simulations=100)
    assert len(result) == 22
    assert not result["ExpectedFinish"].isna().any()


def test_residual_sigma_floor():
    simulator = MonteCarloSimulator(_config())
    full_target = _target()
    _, meta = simulator.simulate(
        full_target,
        np.arange(1.0, len(full_target) + 1.0),
        residuals=np.array([0.01, -0.01, 0.02, 0.0, -0.02]),
        n_simulations=100,
    )
    assert meta["calibrated_sigma"] == 1.5

    # Sparse-grid n < 20 scaling clamps 1.5 * (2 / 20) to the 1.20 floor
    _, meta_sparse = simulator.simulate(
        full_target.iloc[:2],
        np.array([1.0, 2.0]),
        residuals=np.array([0.01, -0.01, 0.02, 0.0, -0.02]),
        n_simulations=100,
    )
    assert meta_sparse["calibrated_sigma"] == 1.2


def test_dnf_fallback_probability_is_configured():
    assert _config()["monte_carlo"]["fallback_dnf_prob"] == 0.158


def test_driver_specific_dnf_probabilities_are_reported():
    simulator = MonteCarloSimulator(_config())
    target = _target().iloc[:2].copy()
    result, _ = simulator.simulate(
        target,
        np.array([1.0, 2.0]),
        driver_dnf_probs=np.array([0.0, 1.0]),
        driver_dnf_driver_ids=target["Abbreviation"].tolist(),
        n_simulations=500,
    )
    assert result.loc[result["Abbreviation"] == "D1", "DNFProbability"].iloc[0] == 0.0
    assert result.loc[result["Abbreviation"] == "D2", "DNFProbability"].iloc[0] == 100.0


def test_driver_specific_dnf_probability_alignment_is_validated():
    simulator = MonteCarloSimulator(_config())
    target = _target().iloc[:2].copy()
    with pytest.raises(ValueError, match="one-dimensional vector"):
        simulator.simulate(target, np.array([1.0, 2.0]), driver_dnf_probs=np.array([0.5]), n_simulations=10)
    with pytest.raises(ValueError, match="ordering"):
        simulator.simulate(
            target,
            np.array([1.0, 2.0]),
            driver_dnf_probs=np.array([0.5, 0.5]),
            driver_dnf_driver_ids=["D2", "D1"],
            n_simulations=10,
        )
