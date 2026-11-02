"""
Regression Tests for Monte Carlo Entrant Integrity & Non-Classified Handling
============================================================================
Verifies that:
1. Entrant count is strictly preserved across all simulation paths (active + DNS/DSQ).
2. DNS and DSQ entrants retain StartingGridStatus, have NaN finish ranks and 0% win prob.
3. In-race DNFs who started remain actively simulated.
4. _audit_prediction_integrity verifies field integrity without falsely rejecting non-starters.
"""

import numpy as np
import pandas as pd
import pytest
from src.simulation.monte_carlo import MonteCarloSimulator
from src.pipeline.race_prediction_pipeline import RacePredictionPipeline


def _build_dummy_target_df(n_drivers: int = 22, dns_list: list = None, dsq_list: list = None):
    drivers = [f"D{i:02d}" for i in range(1, n_drivers + 1)]
    dns_set = set(dns_list or [])
    dsq_set = set(dsq_list or [])

    statuses = []
    for d in drivers:
        if d in dns_set:
            statuses.append("DNS")
        elif d in dsq_set:
            statuses.append("DSQ")
        else:
            statuses.append("CLASSIFIED")

    df = pd.DataFrame({
        "Abbreviation": drivers,
        "TeamName": [f"Team_{i%10}" for i in range(n_drivers)],
        "QualifyingPosition": list(range(1, n_drivers + 1)),
        "GridPenalty": [0] * n_drivers,
        "grid_penalty_places": [0] * n_drivers,
        "ActualStartingGrid": list(range(1, n_drivers + 1)),
        "simulation_start_rank": list(range(1, n_drivers + 1)),
        "is_pit_lane_start": [0] * n_drivers,
        "StartingGridStatus": statuses,
        "p_safety_car": [0.5] * n_drivers,
        "p_vsc": [0.4] * n_drivers,
        "p_red_flag": [0.15] * n_drivers,
        "rain_probability": [0.1] * n_drivers,
    })
    return df


def _get_simulator(n_sims=500):
    return MonteCarloSimulator({
        "simulation": {"default_simulations": n_sims},
        "random_seed": 42,
    })


def test_standard_full_field_simulation():
    mc = _get_simulator(500)
    target_df = _build_dummy_target_df(22)
    scores = np.linspace(1, 22, 22)

    res, summary = mc.simulate(target_df, scores)

    assert len(res) == 22
    assert summary["driver_count"] == 22
    assert summary["active_driver_count"] == 22
    assert sorted(res["PredictedPosition"].tolist()) == list(range(1, 23))
    assert res["ExpectedFinish"].isna().sum() == 0
    assert (res["WinProbability"] >= 0).all() and (res["WinProbability"] <= 100).all()


def test_field_with_one_dns_preserves_total_count():
    mc = _get_simulator(500)
    target_df = _build_dummy_target_df(22, dns_list=["D05"])
    scores = np.linspace(1, 22, 22)

    res, summary = mc.simulate(target_df, scores)

    assert len(res) == 22, f"Expected 22 entrants, found {len(res)}"
    assert summary["driver_count"] == 22
    assert summary["active_driver_count"] == 21

    # D05 must be present as DNS
    d05 = res[res["Abbreviation"] == "D05"].iloc[0]
    assert d05["StartingGridStatus"] == "DNS"
    assert pd.isna(d05["PredictedPosition"])
    assert pd.isna(d05["ExpectedFinish"])
    assert d05["WinProbability"] == 0.0

    # Active 21 drivers must have contiguous ranks 1..21
    active_ranks = res[res["StartingGridStatus"] == "CLASSIFIED"]["PredictedPosition"].tolist()
    assert sorted(active_ranks) == list(range(1, 22))


def test_field_with_multiple_non_starters():
    mc = _get_simulator(500)
    target_df = _build_dummy_target_df(22, dns_list=["D02"], dsq_list=["D18"])
    scores = np.linspace(1, 22, 22)

    res, summary = mc.simulate(target_df, scores)

    assert len(res) == 22
    assert summary["active_driver_count"] == 20

    excluded = res[res["StartingGridStatus"].isin(["DNS", "DSQ"])]
    assert len(excluded) == 2
    assert set(excluded["Abbreviation"]) == {"D02", "D18"}
    assert excluded["PredictedPosition"].isna().all()
    assert (excluded["WinProbability"] == 0.0).all()

    active = res[~res["StartingGridStatus"].isin(["DNS", "DSQ"])]
    assert len(active) == 20
    assert sorted(active["PredictedPosition"].tolist()) == list(range(1, 21))


def test_audit_prediction_integrity_accepts_dns_dsq_without_error():
    pipeline = RacePredictionPipeline.__new__(RacePredictionPipeline)

    mc = _get_simulator(200)
    target_df = _build_dummy_target_df(22, dns_list=["D03"], dsq_list=["D20"])
    scores = np.linspace(1, 22, 22)
    res, _ = mc.simulate(target_df, scores)

    # Should run with zero exceptions
    pipeline._audit_prediction_integrity(res, track_key="austin", year=2026, stage="after_qualifying")


def test_smaller_grid_size_preserved():
    # Test field with 20 drivers
    mc = _get_simulator(200)
    target_df = _build_dummy_target_df(20, dns_list=["D01"])
    scores = np.linspace(1, 20, 20)

    res, summary = mc.simulate(target_df, scores)
    assert len(res) == 20
    assert summary["active_driver_count"] == 19
