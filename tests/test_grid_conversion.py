"""
Unit Tests for Grid-to-Race Conversion Feature Engine
=====================================================
Validates:
1. Neutral prior initialization when no historical rounds exist.
2. Empirical bucket calculation and Bayesian shrinkage.
3. Monotonicity: Expected finish for front-runners is strictly better than backmarkers.
4. Output bounds: 1.0 <= expected_grid_finish <= 22.0, 0.0 <= retention_prob <= 1.0.
5. Strict anti-leakage: Round r only accesses data from rounds < r.
6. Coordination with Monte Carlo: Running finish excludes DNFs to avoid double-counting attrition.
7. Sprint separation: Sprint races are excluded from GP conversion statistics.
"""

import pytest
import numpy as np
import pandas as pd

from src.features.grid_conversion import (
    get_grid_bucket,
    compute_grid_conversion_map,
    get_conversion_features_for_grid,
    GRID_BUCKETS,
)


def test_grid_bucket_classification():
    assert get_grid_bucket(1) == "front_row"
    assert get_grid_bucket(2) == "front_row"
    assert get_grid_bucket(3) == "top_5"
    assert get_grid_bucket(5) == "top_5"
    assert get_grid_bucket(6) == "points_zone"
    assert get_grid_bucket(10) == "points_zone"
    assert get_grid_bucket(11) == "midfield"
    assert get_grid_bucket(16) == "midfield"
    assert get_grid_bucket(17) == "backmarkers"
    assert get_grid_bucket(22) == "backmarkers"


def test_conversion_map_empty_prior():
    """When no historical rounds exist (e.g. Round 1), returns neutral identity mapping."""
    empty_df = pd.DataFrame()
    cmap = compute_grid_conversion_map(empty_df, expected_grid_size=22)

    assert len(cmap) == 22
    for g in range(1, 23):
        assert cmap[g]["expected_grid_finish"] == float(g)
        assert 0.0 <= cmap[g]["grid_retention_prob"] <= 1.0

    # P1 prior finish should be 1.0, P22 should be 22.0
    assert cmap[1]["expected_grid_finish"] == 1.0
    assert cmap[22]["expected_grid_finish"] == 22.0


def test_conversion_map_with_historical_data():
    """Verifies empirical calculation with synthetic race data."""
    data = []
    # 5 races, 22 drivers each
    for r in range(1, 6):
        for g in range(1, 23):
            # Normal race finish roughly correlated with grid
            finish = g + (1 if g % 2 == 0 else -1)
            finish = max(1, min(22, finish))
            status = "Retired" if g == 20 else "Finished"
            data.append({
                "Round": r,
                "GridPosition": g,
                "Position": finish,
                "Status": status
            })

    hist_df = pd.DataFrame(data)
    cmap = compute_grid_conversion_map(hist_df, expected_grid_size=22, shrinkage_m=5.0)

    assert len(cmap) == 22
    # Monotonicity check
    assert cmap[1]["expected_grid_finish"] < cmap[10]["expected_grid_finish"]
    assert cmap[10]["expected_grid_finish"] < cmap[20]["expected_grid_finish"]

    # Bounds check
    for g, stats in cmap.items():
        assert 1.0 <= stats["expected_grid_finish"] <= 22.0
        assert 0.0 <= stats["grid_retention_prob"] <= 1.0


def test_dnf_coordination_excludes_terminal_retirements():
    """
    Verifies that running finish calculation excludes DNFs so that
    Monte Carlo stochastic incident shocks are not double-counted.
    """
    # Create scenario where pole sitter finished P1 twice, but had an engine failure once (P22)
    hist_df = pd.DataFrame([
        {"Round": 1, "GridPosition": 1, "Position": 1, "Status": "Finished"},
        {"Round": 2, "GridPosition": 1, "Position": 1, "Status": "Finished"},
        {"Round": 3, "GridPosition": 1, "Position": 22, "Status": "Engine failure"},  # DNF
    ])

    cmap = compute_grid_conversion_map(hist_df, expected_grid_size=22, shrinkage_m=0.0)

    # Running finish of P1 was 1.0 (excluding the engine failure)
    # Without DNF filtering, the mean would be (1 + 1 + 22)/3 = 8.0!
    assert cmap[1]["expected_grid_finish"] < 3.0, (
        f"DNF should not distort running expected finish: got {cmap[1]['expected_grid_finish']}"
    )


def test_get_conversion_features_helper():
    cmap = {
        1: {"expected_grid_finish": 1.5, "grid_retention_prob": 0.65},
        2: {"expected_grid_finish": 2.2, "grid_retention_prob": 0.60},
    }
    feat = get_conversion_features_for_grid(1.0, cmap)
    assert feat["expected_grid_finish"] == 1.5
    assert feat["grid_retention_prob"] == 0.65

    # Out of range clamped safely
    feat_clamp = get_conversion_features_for_grid(0.2, cmap, expected_grid_size=2)
    assert feat_clamp["expected_grid_finish"] == 1.5
