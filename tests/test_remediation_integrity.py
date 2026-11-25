"""Tests verifying the ML integrity and race prediction remediation (Phase 5).

Covers:
- F-02: Strategy penalty direction and monotonicity
- F-01: Pre-qualifying grid handling and provisional grid utilization
- F-07: Practice-session roster resolution and missing driver recovery
- F-05: Walk-forward validation parity with inference
- F-08: DNF evaluation, retirement accounting, and DNS separation
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.simulation.monte_carlo import MonteCarloSimulator
from src.simulation.strategy_simulator import PitStrategySimulator
from src.models.predict import Predictor
from src.features.build_features import apply_stage_feature_mask, FeatureEngineer
from src.data.roster_resolver import RosterResolver
from src.models.evaluate import ModelEvaluator
from src.pipeline.race_prediction_pipeline import RacePredictionPipeline


# =====================================================================
# F-02: Strategy Penalty Direction
# =====================================================================

class TestF02StrategyPenaltyDirection:
    """Verifies that strategy disadvantage properly worsens finish score."""

    @pytest.fixture
    def config(self):
        return {
            "random_seed": 42,
            "monte_carlo": {
                "default_simulations": 10000,
                "fallback_sigma": 2.5,
                "fallback_dnf_prob": 0.0,
            },
        }

    def test_zero_strategy_loss_invariance(self, config):
        """When strategy_time_loss_sec is 0.0, score remains at baseline."""
        sim = MonteCarloSimulator(config)
        target = pd.DataFrame({
            "Abbreviation": ["NOR", "VER"],
            "TeamName": ["McLaren", "Red Bull"],
            "GridPosition": [1, 2],
            "StartingGridStatus": ["CLASSIFIED", "CLASSIFIED"],
            "strategy_time_loss_sec": [0.0, 0.0],
            "is_pit_lane_start": [0, 0],
        })
        scores = np.array([2.0, 3.0])

        # Run simulation with 0.0 strategy loss
        res_zero, _ = sim.simulate(target, scores, n_simulations=5000)

        # Run simulation without strategy column
        target_no_strat = target.drop(columns=["strategy_time_loss_sec"])
        res_none, _ = sim.simulate(target_no_strat, scores, n_simulations=5000)

        # Expected finish and win probabilities must match
        pd.testing.assert_series_equal(
            res_zero["ExpectedFinish"],
            res_none["ExpectedFinish"],
            check_exact=False,
            rtol=1e-3,
        )
        pd.testing.assert_series_equal(
            res_zero["WinProbability"],
            res_none["WinProbability"],
            check_exact=False,
            rtol=1e-3,
        )

    def test_positive_strategy_loss_penalty_monotonicity(self, config):
        """A driver with positive strategy time loss must have a worse finish than identical driver with 0."""
        sim = MonteCarloSimulator(config)
        target = pd.DataFrame({
            "Abbreviation": ["NOR_A", "NOR_B"],
            "TeamName": ["McLaren", "McLaren"],
            "GridPosition": [1, 2],
            "StartingGridStatus": ["CLASSIFIED", "CLASSIFIED"],
            "strategy_time_loss_sec": [0.0, 15.0],  # NOR_B has 15s strategy disadvantage
            "is_pit_lane_start": [0, 0],
        })
        # Both start with identical baseline ML scores
        scores = np.array([2.0, 2.0])

        res, _ = sim.simulate(target, scores, n_simulations=10000)
        res = res.set_index("Abbreviation")

        # NOR_A (0s loss) must strictly outperform NOR_B (15s loss)
        assert res.loc["NOR_A", "WinProbability"] > res.loc["NOR_B", "WinProbability"]
        assert res.loc["NOR_A", "ExpectedFinish"] < res.loc["NOR_B", "ExpectedFinish"]

    def test_strategy_simulator_loss_is_non_negative(self):
        """PitStrategySimulator must compute non-negative loss relative to fastest grid strategy."""
        strat_sim = PitStrategySimulator()
        sample_target = pd.DataFrame({
            "Abbreviation": ["VER", "NOR"],
            "GridPosition": [1, 2],
            "driver_tyre_deg_rate": [0.08, 0.09],
            "driver_overtake_ability": [0.85, 0.80],
        })
        res = strat_sim.simulate_grid_strategies(sample_target, total_laps=57, circuit_key="bahrain")
        assert len(res) == 2
        for _, r in res.iterrows():
            assert r["strategy_time_loss_sec"] >= 0.0
            assert "recommended_strategy" in r


# =====================================================================
# F-01: Pre-Qualifying Grid Handling
# =====================================================================

class TestF01PreQualifyingGridHandling:
    """Verifies eliminating P12 constant fallback and handling provisional grid."""

    def test_pre_qualifying_stage_masks_official_grid(self):
        """Pre-qualifying stages must mask GridPosition to prevent leakage."""
        df = pd.DataFrame({
            "Driver": ["VER", "NOR"],
            "GridPosition": [1, 2],
            "ActualStartingGrid": [1, 2],
            "FP1_LapTime_sec": [84.1, 84.3],
        })
        masked = apply_stage_feature_mask(df.copy(), "pre_fp1")
        assert masked["GridPosition"].isna().all()
        assert masked["ActualStartingGrid"].isna().all()

        masked_fp3 = apply_stage_feature_mask(df.copy(), "after_fp3")
        assert masked_fp3["GridPosition"].isna().all()
        assert masked_fp3["ActualStartingGrid"].isna().all()

    def test_predict_deterministic_disables_grid_when_no_grid_available(self):
        """When no grid exists, predict_deterministic must NOT assign P12 to all drivers."""
        target_df = pd.DataFrame({
            "Abbreviation": ["VER", "NOR", "LEC", "HAM"],
            "TeamName": ["Red Bull Racing", "McLaren", "Ferrari", "Mercedes"],
            "GridPosition": [np.nan, np.nan, np.nan, np.nan],
            "ActualStartingGrid": [np.nan, np.nan, np.nan, np.nan],
            "feat1": [1.0, 2.0, 3.0, 4.0],
        })
        ml_scores = np.array([1.2, 2.5, 4.0, 5.5])

        class DummyModel:
            def predict(self, X):
                return ml_scores

        class DummyImputer:
            def transform(self, X):
                return X.values

        predictor = Predictor({"feature_columns": ["feat1"]})
        results_df, blended = predictor.predict_deterministic(
            target_df=target_df,
            imputer=DummyImputer(),
            rf_model=DummyModel(),
            circuit_type="street",
        )

        # Predictions must reflect ML rank, not collapsed to constant 12.0
        assert not np.allclose(blended, 12.0)
        # Because eff_grid_w = 0, deterministic prediction should match ml_scores exactly
        np.testing.assert_allclose(blended, ml_scores)

    def test_predict_deterministic_uses_provisional_grid_when_available(self):
        """When provisional_grid_position is present pre-qualifying, it is legitimately used."""
        target_df = pd.DataFrame({
            "Abbreviation": ["VER", "NOR"],
            "TeamName": ["Red Bull Racing", "McLaren"],
            "GridPosition": [np.nan, np.nan],
            "ActualStartingGrid": [np.nan, np.nan],
            "provisional_grid_position": [1.0, 2.0],
            "feat1": [1.0, 2.0],
        })
        ml_scores = np.array([2.0, 1.0])

        class DummyModel:
            def predict(self, X):
                return ml_scores

        class DummyImputer:
            def transform(self, X):
                return X.values

        predictor = Predictor({"feature_columns": ["feat1"]})
        results_df, blended = predictor.predict_deterministic(
            target_df=target_df,
            imputer=DummyImputer(),
            rf_model=DummyModel(),
            circuit_type="permanent",
        )
        # Predictions blend provisional grid with ML score, neither is 12.0
        assert not np.any(blended == 12.0)
        assert len(blended) == 2


# =====================================================================
# F-07: Practice-Session Roster Resolution
# =====================================================================

class TestF07PracticeRosterResolution:
    """Verifies that drivers missing practice are recovered without keeping inappropriate reserves."""

    def test_missing_fp2_driver_recovered_from_fp1(self):
        """When a driver (e.g. LEC) misses FP2, they are recovered from FP1."""
        resolver = RosterResolver()

        fp1_drivers = [
            ("VER", "Red Bull Racing"), ("PER", "Red Bull Racing"),
            ("NOR", "McLaren"), ("PIA", "McLaren"),
            ("HAM", "Mercedes"), ("RUS", "Mercedes"),
            ("LEC", "Ferrari"), ("SAI", "Ferrari"),
            ("ALO", "Aston Martin"), ("STR", "Aston Martin"),
            ("GAS", "Alpine"), ("OCO", "Alpine"),
            ("ALB", "Williams"), ("SAR", "Williams"),
            ("TSU", "RB"), ("RIC", "RB"),
            ("BOT", "Kick Sauber"), ("ZHO", "Kick Sauber"),
            ("MAG", "Haas F1 Team"), ("HUL", "Haas F1 Team"),
        ]
        # Simulate FP2 with LEC missing due to mechanical trouble (19 drivers)
        fp2_drivers = [d for d in fp1_drivers if d[0] != "LEC"]

        practice_sessions = {
            "fp1": pd.DataFrame({
                "Abbreviation": [d[0] for d in fp1_drivers],
                "TeamName": [d[1] for d in fp1_drivers],
            }),
            "fp2": pd.DataFrame({
                "Abbreviation": [d[0] for d in fp2_drivers],
                "TeamName": [d[1] for d in fp2_drivers],
            }),
        }

        roster = resolver._extract_from_practice_sessions(
            weekend_sessions=practice_sessions,
            stage="after_fp2",
            year=2026,
            expected_grid_size=20,
        )

        assert len(roster) == 20
        assert "LEC" in roster["Abbreviation"].values
        fer_drivers = roster[roster["TeamName"] == "Ferrari"]["Abbreviation"].tolist()
        assert "LEC" in fer_drivers
        assert "SAI" in fer_drivers

    def test_fp1_reserve_driver_not_kept_if_both_regulars_in_fp2(self):
        """An FP1 rookie reserve (e.g. BEA) is not kept if regular drivers run in FP2."""
        resolver = RosterResolver()

        # FP1 had BEA driving instead of HAM for Mercedes
        fp1_drivers = [
            ("VER", "Red Bull Racing"), ("PER", "Red Bull Racing"),
            ("NOR", "McLaren"), ("PIA", "McLaren"),
            ("BEA", "Mercedes"), ("RUS", "Mercedes"),
            ("LEC", "Ferrari"), ("SAI", "Ferrari"),
        ]
        # FP2 had regular drivers HAM and RUS
        fp2_drivers = [
            ("VER", "Red Bull Racing"), ("PER", "Red Bull Racing"),
            ("NOR", "McLaren"), ("PIA", "McLaren"),
            ("HAM", "Mercedes"), ("RUS", "Mercedes"),
            ("LEC", "Ferrari"), ("SAI", "Ferrari"),
        ]

        practice_sessions = {
            "fp1": pd.DataFrame({
                "Abbreviation": [d[0] for d in fp1_drivers],
                "TeamName": [d[1] for d in fp1_drivers],
            }),
            "fp2": pd.DataFrame({
                "Abbreviation": [d[0] for d in fp2_drivers],
                "TeamName": [d[1] for d in fp2_drivers],
            }),
        }

        roster = resolver._extract_from_practice_sessions(
            weekend_sessions=practice_sessions,
            stage="after_fp2",
            year=2026,
            expected_grid_size=8,
        )

        assert len(roster) == 8
        assert "HAM" in roster["Abbreviation"].values
        assert "RUS" in roster["Abbreviation"].values
        assert "BEA" not in roster["Abbreviation"].values


# =====================================================================
# F-05: Walk-Forward Validation Parity
# =====================================================================

class TestF05WalkForwardParity:
    """Verifies walk-forward validation applies stage-appropriate grid blending matching inference."""

    def test_post_qualifying_validation_blends_grid(self):
        """ModelEvaluator applies circuit-aware grid blending on out-of-fold predictions."""
        config = {
            "validation": {"min_train_rounds": 1},
            "features": ["feat1", "feat2"],
            "models": {
                "rf": {"n_estimators": 5, "max_depth": 3},
                "xgb": {"n_estimators": 5, "max_depth": 3},
            },
            "random_seed": 42,
        }
        evaluator = ModelEvaluator(config)

        # Create synthetic multi-round training dataframe with Round, GridPosition, ActualFinish
        df = pd.DataFrame({
            "Round": [1] * 10 + [2] * 10,
            "Driver": [f"D{i}" for i in range(10)] * 2,
            "Abbreviation": [f"D{i}" for i in range(10)] * 2,
            "GridPosition": list(range(1, 11)) * 2,
            "ActualStartingGrid": list(range(1, 11)) * 2,
            "circuit_type": ["street"] * 20,
            "feat1": np.linspace(1, 10, 20),
            "feat2": np.linspace(10, 1, 20),
            "ActualFinish": list(range(1, 11)) * 2,
        })

        metrics = evaluator.evaluate_walk_forward(
            training_df=df,
            stage="after_qualifying",
            apply_grid_blending=True,
        )

        assert "ensemble_comparisons" in metrics
        assert len(metrics["ensemble_comparisons"]) > 0
        assert metrics["best_mae"] is not None


# =====================================================================
# F-08: DNF Evaluation and Post-Race Integrity
# =====================================================================

class TestF08DNFEvaluationIntegrity:
    """Verifies that race evaluations separate classified finishers from retirements and DNS."""

    def test_evaluate_race_predictions_metrics(self):
        """Pipeline computes MAE on classified finishers and DNF Brier score on starters."""
        pipe = RacePredictionPipeline()

        sim_results = pd.DataFrame({
            "Abbreviation": ["VER", "NOR", "LEC", "HAM", "GAS"],
            "TeamName": ["Red Bull", "McLaren", "Ferrari", "Mercedes", "Alpine"],
            "PredictedPosition": [1, 2, 3, 4, 5],
            "ExpectedFinish": [1.2, 2.3, 3.1, 4.4, 5.2],
            "Top3Probability": [90.0, 80.0, 70.0, 40.0, 20.0],
            "DNFProbability": [5.0, 8.0, 10.0, 90.0, 50.0],
        })

        # Actual race: VER, NOR finish 1-2. LEC retires (DNF). HAM finishes 3. GAS does not start (DNS).
        actual_race_df = pd.DataFrame({
            "Abbreviation": ["VER", "NOR", "LEC", "HAM", "GAS"],
            "Position": ["1", "2", "R", "3", "DNS"],
            "Status": ["Finished", "Finished", "Engine", "Finished", "Did not start"],
            "Points": [25, 18, 0, 15, 0],
        })

        metrics = pipe._evaluate_race_predictions(sim_results, actual_race_df)

        # Total starters = 4 (VER, NOR, LEC, HAM; GAS is DNS)
        assert metrics["starters_count"] == 4
        assert metrics["classified_count"] == 3
        assert metrics["retirements_count"] == 1
        assert metrics["dns_count"] == 1

        # MAE is computed on the 3 classified finishers
        assert pd.notna(metrics["mae_expected"])
        assert pd.notna(metrics["mae_predicted"])
        assert metrics["evaluated_drivers"] == 5

        # DNF Brier score and accuracy computed on starters
        assert pd.notna(metrics["dnf_brier_score"])
        assert pd.notna(metrics["dnf_accuracy"])

        # Retiring top predicted identified
        assert "LEC" in metrics["retiring_top_predicted"]
