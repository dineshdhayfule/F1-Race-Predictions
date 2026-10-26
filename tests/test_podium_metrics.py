"""
Podium Metrics & Probability Calibration Evaluation Suite (TC-12, TC-13, TC-14)
================================================================================
Evaluates chronological out-of-fold predictive performance for Formula 1 podium prediction.
Reuses production pipeline classes:
- DataLoader
- FeatureEngineer
- DataValidator
- ModelEvaluator & ModelTrainer
- Predictor
- MonteCarloSimulator

Validates:
1. TC-12: Chronological out-of-fold evaluation (Rounds 4-15), measuring exact podium match
   rate and set overlap without future-race leakage.
2. TC-13: Probability bounds ([0, 100]%), total Top 3 sum (~300%), monotonicity, and
   multi-driver Brier score compared against naive uninformative baseline.
3. TC-14: Dual baseline comparison (Actual Starting Grid post-penalties vs Raw Qualifying pre-penalties).

Software correctness is verified via strict structural assertions.
Predictive outcomes (overlap, Brier score) are measured and recorded as empirical performance findings.
Zero files in outputs/ are touched.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.data_loader import DataLoader
from src.data.data_validator import DataValidator
from src.features.build_features import FeatureEngineer
from src.models.evaluate import ModelEvaluator
from src.models.train import ModelTrainer
from src.models.predict import Predictor
from src.simulation.monte_carlo import MonteCarloSimulator

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def oof_podium_results():
    """
    Executes chronological out-of-fold predictions across completed 2026 rounds 4 through 15
    reusing the exact production pipeline components and configurations.
    """
    config_dir = ROOT / "config"
    with open(config_dir / "tracks.json", "r", encoding="utf-8") as f:
        tracks_cfg = json.load(f).get("tracks", {})
    with open(config_dir / "model_config.json", "r", encoding="utf-8") as f:
        model_cfg = json.load(f)

    data_dir = ROOT / "data" / "raw"
    loader = DataLoader(root_dir=ROOT)
    validator = DataValidator()
    feature_engineer = FeatureEngineer(
        ewma_alpha=model_cfg.get("feature_engineering", {}).get("ewma_alpha", 0.3)
    )
    model_trainer = ModelTrainer(model_cfg)
    model_evaluator = ModelEvaluator(model_cfg)
    predictor = Predictor(model_cfg)
    simulator = MonteCarloSimulator(model_cfg)

    # Invert round -> track key mapping for 2026
    round_to_track = {cfg.get("round_2026"): k for k, cfg in tracks_cfg.items() if "round_2026" in cfg}

    evaluated_rounds = list(range(4, 16))  # Rounds 4 through 15 (12 rounds)
    fold_records = []

    for target_round in evaluated_rounds:
        track_key = round_to_track.get(target_round, "unknown")
        track_info = tracks_cfg.get(track_key, {})
        track_name = track_info.get("display_name", track_key.title())
        penalties = track_info.get("known_penalties_2026", {})

        # 1. Strictly prior completed rounds
        train_rounds = [r for r in range(1, target_round)]
        races_df, qualis_df = loader.load_season_rounds(2026, train_rounds)
        sessions_by_round = loader.load_all_practice_sessions(2026, train_rounds)

        # 2. Weekend sessions for target round
        # Look in track-specific folder if available, else load from season/2026/ files
        weekend_sessions = loader.load_weekend_sessions(track_key, 2026, stage="after_qualifying")
        if all(v is None for v in weekend_sessions.values()):
            # Fall back to season/2026 round files for historical rounds 4-14
            is_sprint = track_info.get("sprint_weekend_2026", False)
            season_round_sessions = loader.load_all_practice_sessions(2026, [target_round]).get(target_round, {})
            weekend_sessions = season_round_sessions

        # Target qualifying
        target_quali = loader.load_target_qualifying(track_key, 2026)
        if target_quali is None or target_quali.empty:
            # Fallback to season round qualifying
            _, q_df = loader.load_season_rounds(2026, [target_round])
            target_quali = q_df[q_df["Round"] == target_round].copy() if not q_df.empty else None

        assert target_quali is not None and not target_quali.empty, f"Target qualifying data missing for round {target_round}"

        # 3. Build training features strictly using prior rounds
        training_features = feature_engineer.build_training_dataset(
            races_df=races_df,
            qualis_df=qualis_df,
            practice_sessions_by_round=sessions_by_round,
            train_rounds=train_rounds
        )

        # 4. Build target features strictly pre-race (no race outcome)
        target_features = feature_engineer.build_target_dataset(
            target_round=target_round,
            track_name=track_name,
            quali_df=target_quali,
            practice_sessions=weekend_sessions,
            season_races_df=races_df,
            season_qualis_df=qualis_df,
            penalties=penalties
        )

        # 5. Strict leakage audit
        audit = validator.audit_leakage(training_features, target_features, target_round=target_round)
        assert audit["is_clean"] is True, f"Leakage detected on round {target_round}: {audit['violations']}"

        # 6. Model evaluation and selection on prior rounds
        sel_cfg = model_cfg.get("model_selection", {})
        mode = sel_cfg.get("mode", "fixed_ensemble")
        mc_cfg = model_cfg.get("monte_carlo", {})

        if target_round > 4:
            eval_results = model_evaluator.evaluate_walk_forward(training_features)
            if mode == "fixed_ensemble":
                w_rf = float(sel_cfg.get("fixed_rf_weight", 0.6))
                w_xgb = float(sel_cfg.get("fixed_xgb_weight", 0.4))
                residuals = eval_results.get("selected_residuals", np.array([0.0]))
            else:
                w_rf = eval_results["best_w_rf"]
                w_xgb = eval_results["best_w_xgb"]
                residuals = eval_results["selected_residuals"]
            empirical_dnf = eval_results.get("empirical_dnf_rate", float(mc_cfg.get("fallback_dnf_prob", 0.158)))
        else:
            # Fallback for Round 4 prior to walk-forward validation fold availability
            w_rf = float(sel_cfg.get("fixed_rf_weight", 0.6))
            w_xgb = float(sel_cfg.get("fixed_xgb_weight", 0.4))
            residuals = None  # Triggers simulator fallback_sigma from config
            empirical_dnf = float(mc_cfg.get("fallback_dnf_prob", 0.158))

        # 7. Train final estimators on prior season rounds and predict
        imputer, rf_model, xgb_model, _ = model_trainer.fit(training_features)

        # Drivers pre-sorted strictly P1->P22 by ActualStartingGrid
        target_features = target_features.sort_values("ActualStartingGrid").reset_index(drop=True)

        _, blended_scores = predictor.predict_deterministic(
            target_df=target_features,
            imputer=imputer,
            rf_model=rf_model,
            xgb_model=xgb_model,
            rf_weight=w_rf,
            xgb_weight=w_xgb
        )

        # 8. Monte Carlo Simulation (25,000 iterations, seed=42)
        sim_df, _ = simulator.simulate(
            target_df=target_features,
            deterministic_scores=blended_scores,
            residuals=residuals,
            empirical_dnf_rate=empirical_dnf
        )

        # 9. Load actual official race results for ground truth comparison
        actual_race_df, _ = loader.load_season_rounds(2026, [target_round])
        actual_round_race = actual_race_df[actual_race_df["Round"] == target_round].sort_values("Position").reset_index(drop=True)

        fold_records.append({
            "round": target_round,
            "track_key": track_key,
            "track_name": track_name,
            "predictions": sim_df,
            "actual": actual_round_race,
            "target_features": target_features
        })

    return fold_records


def test_out_of_fold_podium_metrics_evaluation(oof_podium_results):
    """
    TC-12: Evaluates exact podium match rate and set overlap across completed 2026 rounds 4-15.
    Asserts software execution integrity and records empirical performance metrics.
    """
    assert len(oof_podium_results) == 12, f"Expected 12 evaluated folds, got {len(oof_podium_results)}"

    exact_matches = []
    overlap_counts = []

    for fold in oof_podium_results:
        pred_top3 = fold["predictions"].iloc[:3]["Abbreviation"].tolist()
        actual_top3 = fold["actual"].iloc[:3]["Abbreviation"].tolist()

        # Exact podium match (all 3 in exact order P1, P2, P3)
        is_exact = (pred_top3 == actual_top3)
        exact_matches.append(is_exact)

        # Set overlap count (how many predicted top 3 actually finished on the podium)
        overlap = len(set(pred_top3) & set(actual_top3))
        overlap_counts.append(overlap)

    # Software correctness assertions
    assert len(overlap_counts) == 12
    for k in overlap_counts:
        assert k in (0, 1, 2, 3), f"Overlap count {k} is outside discrete set {{0, 1, 2, 3}}"

    # Empirical performance reporting (recorded findings, not hard pass/fail thresholds)
    mean_overlap = float(np.mean(overlap_counts))
    exact_match_rate = float(np.mean(exact_matches)) * 100.0
    macro_f1 = float(np.mean([k / 3.0 for k in overlap_counts]))

    print(f"\n[TC-12 PERFORMANCE METRIC] Evaluated Rounds: {len(oof_podium_results)} (Rounds 4-15)")
    print(f"[TC-12 PERFORMANCE METRIC] Mean Podium Overlap K: {mean_overlap:.2f} / 3.00")
    print(f"[TC-12 PERFORMANCE METRIC] Macro Podium F1: {macro_f1:.3f}")
    print(f"[TC-12 PERFORMANCE METRIC] Exact Podium Match Rate: {exact_match_rate:.1f}%")
    print(f"[TC-12 PERFORMANCE METRIC] Overlap Distribution: K=0: {overlap_counts.count(0)}, K=1: {overlap_counts.count(1)}, K=2: {overlap_counts.count(2)}, K=3: {overlap_counts.count(3)}")


def test_podium_probability_mathematics_and_brier_score(oof_podium_results):
    """
    TC-13: Validates probability bounds, 300% sum constraint, monotonicity,
    and computes the out-of-fold multi-driver Podium Brier Score.
    """
    brier_scores = []
    naive_brier_scores = []

    for fold in oof_podium_results:
        sim_df = fold["predictions"]
        actual_df = fold["actual"].set_index("Abbreviation")["Position"].to_dict()

        # 1. Mathematical bounds: every probability must lie in [0.0, 100.0]%
        for p_col in ["WinProbability", "Top3Probability", "Top5Probability", "PointsProbability"]:
            assert (sim_df[p_col] >= 0.0).all(), f"Negative probability in {p_col}"
            assert (sim_df[p_col] <= 100.0).all(), f"Probability > 100% in {p_col}"

        # 2. Total Top 3 probability sum must equal 300.0% within precision tolerance (+-0.10% due to 2-decimal individual rounding)
        top3_sum = sim_df["Top3Probability"].sum()
        assert abs(top3_sum - 300.0) <= 0.15, f"Top3 probability sum {top3_sum}% diverges from 300.0%"

        win_sum = sim_df["WinProbability"].sum()
        assert abs(win_sum - 100.0) <= 0.15, f"Win probability sum {win_sum}% diverges from 100.0%"

        # 3. Monotonicity: WinProbability <= Top3Probability <= PointsProbability
        # (Allows 0.01 tolerance for any display float rounding)
        monotonic = (sim_df["WinProbability"] <= sim_df["Top3Probability"] + 0.01) & \
                    (sim_df["Top3Probability"] <= sim_df["PointsProbability"] + 0.01)
        assert monotonic.all(), "Probability monotonicity (Win <= Podium <= Points) violated!"

        # 4. Multi-Driver Podium Brier Score calculation
        # Ground truth indicator Y_i = 1 if driver finished in top 3, else 0
        p_podium_scaled = sim_df["Top3Probability"].values / 100.0
        y_true = np.array([1.0 if actual_df.get(drv, 99) <= 3 else 0.0 for drv in sim_df["Abbreviation"]])

        round_brier = float(np.mean((p_podium_scaled - y_true) ** 2))
        brier_scores.append(round_brier)

        # Naive uninformative baseline: every driver assigned p = 3/22
        naive_p = np.full(len(sim_df), 3.0 / len(sim_df))
        round_naive_brier = float(np.mean((naive_p - y_true) ** 2))
        naive_brier_scores.append(round_naive_brier)

    mean_brier = float(np.mean(brier_scores))
    mean_naive_brier = float(np.mean(naive_brier_scores))
    brier_skill_score = 1.0 - (mean_brier / mean_naive_brier)

    print(f"\n[TC-13 PERFORMANCE METRIC] Out-of-Fold Mean Podium Brier Score: {mean_brier:.4f}")
    print(f"[TC-13 PERFORMANCE METRIC] Naive Uninformative Baseline:       {mean_naive_brier:.4f}")
    print(f"[TC-13 PERFORMANCE METRIC] Brier Skill Score (BSS):             {brier_skill_score:.4f}")


def test_dual_baseline_comparison_grid_vs_qualifying(oof_podium_results):
    """
    TC-14: Compares ML predicted Top 3 against two distinct non-ML benchmarks:
    1. Actual Starting Grid Baseline: Top 3 drivers starting P1, P2, P3 after all penalties.
    2. Raw Qualifying Baseline: Top 3 drivers qualifying Q1, Q2, Q3 before grid penalties.
    """
    ml_overlaps = []
    grid_overlaps = []
    quali_overlaps = []

    ml_exacts = []
    grid_exacts = []
    quali_exacts = []

    for fold in oof_podium_results:
        sim_df = fold["predictions"]
        actual_top3 = fold["actual"].iloc[:3]["Abbreviation"].tolist()
        t_feats = fold["target_features"]

        # 1. ML Predicted Top 3
        ml_top3 = sim_df.iloc[:3]["Abbreviation"].tolist()
        ml_overlaps.append(len(set(ml_top3) & set(actual_top3)))
        ml_exacts.append(ml_top3 == actual_top3)

        # 2. Actual Starting Grid Baseline (after known penalties)
        grid_sorted = t_feats.sort_values("ActualStartingGrid").reset_index(drop=True)
        grid_top3 = grid_sorted.iloc[:3]["Abbreviation"].tolist()
        grid_overlaps.append(len(set(grid_top3) & set(actual_top3)))
        grid_exacts.append(grid_top3 == actual_top3)

        # 3. Raw Qualifying Baseline (before penalties)
        # Handle cases where qualifying position exists
        pos_col = "QualifyingPosition" if "QualifyingPosition" in t_feats.columns else "Position"
        quali_sorted = t_feats.sort_values(pos_col).reset_index(drop=True)
        quali_top3 = quali_sorted.iloc[:3]["Abbreviation"].tolist()
        quali_overlaps.append(len(set(quali_top3) & set(actual_top3)))
        quali_exacts.append(quali_top3 == actual_top3)

    # Software correctness assertions
    assert len(grid_overlaps) == 12
    assert len(quali_overlaps) == 12

    # Empirical benchmark comparison reporting
    print("\n" + "=" * 80)
    print("TC-14 DUAL BASELINE COMPARISON SUMMARY (12 Out-of-Fold Rounds: Rounds 4-15)")
    print("=" * 80)
    print(f"{'Approach':<30} {'Mean Overlap (K)':<18} {'Exact 1-2-3 Match Rate'}")
    print("-" * 80)
    print(f"{'ML Engine (RF+XGB+MC)':<30} {np.mean(ml_overlaps):<18.2f} {np.mean(ml_exacts)*100:.1f}%")
    print(f"{'Actual Starting Grid Top 3':<30} {np.mean(grid_overlaps):<18.2f} {np.mean(grid_exacts)*100:.1f}%")
    print(f"{'Raw Qualifying Top 3':<30} {np.mean(quali_overlaps):<18.2f} {np.mean(quali_exacts)*100:.1f}%")
    print("=" * 80)
