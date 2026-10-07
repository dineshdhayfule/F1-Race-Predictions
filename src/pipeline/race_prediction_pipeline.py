"""
Master Race Prediction Pipeline Orchestrator
============================================
Coordinates data loading, leakage validation, feature engineering, model training,
walk-forward cross-validation, and Monte Carlo simulation for any F1 Grand Prix.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import numpy as np
import pandas as pd

from src.data.data_loader import DataLoader
from src.data.data_validator import DataValidator
from src.features.build_features import FeatureEngineer
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator
from src.models.predict import Predictor
from src.simulation.monte_carlo import MonteCarloSimulator
from src.utils.logging_utils import get_logger

logger = get_logger("pipeline")


class RacePredictionPipeline:
    """Config-driven, multi-track race prediction pipeline."""

    def __init__(self, root_dir: Optional[Path] = None):
        self.root_dir = root_dir or Path(__file__).resolve().parents[2]
        self.config_dir = self.root_dir / "config"

        # Load configurations
        with open(self.config_dir / "tracks.json", "r", encoding="utf-8") as f:
            self.tracks_config = json.load(f)

        with open(self.config_dir / "model_config.json", "r", encoding="utf-8") as f:
            self.model_config = json.load(f)

        with open(self.config_dir / "settings.json", "r", encoding="utf-8") as f:
            self.settings = json.load(f)

        self.data_dir = self.root_dir / self.settings.get("paths", {}).get("data_raw", "data/raw")
        self.output_dir = self.root_dir / self.settings.get("paths", {}).get("outputs", "outputs")

        self.data_loader = DataLoader(self.data_dir)
        self.data_validator = DataValidator()
        self.feature_engineer = FeatureEngineer(
            ewma_alpha=self.model_config.get("feature_engineering", {}).get("ewma_alpha", 0.3)
        )
        self.model_trainer = ModelTrainer(self.model_config)
        self.model_evaluator = ModelEvaluator(self.model_config)
        self.predictor = Predictor(self.model_config)
        self.simulator = MonteCarloSimulator(self.model_config)

    def run(
        self,
        track_key: str,
        year: int = 2026,
        stage: str = "after_qualifying"
    ) -> Dict[str, Any]:
        """
        Executes end-to-end race prediction for the specified track.

        Parameters
        ----------
        track_key : str
            Identifier in tracks.json (e.g. 'sepang', 'baku', 'monza').
        year : int
            Championship season year.
        stage : str
            Prediction stage cutoff ('pre_fp1', 'after_fp1', 'after_fp2', 'after_fp3', 'after_qualifying').

        Returns
        -------
        dict containing prediction results, evaluation metrics, and simulation stats.
        """
        track_key = track_key.lower().strip()
        tracks = self.tracks_config.get("tracks", {})
        if track_key not in tracks:
            raise KeyError(f"Track '{track_key}' not found in tracks.json. Available: {list(tracks.keys())}")

        track_info = tracks[track_key]
        target_round = track_info.get("round_2026", 16)
        track_name = track_info.get("display_name", track_info.get("name", track_key.title()))
        penalties = track_info.get("known_penalties_2026", track_info.get("penalties_2026", {}))


        logger.info("=" * 80)
        logger.info(f"STARTING PREDICTION PIPELINE: {track_name.upper()} {year} (ROUND {target_round})")
        logger.info(f"STAGE: {stage}")
        logger.info("=" * 80)

        # 1. Load Season Data (completed rounds strictly < target_round)
        train_rounds = [r for r in range(1, target_round)]
        races_df, qualis_df = self.data_loader.load_season_rounds(year, train_rounds)
        sessions_by_round = self.data_loader.load_all_practice_sessions(year, train_rounds)

        # 2. Load Weekend Sessions (respecting stage cutoff)
        weekend_sessions = self.data_loader.load_weekend_sessions(
            track_key=track_key,
            year=year,
            stage=stage,
            historical_rounds=tracks
        )

        # 3. Load Qualifying (if stage allows)
        quali_df = None
        if stage in ["after_qualifying", "post_race"]:
            quali_df = self.data_loader.load_target_qualifying(track_key, year, historical_tracks=tracks)

        # 4. Build Training Features
        training_features = self.feature_engineer.build_training_dataset(
            races_df=races_df,
            qualis_df=qualis_df,
            practice_sessions_by_round=sessions_by_round,
            train_rounds=train_rounds
        )

        # 5. Build Target Features (STRICT PRE-RACE: No race result)
        target_features = self.feature_engineer.build_target_dataset(
            target_round=target_round,
            track_name=track_name,
            quali_df=quali_df,
            practice_sessions=weekend_sessions,
            season_races_df=races_df,
            season_qualis_df=qualis_df,
            penalties=penalties
        )

        # 6. Audit for Data Leakage
        leakage_report = self.data_validator.audit_leakage(
            training_df=training_features,
            target_df=target_features,
            target_round=target_round
        )
        if not leakage_report["is_clean"]:
            raise AssertionError(f"Leakage check failed: {leakage_report['violations']}")

        # 6b. Fix driver order before deterministic prediction and Monte Carlo:
        # Drivers must be sorted P1 -> P22 by ActualStartingGrid to match sequential NumPy RNG draws
        grid_col = "ActualStartingGrid" if "ActualStartingGrid" in target_features.columns else "GridPosition"
        target_features = target_features.sort_values(grid_col).reset_index(drop=True)

        # 7. Model Walk-Forward Validation & Dynamic Model Selection
        eval_results = self.model_evaluator.evaluate_walk_forward(training_features)

        sel_cfg = self.model_config.get("model_selection", {})
        mode = sel_cfg.get("mode", "walk_forward_auto")

        if mode == "walk_forward_auto":
            selected_rf_weight = eval_results["best_w_rf"]
            selected_xgb_weight = eval_results["best_w_xgb"]
            selected_model_name = eval_results["best_model_name"]
            selected_mae = eval_results["best_mae"]
            selection_method = "Chronological Walk-Forward Validation"
            residuals = eval_results["selected_residuals"]
        elif mode == "fixed_rf":
            selected_rf_weight, selected_xgb_weight = 1.0, 0.0
            selected_model_name = "100% Random Forest / 0% XGBoost"
            selected_mae = eval_results["rf_mae"]
            selection_method = "Fixed Configuration (Random Forest)"
            residuals = eval_results["rf_residuals"]
        elif mode == "fixed_xgb":
            selected_rf_weight, selected_xgb_weight = 0.0, 1.0
            selected_model_name = "0% Random Forest / 100% XGBoost"
            selected_mae = eval_results["xgb_mae"]
            selection_method = "Fixed Configuration (XGBoost)"
            residuals = eval_results["xgb_residuals"]
        elif mode == "fixed_ensemble":
            selected_rf_weight = float(sel_cfg.get("fixed_rf_weight", 0.6))
            selected_xgb_weight = float(sel_cfg.get("fixed_xgb_weight", 0.4))
            selected_model_name = f"{int(selected_rf_weight*100)}% RF / {int(selected_xgb_weight*100)}% XGB"
            selection_method = "Fixed Configuration (Manual Ensemble)"
            match = next((x for x in eval_results["ensemble_comparisons"] if abs(x["rf_weight"] - selected_rf_weight) < 1e-4), None)
            selected_mae = match["mae"] if match else eval_results["best_mae"]
            residuals = eval_results["selected_residuals"]
        else:
            selected_rf_weight = eval_results["best_w_rf"]
            selected_xgb_weight = eval_results["best_w_xgb"]
            selected_model_name = eval_results["best_model_name"]
            selected_mae = eval_results["best_mae"]
            selection_method = "Chronological Walk-Forward Validation"
            residuals = eval_results["selected_residuals"]

        empirical_dnf = eval_results.get("empirical_dnf_rate", 0.158)
        logger.info(f"Selected model: {selected_model_name} via {selection_method} (MAE={selected_mae:.4f})")

        # 8. Train Final Estimators on all training rounds
        imputer, rf_model, xgb_model, feature_names = self.model_trainer.fit(training_features)

        # 9. Deterministic Inference & Grid Blending
        det_results, blended_scores = self.predictor.predict_deterministic(
            target_df=target_features,
            imputer=imputer,
            rf_model=rf_model,
            xgb_model=xgb_model,
            rf_weight=selected_rf_weight,
            xgb_weight=selected_xgb_weight
        )

        # 10. Monte Carlo Simulation (25,000 iterations)
        sim_results, sim_meta = self.simulator.simulate(
            target_df=target_features,
            deterministic_scores=blended_scores,
            residuals=residuals,
            empirical_dnf_rate=empirical_dnf
        )
        sim_meta["selected_model"] = selected_model_name
        sim_meta["selection_method"] = selection_method
        sim_meta["selected_mae"] = selected_mae
        sim_meta["selected_rf_weight"] = selected_rf_weight
        sim_meta["selected_xgb_weight"] = selected_xgb_weight

        # 11. Save Outputs and Reports
        track_output_dir = self.output_dir / track_key
        track_output_dir.mkdir(parents=True, exist_ok=True)

        pred_csv_path = track_output_dir / f"{track_key}_{year}_{stage}_predictions.csv"
        sim_results.to_csv(pred_csv_path, index=False)
        logger.info(f"Saved prediction table to: {pred_csv_path}")

        comp_csv_path = track_output_dir / f"{track_key}_{year}_model_comparison.csv"
        comp_df = pd.DataFrame(eval_results["ensemble_comparisons"])
        comp_df.to_csv(comp_csv_path, index=False)
        logger.info(f"Saved model comparison to: {comp_csv_path}")

        report_path = track_output_dir / f"{track_key}_{year}_{stage}_report.md"
        self._write_markdown_report(
            report_path=report_path,
            track_name=track_name,
            year=year,
            stage=stage,
            target_round=target_round,
            penalties=penalties,
            sim_results=sim_results,
            comp_df=comp_df,
            sim_meta=sim_meta
        )
        logger.info(f"Saved markdown report to: {report_path}")

        return {
            "track_key": track_key,
            "track_name": track_name,
            "year": year,
            "stage": stage,
            "selected_model": selected_model_name,
            "selection_method": selection_method,
            "selected_mae": selected_mae,
            "selected_rf_weight": selected_rf_weight,
            "selected_xgb_weight": selected_xgb_weight,
            "predictions": sim_results,
            "model_comparison": comp_df,
            "simulation_meta": sim_meta,
            "leakage_clean": True,
            "output_dir": str(track_output_dir)
        }

    def _write_markdown_report(
        self,
        report_path: Path,
        track_name: str,
        year: int,
        stage: str,
        target_round: int,
        penalties: Dict[str, int],
        sim_results: pd.DataFrame,
        comp_df: pd.DataFrame,
        sim_meta: Dict[str, Any]
    ):
        """Generates a clean markdown report of the prediction outcomes."""
        top_driver = sim_results.iloc[0]["Abbreviation"]
        top_prob = sim_results.iloc[0]["WinProbability"]

        lines = [
            f"# F1 {year} {track_name} Grand Prix — Race Prediction Report",
            f"**Round**: {target_round} | **Stage**: `{stage}` | **Simulations**: {sim_meta.get('n_simulations', 25000):,}",
            f"**Selected Model**: **{sim_meta.get('selected_model', 'N/A')}**",
            f"**Selection Method**: {sim_meta.get('selection_method', 'Chronological Walk-Forward Validation')}",
            f"**Selected MAE**: {sim_meta.get('selected_mae', 'N/A')}",
            f"**Predicted Winner**: **{top_driver}** ({top_prob}% win probability)",
            "",
            "## Grid Penalties Applied",
        ]

        if penalties:
            for drv, pen in penalties.items():
                lines.append(f"- **{drv}**: +{pen} grid places")
        else:
            lines.append("- No grid penalties applied for this event.")

        lines.extend([
            "",
            "## Predicted Finishing Order & Probabilities",
            "",
            "| Pos | Driver | Team | Grid | Exp. Finish | Median | Win % | Podium % | Top 5 % | Points % | P5-P95 Range |",
            "|:---:|:------:|:-----|:----:|:-----------:|:------:|:-----:|:--------:|:-------:|:--------:|:------------:|",
        ])

        for _, row in sim_results.iterrows():
            pos = int(row["PredictedPosition"])
            drv = row["Abbreviation"]
            team = row["TeamName"]
            grid = int(row["ActualStartingGrid"])
            exp = f"{row['ExpectedFinish']:.2f}"
            med = int(row["MedianFinish"])
            win = f"{row['WinProbability']:.1f}%"
            pod = f"{row['Top3Probability']:.1f}%"
            top5 = f"{row['Top5Probability']:.1f}%"
            pts = f"{row['PointsProbability']:.1f}%"
            rng = f"P{int(row['P5'])}-P{int(row['P95'])}"

            lines.append(f"| {pos} | **{drv}** | {team} | {grid} | {exp} | {med} | {win} | {pod} | {top5} | {pts} | {rng} |")

        lines.extend([
            "",
            "## Walk-Forward Model Comparison",
            "",
            "| Model Configuration | MAE | MedAE | RMSE | Spearman |",
            "|:--------------------|:---:|:-----:|:----:|:--------:|",
        ])

        for _, row in comp_df.iterrows():
            lines.append(f"| {row['model_name']} | {row['mae']:.3f} | {row['medae']:.3f} | {row['rmse']:.3f} | {row['spearman']:.3f} |")

        lines.extend([
            "",
            "## Simulation Parameters",
            f"- **Uncertainty Calibration (Pace σ)**: {sim_meta.get('calibrated_sigma', 'N/A')}",
            f"- **Empirical DNF Risk**: {sim_meta.get('empirical_dnf_prob', 'N/A')}",
            f"- **Random Seed**: {self.model_config.get('random_seed', 42)}",
            "",
            "---",
            "*Report generated by unified F1 Race Prediction Engine with zero data leakage.*"
        ])

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
