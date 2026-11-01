"""
Monte Carlo Race Simulation Module
==================================
Runs 25,000 stochastic race simulations using data-driven uncertainty calibrated from
walk-forward out-of-fold residuals, and empirical incident hazard rates.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

from src.utils.logging_utils import get_logger

logger = get_logger("monte_carlo")


class MonteCarloSimulator:
    """Stochastic race simulation engine for finishing distributions and probabilistic outcomes."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        sim_cfg = config.get("monte_carlo", config.get("simulation", {}))
        self.n_simulations = sim_cfg.get("default_simulations", sim_cfg.get("n_simulations", 25000))
        self.random_state = config.get("random_seed", 42)
        self.incident_scale = sim_cfg.get("incident_shock_scale", sim_cfg.get("incident_scale", 12.0))
        self.default_sigma = sim_cfg.get("fallback_sigma", sim_cfg.get("default_calibrated_sigma", 2.85))
        self.fallback_dnf_prob = sim_cfg.get("fallback_dnf_prob", 0.158)

    def simulate(
        self,
        target_df: pd.DataFrame,
        deterministic_scores: np.ndarray,
        residuals: Optional[np.ndarray] = None,
        empirical_dnf_rate: Optional[float] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes Monte Carlo simulations.

        Parameters
        ----------
        target_df : pd.DataFrame
            Target features dataframe with driver metadata.
        deterministic_scores : np.ndarray
            Base expected finishing score vector from predictor.
        residuals : np.ndarray, optional
            Walk-forward out-of-fold validation residuals for sigma calibration.
        empirical_dnf_rate : float, optional
            Empirical season DNF rate.

        Returns
        -------
        tuple of (simulation_results_df, simulation_metadata_dict)
        """
        rng = np.random.default_rng(self.random_state)

        # Calibrate sigma using MAD (robust standard deviation) if residuals available
        if residuals is not None and len(residuals) > 0:
            calibrated_sigma = float(median_abs_deviation(residuals) * 1.4826)
            calibrated_sigma = max(calibrated_sigma, 1.5)
            logger.info(f"Using residual-calibrated pace sigma: {calibrated_sigma:.4f} (from {len(residuals)} samples)")
        else:
            calibrated_sigma = float(self.default_sigma)
            logger.info(f"Using default pace sigma: {calibrated_sigma:.4f}")

        if empirical_dnf_rate is not None:
            dnf_prob = float(np.clip(empirical_dnf_rate, 0.05, 0.25))
        else:
            dnf_prob = float(self.fallback_dnf_prob)
        logger.info(f"Using empirical DNF incident probability: {dnf_prob:.4f}")

        # Filter out DNS / Race DSQ entrants from active simulation
        status_series = target_df.get("StartingGridStatus", pd.Series(["CLASSIFIED"] * len(target_df), index=target_df.index))
        active_mask = ~status_series.isin(["DNS", "DSQ"])
        if not active_mask.all():
            logger.info(f"Excluding {sum(~active_mask)} DNS/DSQ entrants from active Monte Carlo race simulation.")
            active_df = target_df[active_mask].reset_index(drop=True)
            active_scores = deterministic_scores[active_mask.values]
        else:
            active_df = target_df.reset_index(drop=True)
            active_scores = deterministic_scores

        n = len(active_df)
        sim_scores = np.empty((self.n_simulations, n))

        # Vectorized batch simulation
        for i in range(self.n_simulations):
            noise = rng.normal(0, calibrated_sigma, n)
            # Stochastic incident / collision / mechanical failure shock
            incident = rng.binomial(1, dnf_prob, n) * rng.exponential(self.incident_scale, n)
            sim_scores[i] = active_scores + noise + incident

        # Convert scores to finishing ranks (1 to n)
        sim_finishes = np.argsort(np.argsort(sim_scores, axis=1), axis=1) + 1

        rows = []
        for idx, row in active_df.iterrows():
            finishes = sim_finishes[:, idx]

            q_pos = row.get("QualifyingPosition", row.get("GridPosition", 12))
            actual_grid = row.get("ActualStartingGrid", row.get("GridPosition", 12))
            sim_rank = row.get("simulation_start_rank", actual_grid)
            is_pit = int(row.get("is_pit_lane_start", 0))

            # Semantically distinct fields (Phase 5):
            # 1. grid_penalty_places: assessed steward penalty magnitude (places drop)
            assessed_penalty = int(row.get("grid_penalty_places", row.get("GridPenalty", 0)))
            # 2. ActualStartingGrid: physical starting grid position (nullable for pit-lane starters)
            grid_pos_out = np.nan if (is_pit == 1 or pd.isna(actual_grid)) else int(round(float(actual_grid)))
            # 3. simulation_start_rank: deterministic ranking used by simulation
            sim_rank_out = int(round(float(sim_rank))) if pd.notna(sim_rank) else (
                int(round(float(actual_grid))) if pd.notna(actual_grid) else 22
            )
            # 4. grid_vs_quali_delta: net physical grid displacement from qualifying
            ref_pos = grid_pos_out if pd.notna(grid_pos_out) else sim_rank_out
            delta = int(ref_pos - q_pos) if (pd.notna(ref_pos) and pd.notna(q_pos)) else 0

            rows.append({
                "Abbreviation": str(row["Abbreviation"]),
                "TeamName": str(row.get("TeamName", "Unknown")),
                "QualifyingPosition": int(round(float(q_pos))) if pd.notna(q_pos) else 12,
                "GridPenalty": assessed_penalty,
                "grid_penalty_places": assessed_penalty,
                "grid_vs_quali_delta": delta,
                "ActualStartingGrid": grid_pos_out,
                "simulation_start_rank": sim_rank_out,
                "is_pit_lane_start": is_pit,

                "ExpectedFinish": round(float(finishes.mean()), 2),
                "MedianFinish": int(np.median(finishes)),
                "WinProbability": round(float(np.mean(finishes == 1) * 100), 2),
                "Top3Probability": round(float(np.mean(finishes <= 3) * 100), 2),
                "Top5Probability": round(float(np.mean(finishes <= 5) * 100), 2),
                "PointsProbability": round(float(np.mean(finishes <= 10) * 100), 2),

                "P5": int(np.percentile(finishes, 5)),
                "P25": int(np.percentile(finishes, 25)),
                "P50": int(np.percentile(finishes, 50)),
                "P75": int(np.percentile(finishes, 75)),
                "P95": int(np.percentile(finishes, 95)),

                "MLExpectedFinish": round(float(active_scores[idx]), 2),
            })

        result = (
            pd.DataFrame(rows)
            .sort_values("ExpectedFinish")
            .reset_index(drop=True)
        )
        result["PredictedPosition"] = np.arange(1, len(result) + 1)

        summary = {
            "n_simulations": self.n_simulations,
            "calibrated_sigma": round(calibrated_sigma, 4),
            "empirical_dnf_prob": round(dnf_prob, 4),
            "driver_count": n,
        }

        return result, summary
