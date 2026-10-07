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

        n = len(target_df)
        sim_scores = np.empty((self.n_simulations, n))

        # Vectorized batch simulation
        for i in range(self.n_simulations):
            noise = rng.normal(0, calibrated_sigma, n)
            # Stochastic incident / collision / mechanical failure shock
            incident = rng.binomial(1, dnf_prob, n) * rng.exponential(self.incident_scale, n)
            sim_scores[i] = deterministic_scores + noise + incident

        # Convert scores to finishing ranks (1 to n)
        sim_finishes = np.argsort(np.argsort(sim_scores, axis=1), axis=1) + 1

        rows = []
        for idx, row in target_df.reset_index(drop=True).iterrows():
            finishes = sim_finishes[:, idx]

            q_pos = row.get("QualifyingPosition", row.get("GridPosition", 12))
            grid_pos = row.get("GridPosition", 12)
            penalty = int(grid_pos - q_pos) if q_pos is not None else 0

            rows.append({
                "Abbreviation": str(row["Abbreviation"]),
                "TeamName": str(row.get("TeamName", "Unknown")),
                "QualifyingPosition": int(round(float(q_pos))) if pd.notna(q_pos) else 12,
                "GridPenalty": penalty,
                "ActualStartingGrid": int(round(float(grid_pos))),

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

                "MLExpectedFinish": round(float(deterministic_scores[idx]), 2),
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
