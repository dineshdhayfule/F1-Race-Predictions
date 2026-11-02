"""
Monte Carlo Race Simulation Module
==================================
Runs 25,000 stochastic race simulations using data-driven uncertainty calibrated from
walk-forward out-of-fold residuals, empirical incident hazard rates, pit strategy variation,
circuit-specific race control (SC/VSC/Red Flag) disruptions, and weather effects.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

from src.simulation.race_control_model import RaceControlModel
from src.features.weather_features import WeatherModel
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

        # Capability toggles
        caps = config.get("capabilities", {})
        self.enable_race_control = caps.get("race_control_model", True)
        self.enable_weather = caps.get("weather_aware_prediction", True)
        self.enable_pit_strategy = caps.get("pit_strategy_simulator", True)

        self.race_control_model = RaceControlModel()
        self.weather_model = WeatherModel()

    def simulate(
        self,
        target_df: pd.DataFrame,
        deterministic_scores: np.ndarray,
        residuals: Optional[np.ndarray] = None,
        empirical_dnf_rate: Optional[float] = None,
        circuit_key: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes Monte Carlo simulations incorporating strategic, race control, and weather factors.

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
        circuit_key : str, optional
            Target circuit identifier.

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

        # Extract circuit event probabilities for race control
        p_sc = float(active_df["p_safety_car"].iloc[0]) if "p_safety_car" in active_df.columns and pd.notna(active_df["p_safety_car"].iloc[0]) else 0.50
        p_vsc = float(active_df["p_vsc"].iloc[0]) if "p_vsc" in active_df.columns and pd.notna(active_df["p_vsc"].iloc[0]) else 0.40
        p_rf = float(active_df["p_red_flag"].iloc[0]) if "p_red_flag" in active_df.columns and pd.notna(active_df["p_red_flag"].iloc[0]) else 0.15

        # Extract rain probability
        rain_prob = float(active_df["rain_probability"].iloc[0]) if "rain_probability" in active_df.columns and pd.notna(active_df["rain_probability"].iloc[0]) else 0.15

        # Vector of wet skill bonuses
        wet_skill_bonuses = np.zeros(n)
        for idx, drv in enumerate(active_df["Abbreviation"]):
            wet_skill_bonuses[idx] = self.weather_model.driver_wet_bonuses.get(str(drv).upper(), 0.0)

        # Vector of strategy time loss deltas
        strat_time_loss = np.zeros(n)
        if "strategy_time_loss_sec" in active_df.columns:
            strat_time_loss = active_df["strategy_time_loss_sec"].fillna(0.0).values * 0.15

        # Vectorized batch simulation
        for i in range(self.n_simulations):
            iter_sigma = calibrated_sigma
            iter_dnf_prob = dnf_prob
            iter_scores = active_scores.copy()

            # Weather scenario sampling
            if self.enable_weather and (rng.random() < rain_prob):
                # Wet race scenario: increased pace volatility, higher DNF hazard
                iter_sigma *= 1.35
                iter_dnf_prob = min(0.35, dnf_prob * 1.40)
                # Apply wet driver skill differentials
                iter_scores = iter_scores + wet_skill_bonuses

            # Baseline pace noise
            noise = rng.normal(0, iter_sigma, n)

            # Strategy execution noise (pit stop execution variation: +-0.75s)
            if self.enable_pit_strategy:
                strat_noise = rng.normal(0, 0.75, n) + strat_time_loss
            else:
                strat_noise = 0.0

            # Stochastic DNF shock
            incident = rng.binomial(1, iter_dnf_prob, n) * rng.exponential(self.incident_scale, n)

            iter_scores = iter_scores + noise + strat_noise + incident

            # Race control disruption sampling (Safety Car / VSC / Red Flag)
            if self.enable_race_control:
                iter_scores, _ = self.race_control_model.simulate_race_control_disruptions(
                    base_scores=iter_scores,
                    p_sc=p_sc,
                    p_vsc=p_vsc,
                    p_red_flag=p_rf,
                    rng=rng,
                )

            sim_scores[i] = iter_scores

        # Convert scores to finishing ranks (1 to n)
        sim_finishes = np.argsort(np.argsort(sim_scores, axis=1), axis=1) + 1

        rows = []
        for idx, row in active_df.iterrows():
            finishes = sim_finishes[:, idx]

            # Robust handling of stage-masked or unobserved fields:
            raw_q = row.get("QualifyingPosition")
            q_pos_out = int(round(float(raw_q))) if pd.notna(raw_q) else np.nan

            raw_act = row.get("ActualStartingGrid")
            actual_grid = float(raw_act) if pd.notna(raw_act) else np.nan

            raw_sim = row.get("simulation_start_rank")
            if pd.isna(raw_sim):
                raw_sim = row.get("ActualStartingGrid")
            if pd.isna(raw_sim):
                raw_sim = row.get("GridPosition")
            sim_rank_out = int(round(float(raw_sim))) if pd.notna(raw_sim) else (idx + 1)

            raw_pit = row.get("is_pit_lane_start")
            is_pit = int(round(float(raw_pit))) if pd.notna(raw_pit) else 0

            raw_pen = row.get("grid_penalty_places")
            if pd.isna(raw_pen):
                raw_pen = row.get("GridPenalty")
            assessed_penalty = int(round(float(raw_pen))) if pd.notna(raw_pen) else 0

            grid_pos_out = np.nan if (is_pit == 1 or pd.isna(actual_grid)) else int(round(float(actual_grid)))
            ref_pos = grid_pos_out if pd.notna(grid_pos_out) else sim_rank_out
            delta = int(ref_pos - q_pos_out) if (pd.notna(ref_pos) and pd.notna(q_pos_out)) else 0

            rec_dict = {
                "Abbreviation": str(row["Abbreviation"]),
                "TeamName": str(row.get("TeamName", "Unknown")),
                "QualifyingPosition": q_pos_out,
                "GridPenalty": assessed_penalty,
                "grid_penalty_places": assessed_penalty,
                "grid_vs_quali_delta": delta,
                "ActualStartingGrid": grid_pos_out,
                "simulation_start_rank": sim_rank_out,
                "is_pit_lane_start": is_pit,
                "StartingGridStatus": str(row.get("StartingGridStatus", "CLASSIFIED")),

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
            }

            # Attach strategy recommendation columns if enabled
            if self.enable_pit_strategy:
                if "recommended_strategy" in row and pd.notna(row["recommended_strategy"]):
                    rec_dict["RecommendedStrategy"] = str(row["recommended_strategy"])
                if "optimal_strategy_stops" in row and pd.notna(row["optimal_strategy_stops"]):
                    rec_dict["OptimalStops"] = int(round(float(row["optimal_strategy_stops"])))
                if "strategy_delta_seconds" in row and pd.notna(row["strategy_delta_seconds"]):
                    rec_dict["StrategyDeltaSec"] = round(float(row["strategy_delta_seconds"]), 2)

            rows.append(rec_dict)

        active_result = (
            pd.DataFrame(rows)
            .sort_values("ExpectedFinish")
            .reset_index(drop=True)
        )
        active_result["PredictedPosition"] = np.arange(1, len(active_result) + 1)

        excluded_rows = []
        if not active_mask.all():
            excluded_df = target_df[~active_mask].reset_index(drop=True)
            for _, ex_row in excluded_df.iterrows():
                q_pos_raw = ex_row.get("QualifyingPosition")
                q_pos_out = int(round(float(q_pos_raw))) if pd.notna(q_pos_raw) else np.nan

                sim_rank_raw = ex_row.get("simulation_start_rank")
                sim_rank_out = int(round(float(sim_rank_raw))) if pd.notna(sim_rank_raw) else np.nan

                actual_grid = ex_row.get("ActualStartingGrid")
                raw_pit = ex_row.get("is_pit_lane_start")
                is_pit = int(round(float(raw_pit))) if pd.notna(raw_pit) else 0

                raw_pen = ex_row.get("grid_penalty_places")
                if pd.isna(raw_pen):
                    raw_pen = ex_row.get("GridPenalty")
                assessed_penalty = int(round(float(raw_pen))) if pd.notna(raw_pen) else 0

                grid_pos_out = np.nan if (is_pit == 1 or pd.isna(actual_grid)) else int(round(float(actual_grid)))
                ref_pos = grid_pos_out if pd.notna(grid_pos_out) else sim_rank_out
                delta = int(ref_pos - q_pos_out) if (pd.notna(ref_pos) and pd.notna(q_pos_out)) else 0
                status_val = str(ex_row.get("StartingGridStatus", "NON_STARTER"))

                ex_dict = {
                    "Abbreviation": str(ex_row["Abbreviation"]),
                    "TeamName": str(ex_row.get("TeamName", "Unknown")),
                    "QualifyingPosition": q_pos_out,
                    "GridPenalty": assessed_penalty,
                    "grid_penalty_places": assessed_penalty,
                    "grid_vs_quali_delta": delta,
                    "ActualStartingGrid": grid_pos_out,
                    "simulation_start_rank": sim_rank_out,
                    "is_pit_lane_start": is_pit,
                    "StartingGridStatus": status_val,

                    "ExpectedFinish": np.nan,
                    "MedianFinish": np.nan,
                    "WinProbability": 0.0,
                    "Top3Probability": 0.0,
                    "Top5Probability": 0.0,
                    "PointsProbability": 0.0,

                    "P5": np.nan,
                    "P25": np.nan,
                    "P50": np.nan,
                    "P75": np.nan,
                    "P95": np.nan,

                    "MLExpectedFinish": np.nan,
                    "PredictedPosition": np.nan,
                }
                excluded_rows.append(ex_dict)

        if excluded_rows:
            excluded_result = pd.DataFrame(excluded_rows)
            result = pd.concat([active_result, excluded_result], ignore_index=True)
        else:
            result = active_result

        summary = {
            "n_simulations": self.n_simulations,
            "calibrated_sigma": round(calibrated_sigma, 4),
            "empirical_dnf_prob": round(dnf_prob, 4),
            "driver_count": len(result),
            "active_driver_count": n,
            "p_safety_car": round(p_sc, 3),
            "rain_probability": round(rain_prob, 3),
        }

        return result, summary
