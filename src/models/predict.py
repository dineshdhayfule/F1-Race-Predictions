"""
Inference and Deterministic Prediction Module
=============================================
Generates point predictions from trained estimators and produces blended grid expectations.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import xgboost as xgb

from src.utils.logging_utils import get_logger

logger = get_logger("model_predict")


class Predictor:
    """Generates deterministic finishing predictions for target race."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_cols = config.get("features", config.get("feature_columns", []))
        self.ens_cfg = config.get("models", {}).get("ensemble", {})
        blend_cfg = config.get("blending", {})
        self.rf_weight = self.ens_cfg.get("rf_weight", 1.0)
        self.xgb_weight = self.ens_cfg.get("xgb_weight", 0.0)
        self.grid_weight = blend_cfg.get("grid_weight", self.ens_cfg.get("grid_weight", 0.35))
        self.ml_weight = blend_cfg.get("ml_weight", self.ens_cfg.get("ml_weight", 0.65))
        self.circuit_aware = blend_cfg.get("circuit_aware", True)
        self.circuit_weights = blend_cfg.get("circuit_weights", {
            "street": {"grid_weight": 0.50, "ml_weight": 0.50},
            "permanent": {"grid_weight": 0.35, "ml_weight": 0.65},
            "temporary": {"grid_weight": 0.45, "ml_weight": 0.55},
            "default": {"grid_weight": 0.35, "ml_weight": 0.65}
        })

    def resolve_blending_weights(
        self,
        circuit_type: Optional[str] = None,
        custom_grid_weight: Optional[float] = None,
        custom_ml_weight: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Resolves ML and starting-grid blending weights.
        Prioritizes custom caller weights, then circuit-type specific rules, then config defaults.
        """
        if custom_grid_weight is not None and custom_ml_weight is not None:
            return float(custom_ml_weight), float(custom_grid_weight)

        if self.circuit_aware and circuit_type:
            c_key = str(circuit_type).lower().strip()
            if c_key in self.circuit_weights:
                cw = self.circuit_weights[c_key]
                return float(cw.get("ml_weight", self.ml_weight)), float(cw.get("grid_weight", self.grid_weight))

        return float(self.ml_weight), float(self.grid_weight)

    def predict_deterministic(
        self,
        target_df: pd.DataFrame,
        imputer: SimpleImputer,
        rf_model: RandomForestRegressor,
        xgb_model: Optional[xgb.XGBRegressor] = None,
        rf_weight: Optional[float] = None,
        xgb_weight: Optional[float] = None,
        ridge_model: Optional[Any] = None,
        ridge_weight: Optional[float] = None,
        circuit_type: Optional[str] = None,
        grid_weight: Optional[float] = None,
        ml_weight: Optional[float] = None,
        model_mode: Optional[str] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Calculates deterministic finishing position expectations for each driver.
        Supports RF, XGBoost, Ridge Regression, 2-way Hybrids, and 3-way Multi-Model Ensembles.

        Parameters
        ----------
        target_df : pd.DataFrame
            Target features for upcoming race.
        imputer : SimpleImputer
            Fitted imputer from training pipeline.
        rf_model : RandomForestRegressor
            Fitted Random Forest model.
        xgb_model : XGBRegressor, optional
            Fitted XGBoost model.
        rf_weight : float, optional
            Dynamic or configured Random Forest weight.
        xgb_weight : float, optional
            Dynamic or configured XGBoost weight.
        ridge_model : Pipeline or Ridge, optional
            Fitted Ridge Regression model pipeline.
        ridge_weight : float, optional
            Ridge Regression weight in multi-model ensemble.
        circuit_type : str, optional
            Circuit type ('permanent', 'street', 'temporary') for circuit-aware grid blending.
        grid_weight : float, optional
            Override starting-grid blending weight.
        ml_weight : float, optional
            Override ML blending weight.
        model_mode : str, optional
            Explicit model mode ('rf', 'xgb', 'ridge', 'hybrid', 'ensemble', 'multi').

        Returns
        -------
        tuple of (results_df, blended_deterministic_scores)
        """
        sel_cfg = self.config.get("model_selection", {})
        mode = (model_mode or sel_cfg.get("mode", "fixed_ensemble")).lower().strip()

        for col in self.feature_cols:
            if col not in target_df.columns:
                target_df[col] = np.nan

        X = target_df[self.feature_cols].copy()
        X_imputed = imputer.transform(X)

        rf_pred = rf_model.predict(X_imputed)
        xgb_pred = xgb_model.predict(X_imputed) if xgb_model is not None else rf_pred
        ridge_pred = ridge_model.predict(X_imputed) if ridge_model is not None else rf_pred

        # Determine ML prediction based on mode and weights
        if mode == "ridge" or (ridge_weight is not None and ridge_weight >= 0.99):
            ml_pred = ridge_pred
        elif mode in ("ensemble", "multi"):
            w_rf = 0.40 if rf_weight is None else float(rf_weight)
            w_xgb = 0.30 if xgb_weight is None else float(xgb_weight)
            w_ridge = 0.30 if ridge_weight is None else float(ridge_weight)
            total_w = w_rf + w_xgb + w_ridge
            if total_w > 0:
                w_rf, w_xgb, w_ridge = w_rf / total_w, w_xgb / total_w, w_ridge / total_w
            ml_pred = w_rf * rf_pred + w_xgb * xgb_pred + w_ridge * ridge_pred
        elif mode == "fixed_rf" or mode == "rf":
            ml_pred = rf_pred
        elif mode == "fixed_xgb" or mode == "xgb":
            ml_pred = xgb_pred
        elif rf_weight is not None and xgb_weight is not None:
            try:
                w_rf = float(rf_weight)
                w_xgb = float(xgb_weight)
                w_ridge = float(ridge_weight) if ridge_weight is not None else 0.0
                if np.isnan(w_rf) or w_rf < 0:
                    w_rf = 0.0
                if np.isnan(w_xgb) or w_xgb < 0:
                    w_xgb = 0.0
                if np.isnan(w_ridge) or w_ridge < 0:
                    w_ridge = 0.0
            except (ValueError, TypeError):
                w_rf, w_xgb, w_ridge = 0.60, 0.40, 0.0

            total_w = w_rf + w_xgb + w_ridge
            if total_w <= 1e-6:
                logger.warning("Deterministic weights sum to 0; falling back to 0.60 RF / 0.40 XGB")
                w_rf, w_xgb, w_ridge = 0.60, 0.40, 0.0
                total_w = 1.0

            w_rf, w_xgb, w_ridge = w_rf / total_w, w_xgb / total_w, w_ridge / total_w
            ml_pred = w_rf * rf_pred + w_xgb * xgb_pred + w_ridge * ridge_pred
        elif mode == "fixed_ensemble":
            w_rf = float(sel_cfg.get("fixed_rf_weight", 0.6))
            w_xgb = float(sel_cfg.get("fixed_xgb_weight", 0.4))
            ml_pred = w_rf * rf_pred + w_xgb * xgb_pred
        else:
            w_rf = float(self.ens_cfg.get("rf_weight", 1.0))
            w_xgb = float(self.ens_cfg.get("xgb_weight", 0.0))
            ml_pred = w_rf * rf_pred + w_xgb * xgb_pred if w_xgb > 0 else rf_pred

        # Resolve circuit-type-aware blending weights
        eff_ml_w, eff_grid_w = self.resolve_blending_weights(
            circuit_type=circuit_type,
            custom_grid_weight=grid_weight,
            custom_ml_weight=ml_weight
        )

        # Blended deterministic baseline: eff_ml_w * ML + eff_grid_w * GridPosition
        # Prioritize confirmed simulation_start_rank (handles pit-lane starts), then GridPosition,
        # then legitimate pre-qualifying provisional predictions (provisional_grid_position).
        # If no legitimate starting grid exists (e.g. pre-qualifying without provisional grid),
        # disable grid blending (eff_ml_w = 1.0, eff_grid_w = 0.0) rather than forcing all drivers to P12.
        grid_series = None
        for col_cand in ["simulation_start_rank", "GridPosition", "provisional_grid_position"]:
            if col_cand in target_df.columns:
                s = pd.to_numeric(target_df[col_cand], errors="coerce")
                if s.notna().any():
                    grid_series = s
                    break

        if grid_series is not None and grid_series.notna().any():
            median_pos = float(grid_series.median()) if not np.isnan(grid_series.median()) else 12.0
            grid_positions = grid_series.fillna(median_pos).values
            blended = eff_ml_w * ml_pred + eff_grid_w * grid_positions
        else:
            eff_ml_w, eff_grid_w = 1.0, 0.0
            grid_positions = np.full(len(target_df), np.nan)
            blended = ml_pred

        out = target_df[["Abbreviation", "TeamName"]].copy()
        if "GridPosition" in target_df.columns:
            out["GridPosition"] = target_df["GridPosition"]
        elif "provisional_grid_position" in target_df.columns:
            out["GridPosition"] = target_df["provisional_grid_position"]
        else:
            out["GridPosition"] = np.nan

        if "QualifyingPosition" in target_df.columns:
            out["QualifyingPosition"] = target_df["QualifyingPosition"]
        else:
            out["QualifyingPosition"] = np.nan

        out["ML_PredictedScore"] = np.round(ml_pred, 3)
        out["BlendedScore"] = np.round(blended, 3)
        out["DeterministicRank"] = np.argsort(np.argsort(blended)) + 1

        return out, blended

    def calculate_analytical_weekend_score(
        self,
        target_df: pd.DataFrame,
        stage: str = "after_qualifying",
        circuit_type: Optional[str] = None,
    ) -> np.ndarray:
        """Computes analytical weekend score delegating to module function."""
        return calculate_analytical_weekend_score(
            target_df=target_df,
            stage=stage,
            circuit_type=circuit_type or "permanent",
        )


def calculate_analytical_weekend_score(
    target_df: pd.DataFrame,
    stage: str = "after_qualifying",
    circuit_type: str = "permanent",
) -> np.ndarray:
    """
    Computes an analytical weekend physical pace score for each driver
    based strictly on active weekend session telemetry (FP1-FP3, SQ, Sprint, Quali, Grid).

    Parameters
    ----------
    target_df : pd.DataFrame
        DataFrame of target race drivers with weekend features.
    stage : str
        Prediction stage cutoff (e.g. 'pre_fp1', 'after_fp1', 'after_fp2', 'after_fp3', 'after_qualifying').
    circuit_type : str
        Type of circuit ('street', 'permanent', 'temporary').

    Returns
    -------
    np.ndarray
        Continuous pace scores bounded within [1.0, 22.0] matching ML prediction space.
    """
    n = len(target_df)
    if n == 0:
        return np.array([], dtype=float)

    stage_clean = str(stage or "after_qualifying").lower().strip()
    if stage_clean == "after_sprint":
        stage_clean = "after_sprint_race"

    # 1. Base Practice / Long-Run Pace Score
    # Prefer long_run_pace_gap_pct; fallback to practice_gap_pct, best_practice_lap, or session gaps
    lr_gap = (
        pd.to_numeric(target_df["long_run_pace_gap_pct"], errors="coerce")
        if "long_run_pace_gap_pct" in target_df.columns
        else None
    )

    if lr_gap is not None and lr_gap.notna().any():
        lr_rank = lr_gap.rank(method="average", na_option="bottom")
        # In F1, a 0.0% gap is ~P1, 1.0% is ~P5, 2.5% is ~P15
        gap_pos = 1.0 + np.clip(lr_gap.fillna(lr_rank * 0.15), 0.0, 5.0) * 4.0
        base_pace = 0.50 * lr_rank + 0.50 * gap_pos
    else:
        # Fall back to single-lap practice metrics
        prac_gap = None
        for col in ["practice_gap_pct", "fp3_gap_pct", "fp2_gap_pct", "fp1_gap_pct"]:
            if col in target_df.columns:
                s = pd.to_numeric(target_df[col], errors="coerce")
                if s.notna().any():
                    prac_gap = s
                    break

        if prac_gap is not None and prac_gap.notna().any():
            prac_rank = prac_gap.rank(method="average", na_option="bottom")
            gap_pos = 1.0 + np.clip(prac_gap.fillna(prac_rank * 0.15), 0.0, 5.0) * 4.0
            base_pace = 0.50 * prac_rank + 0.50 * gap_pos
        else:
            # Fall back to best practice lap or neutral order
            best_lap = (
                pd.to_numeric(target_df["best_practice_lap"], errors="coerce")
                if "best_practice_lap" in target_df.columns
                else None
            )
            if best_lap is not None and best_lap.notna().any():
                base_pace = best_lap.rank(method="average", na_option="bottom")
            else:
                base_pace = pd.Series(np.linspace(1.0, float(n), n), index=target_df.index)

    # 2. Tyre Degradation Adjustment
    deg_col = None
    for col in ["fp_tyre_deg_slope", "estimated_tyre_deg_rate", "medium_tyre_deg_rate", "soft_tyre_deg_rate"]:
        if col in target_df.columns:
            s = pd.to_numeric(target_df[col], errors="coerce")
            if s.notna().nunique() > 1:
                deg_col = s
                break

    if deg_col is not None and deg_col.notna().any():
        med_deg = deg_col.median()
        deg_delta = deg_col.fillna(med_deg) - med_deg
        # Scale: +/- 0.05 s/lap difference contributes up to +/- 1.0 position
        deg_penalty = np.clip(deg_delta * 15.0, -2.0, 2.0)
        base_pace = base_pace + deg_penalty

    # 3. Sprint Telemetry Adjustment (if available on Sprint weekends)
    sprint_fin = (
        pd.to_numeric(target_df["sprint_finish_pos"], errors="coerce")
        if "sprint_finish_pos" in target_df.columns
        else None
    )
    sprint_delta = (
        pd.to_numeric(target_df["sprint_pos_delta"], errors="coerce")
        if "sprint_pos_delta" in target_df.columns
        else None
    )
    if sprint_fin is not None and sprint_fin.notna().any():
        if stage_clean == "after_sprint_race":
            delta_val = sprint_delta.fillna(0.0) if sprint_delta is not None else 0.0
            sprint_score = sprint_fin.fillna(base_pace) - 0.20 * np.clip(delta_val, -5.0, 5.0)
            base_pace = 0.40 * sprint_score + 0.40 * base_pace + 0.20 * base_pace.rank()
        elif stage_clean in ["after_qualifying", "before_race", "post_race"]:
            # On Sprint weekends, persist live Sprint Race finishing order into base pace
            valid_sprint = sprint_fin.fillna(base_pace)
            base_pace = 0.80 * base_pace + 0.20 * valid_sprint

    # 4. Qualifying / Starting Grid Blending
    # Circuit-aware weights
    c_key = str(circuit_type or "permanent").lower().strip()
    if c_key == "street":
        w_grid, w_pace = 0.50, 0.50
    elif c_key == "temporary":
        w_grid, w_pace = 0.45, 0.55
    else:
        w_grid, w_pace = 0.35, 0.65

    # Look for confirmed grid or qualifying
    grid_series = None
    if stage_clean in ["after_qualifying", "before_race", "post_race"]:
        for col_cand in ["simulation_start_rank", "ActualStartingGrid", "GridPosition", "QualifyingPosition"]:
            if col_cand in target_df.columns:
                s = pd.to_numeric(target_df[col_cand], errors="coerce")
                if s.notna().any():
                    grid_series = s
                    break
    elif stage_clean in ["after_fp1", "after_fp2", "after_fp3"]:
        # If provisional qualifying prediction exists in pre-qualifying stages
        if "provisional_grid_position" in target_df.columns:
            s = pd.to_numeric(target_df["provisional_grid_position"], errors="coerce")
            if s.notna().any():
                grid_series = s
                # Pre-qualifying provisional grid uses conservative weight
                w_grid, w_pace = 0.25, 0.75

    if grid_series is not None and grid_series.notna().any():
        med_grid = float(grid_series.median()) if not np.isnan(grid_series.median()) else 12.0
        grid_positions = grid_series.fillna(med_grid)
        final_score = w_grid * grid_positions + w_pace * base_pace
    else:
        final_score = base_pace

    # 5. Sanitize and bound
    final_arr = pd.to_numeric(final_score, errors="coerce").to_numpy(dtype=float)
    if np.isnan(final_arr).any():
        nan_mask = np.isnan(final_arr)
        final_arr[nan_mask] = 12.0

    # Ensure strictly bounded in [1.0, 22.0]
    final_arr = np.clip(final_arr, 1.0, max(22.0, float(n)))
    return final_arr
