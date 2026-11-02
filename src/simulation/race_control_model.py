"""
Race Control Probability and Incident Simulation Module
=======================================================
Models circuit-specific Safety Car (SC), Virtual Safety Car (VSC), and Red Flag probabilities.
Simulates field compression, cheap pit stop opportunities, and red-flag restarts inside Monte Carlo iterations
without double-counting baseline DNF hazard rates.
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger("race_control")

DEFAULT_STRATEGY_CFG_PATH = Path(__file__).resolve().parents[2] / "config" / "strategy_config.json"


class RaceControlModel:
    """Estimates and simulates Safety Car, VSC, and Red Flag disruptions."""

    def __init__(self, config_path: Optional[Path] = None):
        cfg_path = config_path or DEFAULT_STRATEGY_CFG_PATH
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = {}

        self.circuit_priors = self.config.get("circuit_priors", {})
        self.sc_discount = self.config.get("sc_pit_loss_discount", 0.40)
        self.vsc_discount = self.config.get("vsc_pit_loss_discount", 0.30)

    def get_circuit_event_probabilities(
        self,
        circuit_key: Optional[str] = None,
        is_wet: bool = False,
    ) -> Dict[str, float]:
        """
        Retrieves smoothed empirical probabilities for SC, VSC, and Red Flag.
        Applies wet-weather hazard multipliers when track is wet.
        """
        from src.utils.circuit_mapping import resolve_circuit_key
        canonical = resolve_circuit_key(circuit_key)
        ck = canonical or (str(circuit_key).lower().strip() if circuit_key else "default")
        info = self.circuit_priors.get(ck)
        if not info and ck == "japan":
            info = self.circuit_priors.get("suzuka")
        if not info and ck == "sao_paulo":
            info = self.circuit_priors.get("interlagos")
        if not info:
            info = self.circuit_priors.get("default", {
                "sc_probability": 0.50,
                "vsc_probability": 0.40,
                "red_flag_probability": 0.15,
            })

        p_sc = float(info.get("sc_probability", 0.50))
        p_vsc = float(info.get("vsc_probability", 0.40))
        p_rf = float(info.get("red_flag_probability", 0.15))

        if is_wet:
            wet_mult = float(self.config.get("wet_weather", {}).get("sc_probability_wet_multiplier", 1.40))
            p_sc = min(0.95, p_sc * wet_mult)
            p_vsc = min(0.90, p_vsc * wet_mult)
            p_rf = min(0.60, p_rf * wet_mult)

        return {
            "p_safety_car": round(p_sc, 3),
            "p_vsc": round(p_vsc, 3),
            "p_red_flag": round(p_rf, 3),
            "circuit_key": ck,
        }

    get_circuit_probabilities = get_circuit_event_probabilities

    def simulate_race_control_disruptions(
        self,
        base_scores: np.ndarray,
        p_sc: float,
        p_vsc: float,
        p_red_flag: float,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, Dict[str, bool]]:
        """
        Simulates race control interventions for a single Monte Carlo run.
        Applies:
        - Field compression: Safety Car bunches the pack, shrinking pace deltas by ~25%
        - Opportunistic pit stops: random mid-field cars gain ~1.5 to 2.5 positions from pit stop timing
        - Red Flag restart: high volatility shuffle
        """
        scores = base_scores.copy()
        n = len(scores)

        has_sc = bool(rng.random() < p_sc)
        has_vsc = bool(rng.random() < p_vsc)
        has_rf = bool(rng.random() < p_red_flag)

        events_occurred = {
            "safety_car": has_sc,
            "vsc": has_vsc,
            "red_flag": has_rf,
        }

        def _rand_int(low: int, high: int) -> int:
            if hasattr(rng, "integers"):
                return int(rng.integers(low, high))
            return int(rng.randint(low, high))

        if has_sc:
            # Field compression: shrink distance between cars towards mean
            mean_score = np.mean(scores)
            compression_factor = 0.75  # 25% gap reduction
            scores = mean_score + compression_factor * (scores - mean_score)

            # Opportunistic "cheap pit stop" under SC:
            # 2 to 4 random runners happen to pit at optimal SC moment, gaining 1.5 to 3.0 seconds
            n_lucky = _rand_int(2, 5)
            lucky_indices = rng.choice(n, size=min(n_lucky, n), replace=False)
            scores[lucky_indices] -= rng.uniform(1.2, 2.8, size=len(lucky_indices))

        elif has_vsc:
            # VSC has milder compression (~10%) and smaller pit advantage
            mean_score = np.mean(scores)
            scores = mean_score + 0.90 * (scores - mean_score)
            n_lucky = _rand_int(1, 3)
            lucky_indices = rng.choice(n, size=min(n_lucky, n), replace=False)
            scores[lucky_indices] -= rng.uniform(0.6, 1.4, size=len(lucky_indices))

        if has_rf:
            # Red flag neutralizes tyre degradation across the field and injects restart volatility
            restart_shock = rng.normal(0.0, 1.2, size=n)
            scores += restart_shock

        return scores, events_occurred


def build_race_control_features(
    circuit_key: Optional[str] = None,
    is_wet: bool = False,
    drivers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Generates circuit race-control features for driver feature matrices.
    """
    model = RaceControlModel()
    probs = model.get_circuit_event_probabilities(circuit_key, is_wet=is_wet)
    drvs = drivers or []
    rows = []
    for drv in drvs:
        rows.append({
            "Abbreviation": drv,
            "p_safety_car": probs["p_safety_car"],
            "p_vsc": probs["p_vsc"],
            "p_red_flag": probs["p_red_flag"],
        })
    return pd.DataFrame(rows)

