"""
Pit Stop Strategy Simulator Module
==================================
Simulates multi-stint race strategies (1-stop, 2-stop, 3-stop), evaluates tyre degradation
and fuel burn curves, enforces FIA sporting regulations (mandatory 2 dry compounds),
models undercut/overcut dynamics, and generates strategy features and Monte Carlo scenario inputs.
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger
from src.utils.circuit_mapping import resolve_circuit_key

logger = get_logger("strategy_simulator")

DEFAULT_STRATEGY_CFG_PATH = Path(__file__).resolve().parents[2] / "config" / "strategy_config.json"


class PitStrategySimulator:
    """Simulates strategic race scenarios and compound sequences."""

    def __init__(self, config_path: Optional[Path] = None):
        cfg_path = config_path or DEFAULT_STRATEGY_CFG_PATH
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = {}

        self.default_pit_loss = self.config.get("default_pit_loss_seconds", 22.0)
        self.stationary_stop = self.config.get("stationary_stop_seconds", 2.5)
        self.fuel_effect = self.config.get("fuel_effect_seconds_per_lap", 0.035)
        self.compounds = self.config.get("compounds", {
            "SOFT": {"base_deg_rate_sec_per_lap": 0.110, "pace_delta_sec": -0.65, "max_usable_laps": 22},
            "MEDIUM": {"base_deg_rate_sec_per_lap": 0.065, "pace_delta_sec": 0.0, "max_usable_laps": 35},
            "HARD": {"base_deg_rate_sec_per_lap": 0.038, "pace_delta_sec": 0.55, "max_usable_laps": 48},
        })
        self.circuit_priors = self.config.get("circuit_priors", {})

    def get_circuit_pit_loss(self, circuit_key: Optional[str]) -> float:
        """Returns total pit-lane time loss for a given circuit."""
        raw = str(circuit_key).lower().strip() if circuit_key else "default"
        canon = resolve_circuit_key(raw)
        candidates = [c for c in [canon, raw] if c]
        if "japan" in candidates and "suzuka" not in candidates:
            candidates.append("suzuka")
        if "suzuka" in candidates and "japan" not in candidates:
            candidates.append("japan")
        if "sao_paulo" in candidates and "interlagos" not in candidates:
            candidates.append("interlagos")
        if "interlagos" in candidates and "sao_paulo" not in candidates:
            candidates.append("sao_paulo")
        for k in candidates:
            if k in self.circuit_priors:
                return float(self.circuit_priors[k].get("pit_loss_seconds", self.default_pit_loss))
        return float(self.circuit_priors.get("default", {}).get("pit_loss_seconds", self.default_pit_loss))

    def calculate_stint_time(
        self,
        base_lap_pace: float,
        compound: str,
        stint_laps: int,
        start_lap_race: int,
        total_race_laps: int,
        custom_deg_rate: Optional[float] = None,
    ) -> float:
        """
        Calculates projected total time for a single stint:
        - Incorporates tyre compound pace delta
        - Tyre degradation increasing quadratically/linearly over tyre age
        - Fuel burn: car gets lighter and faster by fuel_effect per lap
        """
        c_info = self.compounds.get(compound.upper(), self.compounds["MEDIUM"])
        compound_delta = c_info.get("pace_delta_sec", 0.0)
        deg_rate = custom_deg_rate if custom_deg_rate is not None else c_info.get("base_deg_rate_sec_per_lap", 0.065)

        total_stint_seconds = 0.0
        for i in range(stint_laps):
            tyre_age = i + 1
            race_lap = start_lap_race + i
            # Fuel effect: car is fastest at the end of the race, slowest at lap 1 with full tank
            remaining_fuel_laps = max(0, total_race_laps - race_lap)
            fuel_penalty = remaining_fuel_laps * self.fuel_effect

            # Lap time = base + compound offset + degradation - fuel burn benefit
            lap_sec = base_lap_pace + compound_delta + (deg_rate * tyre_age) + fuel_penalty
            total_stint_seconds += lap_sec

        return total_stint_seconds

    def evaluate_strategy_candidates(
        self,
        base_lap_pace: float,
        total_laps: int,
        circuit_key: Optional[str] = None,
        driver_deg_rates: Optional[Dict[str, float]] = None,
        is_wet: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generates and ranks plausible strategy candidates for a race distance.
        Enforces 2-compound rule for dry races.
        """
        pit_loss = self.get_circuit_pit_loss(circuit_key)
        deg_rates = driver_deg_rates or {}
        candidates = []

        if is_wet:
            # Wet weather strategies: Intermediates or Full Wet, single compound permitted
            inter_deg = self.compounds.get("INTERMEDIATE", {}).get("base_deg_rate_sec_per_lap", 0.055)
            # 1-Stop Intermediate -> Intermediate
            split_ii = int(round(total_laps * 0.50))
            t1 = self.calculate_stint_time(base_lap_pace, "INTERMEDIATE", split_ii, 1, total_laps, inter_deg)
            t2 = self.calculate_stint_time(base_lap_pace, "INTERMEDIATE", total_laps - split_ii, split_ii + 1, total_laps, inter_deg)
            candidates.append({
                "strategy_name": f"1-Stop (Inter-Inter, Lap {split_ii})",
                "stop_count": 1,
                "pit_laps": [split_ii],
                "compounds": ["INTERMEDIATE", "INTERMEDIATE"],
                "total_time_seconds": round(t1 + t2 + pit_loss, 2),
                "is_valid_regulation": True,
            })
            # 2-Stop Intermediate -> Intermediate -> Intermediate
            p1 = int(round(total_laps * 0.33))
            p2 = int(round(total_laps * 0.67))
            t1 = self.calculate_stint_time(base_lap_pace, "INTERMEDIATE", p1, 1, total_laps, inter_deg)
            t2 = self.calculate_stint_time(base_lap_pace, "INTERMEDIATE", p2 - p1, p1 + 1, total_laps, inter_deg)
            t3 = self.calculate_stint_time(base_lap_pace, "INTERMEDIATE", total_laps - p2, p2 + 1, total_laps, inter_deg)
            candidates.append({
                "strategy_name": f"2-Stop (Inter-Inter-Inter, Laps {p1}, {p2})",
                "stop_count": 2,
                "pit_laps": [p1, p2],
                "compounds": ["INTERMEDIATE", "INTERMEDIATE", "INTERMEDIATE"],
                "total_time_seconds": round(t1 + t2 + t3 + 2 * pit_loss, 2),
                "is_valid_regulation": True,
            })
            candidates.sort(key=lambda x: x["total_time_seconds"])
            return candidates

        soft_deg = deg_rates.get("SOFT", self.compounds["SOFT"]["base_deg_rate_sec_per_lap"])
        med_deg = deg_rates.get("MEDIUM", self.compounds["MEDIUM"]["base_deg_rate_sec_per_lap"])
        hard_deg = deg_rates.get("HARD", self.compounds["HARD"]["base_deg_rate_sec_per_lap"])

        candidates = []

        # --- 1-Stop Strategies ---
        # 1. Medium -> Hard (Standard default 1-stop)
        split_mh = int(round(total_laps * 0.42))
        if split_mh <= self.compounds["MEDIUM"]["max_usable_laps"] and (total_laps - split_mh) <= self.compounds["HARD"]["max_usable_laps"]:
            t1 = self.calculate_stint_time(base_lap_pace, "MEDIUM", split_mh, 1, total_laps, med_deg)
            t2 = self.calculate_stint_time(base_lap_pace, "HARD", total_laps - split_mh, split_mh + 1, total_laps, hard_deg)
            total_time = t1 + t2 + pit_loss
            candidates.append({
                "strategy_name": f"1-Stop (M-H, Lap {split_mh})",
                "stop_count": 1,
                "pit_laps": [split_mh],
                "compounds": ["MEDIUM", "HARD"],
                "total_time_seconds": round(total_time, 2),
                "is_valid_regulation": True,
            })

        # 2. Hard -> Medium (Reverse 1-stop)
        split_hm = int(round(total_laps * 0.58))
        if split_hm <= self.compounds["HARD"]["max_usable_laps"] and (total_laps - split_hm) <= self.compounds["MEDIUM"]["max_usable_laps"]:
            t1 = self.calculate_stint_time(base_lap_pace, "HARD", split_hm, 1, total_laps, hard_deg)
            t2 = self.calculate_stint_time(base_lap_pace, "MEDIUM", total_laps - split_hm, split_hm + 1, total_laps, med_deg)
            total_time = t1 + t2 + pit_loss
            candidates.append({
                "strategy_name": f"1-Stop (H-M, Lap {split_hm})",
                "stop_count": 1,
                "pit_laps": [split_hm],
                "compounds": ["HARD", "MEDIUM"],
                "total_time_seconds": round(total_time, 2),
                "is_valid_regulation": True,
            })

        # 3. Soft -> Hard (Aggressive early-stop 1-stop)
        split_sh = int(round(total_laps * 0.28))
        if split_sh <= self.compounds["SOFT"]["max_usable_laps"] and (total_laps - split_sh) <= self.compounds["HARD"]["max_usable_laps"]:
            t1 = self.calculate_stint_time(base_lap_pace, "SOFT", split_sh, 1, total_laps, soft_deg)
            t2 = self.calculate_stint_time(base_lap_pace, "HARD", total_laps - split_sh, split_sh + 1, total_laps, hard_deg)
            total_time = t1 + t2 + pit_loss
            candidates.append({
                "strategy_name": f"1-Stop (S-H, Lap {split_sh})",
                "stop_count": 1,
                "pit_laps": [split_sh],
                "compounds": ["SOFT", "HARD"],
                "total_time_seconds": round(total_time, 2),
                "is_valid_regulation": True,
            })

        # --- 2-Stop Strategies ---
        # 4. Medium -> Hard -> Medium
        p1_mhm = int(round(total_laps * 0.30))
        p2_mhm = int(round(total_laps * 0.68))
        l1 = p1_mhm
        l2 = p2_mhm - p1_mhm
        l3 = total_laps - p2_mhm
        t1 = self.calculate_stint_time(base_lap_pace, "MEDIUM", l1, 1, total_laps, med_deg)
        t2 = self.calculate_stint_time(base_lap_pace, "HARD", l2, p1_mhm + 1, total_laps, hard_deg)
        t3 = self.calculate_stint_time(base_lap_pace, "MEDIUM", l3, p2_mhm + 1, total_laps, med_deg)
        total_time = t1 + t2 + t3 + (2 * pit_loss)
        candidates.append({
            "strategy_name": f"2-Stop (M-H-M, Laps {p1_mhm}, {p2_mhm})",
            "stop_count": 2,
            "pit_laps": [p1_mhm, p2_mhm],
            "compounds": ["MEDIUM", "HARD", "MEDIUM"],
            "total_time_seconds": round(total_time, 2),
            "is_valid_regulation": True,
        })

        # 5. Soft -> Medium -> Hard (Sprint-aggressive 2-stop)
        p1_smh = int(round(total_laps * 0.22))
        p2_smh = int(round(total_laps * 0.58))
        l1 = p1_smh
        l2 = p2_smh - p1_smh
        l3 = total_laps - p2_smh
        t1 = self.calculate_stint_time(base_lap_pace, "SOFT", l1, 1, total_laps, soft_deg)
        t2 = self.calculate_stint_time(base_lap_pace, "MEDIUM", l2, p1_smh + 1, total_laps, med_deg)
        t3 = self.calculate_stint_time(base_lap_pace, "HARD", l3, p2_smh + 1, total_laps, hard_deg)
        total_time = t1 + t2 + t3 + (2 * pit_loss)
        candidates.append({
            "strategy_name": f"2-Stop (S-M-H, Laps {p1_smh}, {p2_smh})",
            "stop_count": 2,
            "pit_laps": [p1_smh, p2_smh],
            "compounds": ["SOFT", "MEDIUM", "HARD"],
            "total_time_seconds": round(total_time, 2),
            "is_valid_regulation": True,
        })

        # Sort candidates strictly by total race time
        candidates.sort(key=lambda x: x["total_time_seconds"])
        return candidates

    def simulate_grid_strategies(
        self,
        target_df: pd.DataFrame,
        total_laps: int = 56,
        circuit_key: Optional[str] = None,
        is_wet: bool = False,
    ) -> pd.DataFrame:
        """
        Runs strategy simulation for every driver on the target starting grid.

        Returns
        -------
        pd.DataFrame
            Features and strategy report columns:
            - optimal_strategy_stops
            - strategy_time_loss_sec
            - recommended_strategy
            - alternative_strategy
            - strategy_delta_seconds
            - strategy_confidence
        """
        drivers = target_df["Abbreviation"].tolist()
        results = []

        base_nominal_pace = 85.0  # seconds

        for _, row in target_df.iterrows():
            drv = row["Abbreviation"]
            # Extract driver-specific compound degradation if available
            d_deg = {
                "SOFT": float(row.get("soft_tyre_deg_rate", 0.110)) if pd.notna(row.get("soft_tyre_deg_rate")) else 0.110,
                "MEDIUM": float(row.get("medium_tyre_deg_rate", 0.065)) if pd.notna(row.get("medium_tyre_deg_rate")) else 0.065,
                "HARD": float(row.get("hard_tyre_deg_rate", 0.038)) if pd.notna(row.get("hard_tyre_deg_rate")) else 0.038,
            }
            conf = float(row.get("tyre_deg_confidence", 0.50)) if pd.notna(row.get("tyre_deg_confidence")) else 0.50

            # Slightly adjust base pace by qualifying rank (fastest qualifier has lower base lap time)
            q_pos = float(row.get("QualifyingPosition", 12.0))
            drv_base_pace = base_nominal_pace + (q_pos - 1.0) * 0.08

            strategies = self.evaluate_strategy_candidates(
                base_lap_pace=drv_base_pace,
                total_laps=total_laps,
                circuit_key=circuit_key,
                driver_deg_rates=d_deg,
                is_wet=is_wet,
            )

            if strategies:
                optimal = strategies[0]
                alt = strategies[1] if len(strategies) > 1 else strategies[0]
                delta_sec = round(float(alt["total_time_seconds"] - optimal["total_time_seconds"]), 2)
                best_stop_count = float(optimal["stop_count"])
                rec_name = optimal["strategy_name"]
                alt_name = alt["strategy_name"]
            else:
                best_stop_count = 1.0
                delta_sec = 5.0
                rec_name = "1-Stop (M-H)"
                alt_name = "2-Stop (S-M-H)"

            results.append({
                "Abbreviation": drv,
                "optimal_strategy_stops": best_stop_count,
                "strategy_time_loss_sec": delta_sec,
                "recommended_strategy": rec_name,
                "alternative_strategy": alt_name,
                "strategy_delta_seconds": delta_sec,
                "strategy_confidence": round(conf, 2),
            })

        return pd.DataFrame(results)


def build_pit_strategy_features(
    target_df: pd.DataFrame,
    total_laps: int = 56,
    circuit_key: Optional[str] = None,
    is_wet: bool = False,
) -> pd.DataFrame:
    """Unified entry point for generating pit strategy features."""
    sim = PitStrategySimulator()
    return sim.simulate_grid_strategies(
        target_df=target_df,
        total_laps=total_laps,
        circuit_key=circuit_key,
        is_wet=is_wet,
    )

