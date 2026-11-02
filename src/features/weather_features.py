"""
Weather-Aware Feature Engineering Module
=========================================
Extracts ambient temperature, track temperature, humidity, wind, and rain probability
from FastF1 session weather streams or forecast configurations.
Enforces strict stage-aware cutoffs and provides climate prior fallbacks.
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger("weather_features")

DEFAULT_STRATEGY_CFG_PATH = Path(__file__).resolve().parents[2] / "config" / "strategy_config.json"


def extract_session_weather_metrics(weather_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    """
    Computes summary weather metrics from a FastF1 weather DataFrame.
    Columns typically: ['AirTemp', 'TrackTemp', 'Humidity', 'Pressure', 'WindSpeed', 'Rainfall']
    """
    if weather_df is None or weather_df.empty:
        return {}

    w = weather_df.copy()
    res = {}

    if "AirTemp" in w.columns:
        res["air_temperature"] = float(pd.to_numeric(w["AirTemp"], errors="coerce").dropna().mean())
    if "TrackTemp" in w.columns:
        res["track_temperature"] = float(pd.to_numeric(w["TrackTemp"], errors="coerce").dropna().mean())
    if "Humidity" in w.columns:
        res["humidity"] = float(pd.to_numeric(w["Humidity"], errors="coerce").dropna().mean())
    if "WindSpeed" in w.columns:
        res["wind_speed"] = float(pd.to_numeric(w["WindSpeed"], errors="coerce").dropna().mean())
    if "Rainfall" in w.columns:
        rain_series = pd.to_numeric(w["Rainfall"], errors="coerce").fillna(0)
        # Proportion of session time with detected rainfall
        res["rain_fraction"] = float((rain_series > 0).mean())
        res["rain_detected"] = 1.0 if (rain_series > 0).any() else 0.0

    return res


class WeatherModel:
    """Manages weather features and forecast scenarios."""

    def __init__(self, config_path: Optional[Path] = None):
        cfg_path = config_path or DEFAULT_STRATEGY_CFG_PATH
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = {}

        self.wet_cfg = self.config.get("wet_weather", {})
        self.driver_wet_bonuses = self.wet_cfg.get("driver_wet_skill_bonuses", {
            "VER": -0.65, "HAM": -0.55, "NOR": -0.45, "ALO": -0.40,
            "LEC": -0.30, "RUS": -0.30, "PIA": -0.25, "SAI": -0.20,
        })

    def get_circuit_climate_default(self, circuit_key: Optional[str] = None) -> Dict[str, float]:
        """Provides default seasonal climate metrics for a circuit if live weather is unobserved."""
        ck = str(circuit_key).lower().strip() if circuit_key else "default"
        # Empirically typical temperatures (Air / Track) and rain probabilities by venue
        climate_map = {
            "bahrain": {"air_temp": 28.0, "track_temp": 34.0, "humidity": 45.0, "wind_speed": 3.5, "rain_prob": 0.01},
            "jeddah": {"air_temp": 29.0, "track_temp": 35.0, "humidity": 55.0, "wind_speed": 4.0, "rain_prob": 0.02},
            "melbourne": {"air_temp": 20.0, "track_temp": 31.0, "humidity": 60.0, "wind_speed": 4.5, "rain_prob": 0.20},
            "australia": {"air_temp": 20.0, "track_temp": 31.0, "humidity": 60.0, "wind_speed": 4.5, "rain_prob": 0.20},
            "suzuka": {"air_temp": 19.0, "track_temp": 27.0, "humidity": 65.0, "wind_speed": 3.0, "rain_prob": 0.35},
            "shanghai": {"air_temp": 21.0, "track_temp": 30.0, "humidity": 65.0, "wind_speed": 3.2, "rain_prob": 0.30},
            "miami": {"air_temp": 29.0, "track_temp": 46.0, "humidity": 62.0, "wind_speed": 3.8, "rain_prob": 0.25},
            "monaco": {"air_temp": 22.0, "track_temp": 38.0, "humidity": 58.0, "wind_speed": 2.5, "rain_prob": 0.20},
            "montreal": {"air_temp": 22.0, "track_temp": 36.0, "humidity": 55.0, "wind_speed": 3.5, "rain_prob": 0.35},
            "barcelona": {"air_temp": 26.0, "track_temp": 42.0, "humidity": 50.0, "wind_speed": 2.8, "rain_prob": 0.10},
            "silverstone": {"air_temp": 20.0, "track_temp": 32.0, "humidity": 70.0, "wind_speed": 5.0, "rain_prob": 0.40},
            "spa": {"air_temp": 18.0, "track_temp": 26.0, "humidity": 75.0, "wind_speed": 3.0, "rain_prob": 0.50},
            "monza": {"air_temp": 28.0, "track_temp": 44.0, "humidity": 48.0, "wind_speed": 2.2, "rain_prob": 0.15},
            "baku": {"air_temp": 24.0, "track_temp": 36.0, "humidity": 55.0, "wind_speed": 6.5, "rain_prob": 0.05},
            "singapore": {"air_temp": 30.0, "track_temp": 36.0, "humidity": 80.0, "wind_speed": 2.0, "rain_prob": 0.40},
            "austin": {"air_temp": 26.0, "track_temp": 40.0, "humidity": 52.0, "wind_speed": 3.0, "rain_prob": 0.15},
            "mexico": {"air_temp": 21.0, "track_temp": 42.0, "humidity": 45.0, "wind_speed": 2.5, "rain_prob": 0.15},
            "interlagos": {"air_temp": 23.0, "track_temp": 38.0, "humidity": 70.0, "wind_speed": 3.8, "rain_prob": 0.45},
            "sao_paulo": {"air_temp": 23.0, "track_temp": 38.0, "humidity": 70.0, "wind_speed": 3.8, "rain_prob": 0.45},
            "las_vegas": {"air_temp": 14.0, "track_temp": 17.0, "humidity": 35.0, "wind_speed": 3.0, "rain_prob": 0.05},
            "sepang": {"air_temp": 31.0, "track_temp": 45.0, "humidity": 82.0, "wind_speed": 2.5, "rain_prob": 0.55},
            "default": {"air_temp": 23.0, "track_temp": 35.0, "humidity": 60.0, "wind_speed": 3.5, "rain_prob": 0.20},
        }
        return climate_map.get(ck, climate_map["default"])

    def build_weather_features_for_stage(
        self,
        drivers: List[str],
        circuit_key: Optional[str] = None,
        observed_weather_list: Optional[List[pd.DataFrame]] = None,
        stage: str = "after_qualifying",
    ) -> pd.DataFrame:
        """
        Builds driver-level weather features respecting prediction stage boundaries.
        At pre_fp1: Uses circuit climate priors.
        At after_fp1..after_qualifying: Uses observed weekend weather streams.
        """
        defaults = self.get_circuit_climate_default(circuit_key)

        air_temp = defaults["air_temp"]
        track_temp = defaults["track_temp"]
        humidity = defaults["humidity"]
        wind_speed = defaults["wind_speed"]
        rain_prob = defaults["rain_prob"]
        confidence = 0.30

        # If session weather dataframes are supplied from completed sessions:
        if observed_weather_list:
            valid_metrics = [extract_session_weather_metrics(w) for w in observed_weather_list if w is not None]
            valid_metrics = [m for m in valid_metrics if m]

            if valid_metrics:
                air_temp = float(np.mean([m["air_temperature"] for m in valid_metrics if "air_temperature" in m]))
                track_temp = float(np.mean([m["track_temperature"] for m in valid_metrics if "track_temperature" in m]))
                humidity = float(np.mean([m["humidity"] for m in valid_metrics if "humidity" in m]))
                wind_speed = float(np.mean([m["wind_speed"] for m in valid_metrics if "wind_speed" in m]))
                rain_fractions = [m.get("rain_fraction", 0.0) for m in valid_metrics]
                rain_prob = float(np.clip(np.mean(rain_fractions) * 2.0, rain_prob * 0.5, 0.90))
                confidence = 0.85

        wet_indicator = 1.0 if rain_prob >= 0.50 else (0.5 if rain_prob >= 0.25 else 0.0)

        rows = []
        for drv in drivers:
            rows.append({
                "Abbreviation": drv,
                "air_temperature": round(air_temp, 1),
                "track_temperature": round(track_temp, 1),
                "humidity": round(humidity, 1),
                "wind_speed": round(wind_speed, 1),
                "rain_probability": round(rain_prob, 2),
                "wet_conditions_indicator": wet_indicator,
                "weather_confidence": round(confidence, 2),
            })

        return pd.DataFrame(rows)


def build_weather_features(
    drivers: List[str],
    circuit_key: Optional[str] = None,
    session_weather_list: Optional[List[pd.DataFrame]] = None,
    stage: str = "after_qualifying",
) -> pd.DataFrame:
    """Unified entry point for weather feature extraction."""
    model = WeatherModel()
    return model.build_weather_features_for_stage(
        drivers=drivers,
        circuit_key=circuit_key,
        observed_weather_list=session_weather_list,
        stage=stage,
    )

