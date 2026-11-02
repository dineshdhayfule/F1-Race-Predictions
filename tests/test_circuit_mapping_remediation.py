"""
Regression Tests for Centralized Circuit Mapping and Resolution
===============================================================
Verifies that:
1. All 23 canonical keys configured in config/tracks.json resolve deterministically.
2. Event names, display names, slugs, and nicknames resolve to their canonical key.
3. Fallback behaviors function safely.
4. Downstream simulation and feature models correctly interpret resolved canonical keys.
"""

import pytest
from src.utils.circuit_mapping import resolve_circuit_key, get_canonical_circuit_keys, load_tracks_config
from src.simulation.race_control_model import RaceControlModel
from src.features.tire_degradation import TireDegradationModel
from src.features.weather_features import WeatherModel
from src.simulation.strategy_simulator import PitStrategySimulator


def test_tracks_config_contains_expected_23_circuits():
    keys = get_canonical_circuit_keys()
    assert len(keys) == 23, f"Expected 23 canonical circuits, found {len(keys)}"
    expected_samples = ["australia", "baku", "austin", "mexico", "sao_paulo", "sepang", "monaco", "spa"]
    for s in expected_samples:
        assert s in keys, f"Expected {s} in canonical circuit keys"


def test_canonical_keys_resolve_to_themselves():
    for key in get_canonical_circuit_keys():
        assert resolve_circuit_key(key) == key


def test_known_aliases_resolve_correctly():
    cases = {
        "cota": "austin",
        "circuit_of_the_americas": "austin",
        "interlagos": "sao_paulo",
        "autodromo_jose_carlos_pace": "sao_paulo",
        "montreal": "canada",
        "circuit_gilles_villeneuve": "canada",
        "suzuka": "japan",
        "melbourne": "australia",
        "albert_park": "australia",
        "shanghai": "china",
        "red_bull_ring": "austria",
        "spielberg": "austria",
        "catalunya": "barcelona",
        "hungaroring": "hungary",
        "marina_bay": "singapore",
        "yas_marina": "abu_dhabi",
        "losail": "qatar",
        "lusail": "qatar",
    }
    for alias, expected in cases.items():
        assert resolve_circuit_key(alias) == expected, f"Failed for alias: {alias}"


def test_display_names_and_event_names_resolve():
    cases = {
        "Azerbaijan Grand Prix": "baku",
        "azerbaijan_grand_prix": "baku",
        "United States Grand Prix": "austin",
        "united_states_grand_prix": "austin",
        "Mexico City Grand Prix": "mexico",
        "mexico_city_grand_prix": "mexico",
        "São Paulo Grand Prix": "sao_paulo",
        "sao_paulo_grand_prix": "sao_paulo",
        "Malaysian Grand Prix": "sepang",
        "Monaco Grand Prix": "monaco",
        "Belgian Grand Prix": "spa",
        "Italian Grand Prix": "monza",
    }
    for identifier, expected in cases.items():
        assert resolve_circuit_key(identifier) == expected, f"Failed for: {identifier}"


def test_fallback_behavior():
    assert resolve_circuit_key(None) is None
    assert resolve_circuit_key(None, fallback="default") == "default"
    assert resolve_circuit_key("nonexistent_gp_12345", fallback="fallback_val") == "fallback_val"


def test_downstream_models_consume_canonical_keys():
    # Race Control
    rcm = RaceControlModel()
    rc_baku = rcm.get_circuit_event_probabilities("azerbaijan_grand_prix")
    assert rc_baku["p_safety_car"] > 0.40  # Baku high SC
    assert rc_baku["circuit_key"] == "baku"
    rc_cota = rcm.get_circuit_event_probabilities("cota")
    assert "p_vsc" in rc_cota
    assert rc_cota["circuit_key"] == "austin"

    # Tire Degradation Model
    tdm = TireDegradationModel()
    prior_cota = tdm.get_circuit_prior("cota")
    assert "deg_multiplier" in prior_cota
    prior_interlagos = tdm.get_circuit_prior("sao_paulo")
    assert prior_interlagos["deg_multiplier"] > 0

    # Weather Model
    wm = WeatherModel()
    climate_canada = wm.get_circuit_climate_default("canada")
    assert climate_canada["track_temp"] == 36.0
    climate_cota = wm.get_circuit_climate_default("cota")
    assert climate_cota["track_temp"] == 40.0

    # Pit Strategy Simulator
    sim = PitStrategySimulator()
    pit_loss_baku = sim.get_circuit_pit_loss("azerbaijan_grand_prix")
    assert pit_loss_baku > 0.0
