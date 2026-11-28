"""Unit tests for canonical and Ergast circuit ID resolution."""

from apexgrid.utils.circuit_mapping import resolve_ergast_circuit_id


def test_resolve_ergast_circuit_id_qatar_and_lusail():
    assert resolve_ergast_circuit_id("qatar") == "losail"
    assert resolve_ergast_circuit_id("lusail") == "losail"
    assert resolve_ergast_circuit_id("  Qatar ") == "losail"


def test_resolve_ergast_circuit_id_austin_and_cota():
    assert resolve_ergast_circuit_id("austin") == "americas"
    assert resolve_ergast_circuit_id("cota") == "americas"


def test_resolve_ergast_circuit_id_abu_dhabi():
    assert resolve_ergast_circuit_id("abu dhabi") == "yas_marina"
    assert resolve_ergast_circuit_id("abu_dhabi") == "yas_marina"


def test_resolve_ergast_circuit_id_historic_passthrough():
    assert resolve_ergast_circuit_id("nurburgring") == "nurburgring"
    assert resolve_ergast_circuit_id("hockenheimring") == "hockenheimring"
    assert resolve_ergast_circuit_id("brands_hatch") == "brands_hatch"
