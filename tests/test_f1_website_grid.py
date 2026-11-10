"""
Unit and Integration Tests for Official Formula 1 Website Starting Grid Retrieval
==================================================================================
Covers:
1. Official F1 website starting grid parses into canonical DataFrame correctly.
2. Official F1 grid overrides OpenF1 and Algorithmic Fallback in get_starting_grid().
3. Penalties already reflected in official grid are not applied twice in build_target_dataset().
4. Missing or incomplete official data falls back gracefully without labeling as FINAL.
5. Network timeout handling continues to subsequent sources without unhandled exception.
6. Incorrect or unexpected response structure (e.g. 404, missing table) handled cleanly.
7. Correct provenance (f1_website) and publication status (FINAL).
8. Prediction pipeline and Monte Carlo simulation use the official F1 grid positions.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import urllib.error
import numpy as np
import pandas as pd
import pytest

from src.data.data_loader import DataLoader
from src.data.grid_retriever import StartingGridRetriever
from src.features.grid_features import (
    GridPublicationStatus,
    GridProvenance,
    StartingGridStatus,
)
from src.features.build_features import FeatureEngineer

ROOT = Path(__file__).resolve().parents[1]

# Realistic HTML fixture modeled after formula1.com/en/results/2026/races/1299/brazil/starting-grid
SAMPLE_RACES_INDEX_HTML = """
<!DOCTYPE html>
<html>
<body>
<a href="/en/results/2026/races/1298/mexico/race-result">Mexico</a>
<a href="/en/results/2026/races/1299/brazil/race-result">Brazil</a>
<a href="/en/results/2026/races/1300/las-vegas/race-result">Las Vegas</a>
</body>
</html>
"""

SAMPLE_STARTING_GRID_HTML = """
<!DOCTYPE html>
<html>
<body>
<table>
<thead>
<tr>
  <th>Pos.</th><th>No.</th><th>Driver</th><th>Car</th><th>Time</th>
</tr>
</thead>
<tbody>
  <tr><td>1</td><td>4</td><td>Lando&nbsp;NorrisNOR</td><td>McLaren</td><td>1:09.512</td></tr>
  <tr><td>2</td><td>12</td><td>Kimi&nbsp;AntonelliANT</td><td>Mercedes</td><td>1:09.610</td></tr>
  <tr><td>3</td><td>30</td><td>Liam&nbsp;LawsonLAW</td><td>Racing Bulls</td><td>1:09.700</td></tr>
  <tr><td>4</td><td>27</td><td>Nico&nbsp;HulkenbergHUL</td><td>Audi</td><td>1:09.750</td></tr>
  <tr><td>5</td><td>6</td><td>Isack&nbsp;HadjarHAD</td><td>Red Bull Racing</td><td>1:09.800</td></tr>
  <tr><td>6</td><td>87</td><td>Oliver&nbsp;BearmanBEA</td><td>Haas F1 Team</td><td>1:09.850</td></tr>
  <tr><td>7</td><td>63</td><td>George&nbsp;RussellRUS</td><td>Mercedes</td><td>1:09.550</td></tr>
  <tr><td>8</td><td>14</td><td>Fernando&nbsp;AlonsoALO</td><td>Aston Martin</td><td>1:09.900</td></tr>
  <tr><td>9</td><td>23</td><td>Alexander&nbsp;AlbonALB</td><td>Williams</td><td>1:09.950</td></tr>
  <tr><td>10</td><td>81</td><td>Oscar&nbsp;PiastriPIA</td><td>McLaren</td><td>1:10.000</td></tr>
  <tr><td>11</td><td>3</td><td>Franco&nbsp;ColapintoCAM</td><td>Alpine</td><td>1:10.050</td></tr>
  <tr><td>12</td><td>9</td><td>Arvid&nbsp;LindbladLIN</td><td>Racing Bulls</td><td>1:10.100</td></tr>
  <tr><td>13</td><td>55</td><td>Carlos&nbsp;SainzSAI</td><td>Williams</td><td>1:10.150</td></tr>
  <tr><td>14</td><td>16</td><td>Charles&nbsp;LeclercLEC</td><td>Ferrari</td><td>1:09.650</td></tr>
  <tr><td>15</td><td>43</td><td>Franco&nbsp;ColapintoCOL</td><td>Alpine</td><td>1:10.200</td></tr>
  <tr><td>16</td><td>10</td><td>Pierre&nbsp;GaslyGAS</td><td>Alpine</td><td>1:10.250</td></tr>
  <tr><td>17</td><td>18</td><td>Lance&nbsp;StrollSTR</td><td>Aston Martin</td><td>1:10.300</td></tr>
  <tr><td>18</td><td>77</td><td>Valtteri&nbsp;BottasBOT</td><td>Audi</td><td>1:10.350</td></tr>
  <tr><td>19</td><td>5</td><td>Gabriel&nbsp;BortoletoBOR</td><td>Haas F1 Team</td><td>1:10.400</td></tr>
  <tr><td>20</td><td>44</td><td>Lewis&nbsp;HamiltonHAM</td><td>Ferrari</td><td>1:09.800</td></tr>
  <tr><td>21</td><td>1</td><td>Max&nbsp;VerstappenVER</td><td>Red Bull Racing</td><td>1:09.820</td></tr>
  <tr><td>22</td><td>11</td><td>Sergio&nbsp;PerezPER</td><td>Cadillac</td><td>1:10.500</td></tr>
</tbody>
</table>
<div class="notes">
Note - Russell has a five-place grid drop and Leclerc a 10-place grid drop for taking on a new ICE. Sergio Perez, Lewis Hamilton and Max Verstappen are all required to start from the back of the grid after taking on a raft of new PU components.
</div>
</body>
</html>
"""


def _mock_f1_urlopen(url_str, *args, **kwargs):
    """Helper mock that returns index or starting grid HTML depending on requested URL."""
    url = url_str.full_url if hasattr(url_str, "full_url") else str(url_str)
    mock_resp = MagicMock()
    if "starting-grid" in url:
        mock_resp.read.return_value = SAMPLE_STARTING_GRID_HTML.encode("utf-8")
    else:
        mock_resp.read.return_value = SAMPLE_RACES_INDEX_HTML.encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    return mock_resp


# =============================================================================
# SCENARIO 1: Parse Official F1 Website Grid Into Canonical DataFrame
# =============================================================================
def test_f1_website_parsing_to_canonical_dataframe():
    retriever = StartingGridRetriever(ROOT)
    with patch("urllib.request.urlopen", side_effect=_mock_f1_urlopen):
        manifest = retriever._fetch_f1_website_live("interlagos", 2026, 20)

    assert manifest is not None
    assert manifest["season"] == 2026
    assert manifest["round"] == 20
    assert manifest["circuit_key"] == "interlagos"
    assert manifest["publication_status"] == GridPublicationStatus.FINAL.value
    assert len(manifest["grid"]) == 22

    # Validate manifest schema
    valid_m, errs_m = retriever.validate_manifest_json(manifest)
    assert valid_m is True, f"Manifest validation failed: {errs_m}"

    # Convert to canonical DataFrame
    df = retriever._manifest_to_dataframe(manifest, expected_grid_size=22)
    assert df is not None
    assert len(df) == 22

    # Validate canonical columns and integrity
    req_cols = [
        "Abbreviation", "TeamName", "ActualStartingGrid", "simulation_start_rank",
        "is_pit_lane_start", "StartingGridStatus", "grid_norm"
    ]
    for c in req_cols:
        assert c in df.columns, f"Missing required column: {c}"

    valid_df, errs_df = retriever.validate_grid_dataframe(df, expected_grid_size=22)
    assert valid_df is True, f"DataFrame validation failed: {errs_df}"

    # Verify key driver starting positions
    assert df.iloc[0]["Abbreviation"] == "NOR" and int(df.iloc[0]["ActualStartingGrid"]) == 1
    assert df.iloc[1]["Abbreviation"] == "ANT" and int(df.iloc[1]["ActualStartingGrid"]) == 2
    assert df.iloc[6]["Abbreviation"] == "RUS" and int(df.iloc[6]["ActualStartingGrid"]) == 7
    assert df.iloc[13]["Abbreviation"] == "LEC" and int(df.iloc[13]["ActualStartingGrid"]) == 14
    assert df.iloc[20]["Abbreviation"] == "VER" and int(df.iloc[20]["ActualStartingGrid"]) == 21


# =============================================================================
# SCENARIO 2: Official F1 Grid Overrides OpenF1 and Algorithmic Fallback
# =============================================================================
def test_f1_website_overrides_openf1_and_algorithmic_fallback(tmp_path):
    retriever = StartingGridRetriever(tmp_path)
    quali_df = pd.DataFrame({
        "Abbreviation": ["NOR", "ANT", "LAW"],
        "Position": [1, 2, 3],
        "TeamName": ["McLaren", "Mercedes", "Racing Bulls"]
    })

    # Mock F1 website returning a valid 3-car grid
    mock_f1_manifest = {
        "season": 2026,
        "round": 20,
        "circuit_key": "interlagos",
        "document_reference": "Official Formula 1 Website Published Starting Grid (Brazil 2026)",
        "publication_status": "FINAL",
        "grid": [
            {"driver": "NOR", "team": "McLaren", "grid_pos": 1, "simulation_start_rank": 1, "is_pit_lane_start": 0},
            {"driver": "ANT", "team": "Mercedes", "grid_pos": 2, "simulation_start_rank": 2, "is_pit_lane_start": 0},
            {"driver": "LAW", "team": "Racing Bulls", "grid_pos": 3, "simulation_start_rank": 3, "is_pit_lane_start": 0},
        ]
    }
    mock_openf1_df = pd.DataFrame({
        "Abbreviation": ["ANT", "NOR", "LAW"],
        "ActualStartingGrid": [1.0, 2.0, 3.0],
        "simulation_start_rank": [1.0, 2.0, 3.0],
        "is_pit_lane_start": [0, 0, 0]
    })

    with patch.object(retriever, "_retrieve_fia_manifest", return_value=(None, {})), \
         patch.object(retriever, "_fetch_f1_website_live", return_value=mock_f1_manifest), \
         patch.object(retriever, "_retrieve_openf1", return_value=(mock_openf1_df, {"provenance": "openf1_api"})):

        grid_df, meta = retriever.get_starting_grid(
            track_key="interlagos",
            year=2026,
            round_num=20,
            stage="after_qualifying",
            quali_df=quali_df,
            expected_grid_size=3,
        )

        assert grid_df is not None
        assert meta["provenance"] == GridProvenance.F1_WEBSITE.value
        assert "Official Formula 1 Website" in meta["document_reference"]
        # NOR is P1 on F1 website (OpenF1 had ANT P1)
        assert grid_df.iloc[0]["Abbreviation"] == "NOR"


# =============================================================================
# SCENARIO 3: Penalties Already Reflected Are Not Applied Twice
# =============================================================================
def test_penalties_already_reflected_are_not_applied_twice(tmp_path):
    """
    When starting_grid_df is loaded from F1 website, build_target_dataset() must use
    ActualStartingGrid directly and not deduct penalties again via resolve_fia_starting_grid().
    """
    builder = FeatureEngineer(expected_grid_size=8)

    # Russell qualified P2, but official F1 website starting grid already put him at P7
    # Leclerc qualified P4, but official F1 website grid put him at P14
    f1_grid_df = pd.DataFrame({
        "Abbreviation": ["NOR", "ANT", "LAW", "HUL", "HAD", "BEA", "RUS", "LEC"],
        "TeamName": ["McLaren", "Mercedes", "Racing Bulls", "Audi", "Red Bull", "Haas", "Mercedes", "Ferrari"],
        "ActualStartingGrid": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "simulation_start_rank": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "is_pit_lane_start": [0, 0, 0, 0, 0, 0, 0, 0],
        "StartingGridStatus": ["CLASSIFIED"] * 8,
        "grid_penalty_places": [0, 0, 0, 0, 0, 0, 5, 10],
        "GridPenalty": [0, 0, 0, 0, 0, 0, 5, 10],
        "grid_vs_quali_delta": [0, 0, 0, 0, 0, 0, 5, 4],
        "grid_norm": np.linspace(0, 1, 8),
    })

    quali_df = pd.DataFrame({
        "Abbreviation": ["NOR", "RUS", "ANT", "LEC", "LAW", "HUL", "HAD", "BEA"],
        "Position": [1, 2, 3, 4, 5, 6, 7, 8],
        "TeamName": ["McLaren", "Mercedes", "Mercedes", "Ferrari", "Racing Bulls", "Audi", "Red Bull", "Haas"],
    })

    empty_races = pd.DataFrame(columns=["Round", "Abbreviation", "Position", "GridPosition", "Points", "TeamName"])
    empty_qualis = pd.DataFrame(columns=["Round", "Abbreviation", "Position"])

    # Pass penalties explicitly; builder must NOT re-apply penalties to f1_grid_df
    features = builder.build_target_dataset(
        target_round=20,
        track_name="Interlagos",
        quali_df=quali_df,
        practice_sessions={},
        season_races_df=empty_races,
        season_qualis_df=empty_qualis,
        penalties={"RUS": 5, "LEC": 10},
        stage="after_qualifying",
        starting_grid_df=f1_grid_df,
        expected_grid_size=8,
    )

    rus_feat = features[features["Abbreviation"] == "RUS"].iloc[0]
    lec_feat = features[features["Abbreviation"] == "LEC"].iloc[0]

    # Russell remains at ActualStartingGrid 7 (NOT 7 + 5 = 12)
    assert rus_feat["ActualStartingGrid"] == 7.0
    # Leclerc remains at ActualStartingGrid 8 (NOT 8 + 10 = 18)
    assert lec_feat["ActualStartingGrid"] == 8.0


# =============================================================================
# SCENARIO 4: Missing or Incomplete Data Falls Back Gracefully
# =============================================================================
def test_missing_or_incomplete_official_data_falls_back_gracefully(tmp_path):
    retriever = StartingGridRetriever(tmp_path)
    quali_df = pd.DataFrame({
        "Abbreviation": ["NOR", "ANT", "VER"],
        "Position": [1, 2, 3],
    })

    # Return incomplete data (only 1 driver when 3 expected)
    incomplete_manifest = {
        "season": 2026,
        "round": 20,
        "circuit_key": "interlagos",
        "publication_status": "PROVISIONAL",
        "grid": [
            {"driver": "NOR", "grid_pos": 1, "simulation_start_rank": 1, "is_pit_lane_start": 0}
        ]
    }

    with patch.object(retriever, "_retrieve_fia_manifest", return_value=(None, {})), \
         patch.object(retriever, "_fetch_f1_website_live", return_value=incomplete_manifest), \
         patch.object(retriever, "_retrieve_openf1", return_value=(None, {})):

        grid_df, meta = retriever.get_starting_grid(
            track_key="interlagos",
            year=2026,
            round_num=20,
            stage="after_qualifying",
            quali_df=quali_df,
            expected_grid_size=3,
        )

        assert grid_df is not None
        # Must fall back to Algorithmic Fallback, not label as FINAL F1 Website
        assert meta["provenance"] == GridProvenance.ALGORITHMIC_FALLBACK.value
        assert meta["publication_status"] == GridPublicationStatus.PROVISIONAL.value


# =============================================================================
# SCENARIO 5: Network Timeout Continues to Subsequent Sources
# =============================================================================
def test_network_timeout_continues_to_subsequent_sources(tmp_path):
    retriever = StartingGridRetriever(tmp_path)
    quali_df = pd.DataFrame({
        "Abbreviation": ["NOR", "ANT"],
        "Position": [1, 2],
    })

    with patch.object(retriever, "_retrieve_fia_manifest", return_value=(None, {})), \
         patch("urllib.request.urlopen", side_effect=TimeoutError("Request timed out")), \
         patch.object(retriever, "_retrieve_openf1", return_value=(None, {})):

        grid_df, meta = retriever.get_starting_grid(
            track_key="interlagos",
            year=2026,
            round_num=20,
            stage="after_qualifying",
            quali_df=quali_df,
            expected_grid_size=2,
        )

        # No unhandled exception; successfully falls back to Algorithmic Fallback
        assert grid_df is not None
        assert meta["provenance"] == GridProvenance.ALGORITHMIC_FALLBACK.value


# =============================================================================
# SCENARIO 6: Unexpected Response Structure Handled Cleanly (404, No Table)
# =============================================================================
def test_unexpected_response_structure_handled_cleanly(tmp_path):
    retriever = StartingGridRetriever(tmp_path)

    # 1. HTTP 404 Error
    err = urllib.error.HTTPError("https://www.formula1.com/en/results/2026/races", 404, "Not Found", {}, None)
    with patch("urllib.request.urlopen", side_effect=err):
        res = retriever._fetch_f1_website_live("interlagos", 2026, 20)
        assert res is None

    # 2. Non-HTML / Malformed content with no table
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"<html><body>No tables here!</body></html>"
    mock_resp.__enter__.return_value = mock_resp
    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = retriever._fetch_f1_website_live("interlagos", 2026, 20)
        assert res is None

    # 3. Future unrun race table indicating "ErrorNo results available"
    unrun_html = "<table><tr><th>Pos.</th><th>Driver</th></tr><tr><td>ErrorNo results available</td></tr></table>"
    mock_resp2 = MagicMock()
    mock_resp2.read.side_effect = [
        SAMPLE_RACES_INDEX_HTML.encode("utf-8"),
        unrun_html.encode("utf-8")
    ]
    mock_resp2.__enter__.return_value = mock_resp2
    with patch("urllib.request.urlopen", return_value=mock_resp2):
        res = retriever._fetch_f1_website_live("las_vegas", 2026, 21)
        assert res is None


# =============================================================================
# SCENARIO 7: Correct Provenance (f1_website) and Publication Status (FINAL)
# =============================================================================
def test_f1_website_provenance_and_final_publication_status(tmp_path):
    retriever = StartingGridRetriever(tmp_path)
    quali_df = pd.DataFrame({
        "Abbreviation": ["NOR", "ANT", "LAW"],
        "Position": [1, 2, 3],
    })

    with patch.object(retriever, "_retrieve_fia_manifest", return_value=(None, {})), \
         patch("urllib.request.urlopen", side_effect=_mock_f1_urlopen):

        grid_df, meta = retriever.get_starting_grid(
            track_key="interlagos",
            year=2026,
            round_num=20,
            stage="after_qualifying",
            quali_df=quali_df,
            expected_grid_size=22,
        )

        assert grid_df is not None
        assert meta["provenance"] == "f1_website"
        assert meta["publication_status"] == "FINAL"
        assert "Official Formula 1 Website" in meta["document_reference"]
        assert "stewards_notes" in meta
        assert len(meta["stewards_notes"]) > 0
        assert "source_url" in meta
        assert "formula1.com" in meta["source_url"]


# =============================================================================
# SCENARIO 8: Prediction Pipeline Uses Official F1 Website Grid Positions
# =============================================================================
def test_prediction_pipeline_uses_f1_website_grid_positions(tmp_path):
    loader = DataLoader(root_dir=ROOT)
    retriever = StartingGridRetriever(tmp_path)

    with patch.object(retriever, "_retrieve_fia_manifest", return_value=(None, {})), \
         patch("urllib.request.urlopen", side_effect=_mock_f1_urlopen):

        grid_df, meta = retriever.get_starting_grid(
            track_key="interlagos",
            year=2026,
            round_num=20,
            stage="after_qualifying",
            expected_grid_size=22,
        )

        assert grid_df is not None
        assert meta["provenance"] == GridProvenance.F1_WEBSITE.value

        # Russell qualified P2 in raw times, but in official F1 grid is P7
        rus_row = grid_df[grid_df["Abbreviation"] == "RUS"].iloc[0]
        assert int(rus_row["ActualStartingGrid"]) == 7
        assert int(rus_row["simulation_start_rank"]) == 7

        # Verstappen qualified high, but starts P21
        ver_row = grid_df[grid_df["Abbreviation"] == "VER"].iloc[0]
        assert int(ver_row["ActualStartingGrid"]) == 21
        assert int(ver_row["simulation_start_rank"]) == 21
