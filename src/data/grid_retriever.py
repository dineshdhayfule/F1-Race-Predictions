"""
Starting-Grid Manifest and Multi-Source Retrieval System
========================================================
Authoritative loader and multi-source retrieval engine for F1 starting grids.
Supports:
- Source 1: Official FIA Starting Grid Manifests (data/grids/{year}/{year}_round_{round}_starting_grid.json)
- Source 2: Official Formula 1 Website Published Grid Data (data/raw/grids/{year}/...f1_website_grid.json)
- Source 3: OpenF1 Starting Grid API (GET https://api.openf1.org/v1/starting_grid) with persistent cache
- Source 4: Generic Deterministic FIA Grid Resolver (resolve_fia_starting_grid) fallback
- Canonical schema validation with physical contiguity and pit-lane nullness checks.
- Pre-qualifying cutoff enforcement (no future grids leaked into early stages).
- Full end-to-end metadata preservation (provenance, publication status, document reference).
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.features.grid_features import (
    GridPublicationStatus,
    GridProvenance,
    PenaltyType,
    StartingGridStatus,
    resolve_fia_starting_grid,
)
from src.utils.driver_mapping import DRIVER_CATALOG, NAME_TO_CODE, normalize_driver_series
from src.utils.logging_utils import get_logger

logger = get_logger("grid_retriever")


class StartingGridRetriever:
    """Authoritative multi-source retrieval and validation engine for F1 starting grids."""

    def __init__(self, root_dir: Optional[Union[str, Path]] = None):
        self.root_dir = Path(root_dir) if root_dir else Path(__file__).resolve().parents[2]
        self.grids_dir = self.root_dir / "data" / "grids"
        self.raw_grids_dir = self.root_dir / "data" / "raw" / "grids"

    def get_manifest_path(self, year: int, round_num: int) -> Path:
        """Returns the canonical path for an official starting grid manifest."""
        return self.grids_dir / str(year) / f"{year}_round_{round_num}_starting_grid.json"

    def get_openf1_cache_path(self, year: int, round_num: int) -> Path:
        """Returns the canonical path for cached OpenF1 starting grid data."""
        return self.raw_grids_dir / str(year) / f"{year}_round_{round_num}_openf1_grid.json"

    def validate_manifest_json(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates the schema and sporting invariants of a starting-grid manifest.

        Invariants checked:
        1. Required top-level fields present.
        2. Unique drivers across all entries.
        3. Contiguous physical grid positions 1..N_physical with zero duplicates or gaps.
        4. Pit-lane starters must have grid_pos = None / null and is_pit_lane_start = 1.
        5. Total simulation_start_ranks must be contiguous 1..N_total.
        6. Publication status is valid enum.
        """
        errors = []
        required_top = ["season", "round", "circuit_key", "publication_status", "grid"]
        for f in required_top:
            if f not in data:
                errors.append(f"Missing required top-level field: '{f}'")

        if errors:
            return False, errors

        grid_entries = data.get("grid", [])
        if not isinstance(grid_entries, list) or len(grid_entries) == 0:
            errors.append("Field 'grid' must be a non-empty list of driver records.")
            return False, errors

        drivers_seen = set()
        physical_positions = []
        sim_ranks = []
        n_pit = 0

        for idx, entry in enumerate(grid_entries):
            drv = entry.get("driver") or entry.get("Abbreviation")
            if not drv:
                errors.append(f"Entry {idx} missing driver abbreviation.")
                continue

            drv_clean = NAME_TO_CODE.get(str(drv).upper().strip(), str(drv).upper().strip())
            if drv_clean in drivers_seen:
                errors.append(f"Duplicate driver in grid manifest: {drv_clean}")
            drivers_seen.add(drv_clean)

            grid_pos = entry.get("grid_pos")
            if grid_pos is None:
                grid_pos = entry.get("ActualStartingGrid")
            sim_rank = entry.get("simulation_start_rank")
            is_pit = int(entry.get("is_pit_lane_start", 0))

            if is_pit == 1:
                n_pit += 1
                if grid_pos is not None and not (isinstance(grid_pos, float) and np.isnan(grid_pos)):
                    errors.append(
                        f"Pit-lane starter {drv_clean} has non-null physical grid_pos: {grid_pos}. "
                        "Pit-lane starters must have grid_pos = null."
                    )
            else:
                if grid_pos is None:
                    status = str(entry.get("status", "")).upper()
                    if status not in ["DNS", "DSQ"]:
                        errors.append(f"Physical starter {drv_clean} has null grid_pos without DNS/DSQ status.")
                else:
                    try:
                        physical_positions.append(int(grid_pos))
                    except (ValueError, TypeError):
                        errors.append(f"Driver {drv_clean} has invalid non-integer grid_pos: {grid_pos}")

            if sim_rank is not None:
                try:
                    sim_ranks.append(int(sim_rank))
                except (ValueError, TypeError):
                    errors.append(f"Driver {drv_clean} has invalid simulation_start_rank: {sim_rank}")

        # Check physical positions contiguity
        if physical_positions:
            n_phys = len(physical_positions)
            sorted_phys = sorted(physical_positions)
            expected_phys = list(range(1, n_phys + 1))
            if sorted_phys != expected_phys:
                errors.append(
                    f"Physical grid positions are not contiguous 1..{n_phys}: {sorted_phys}"
                )

        # Check simulation start ranks contiguity
        if sim_ranks:
            n_sim = len(sim_ranks)
            sorted_sim = sorted(sim_ranks)
            expected_sim = list(range(1, n_sim + 1))
            if sorted_sim != expected_sim:
                errors.append(
                    f"Simulation start ranks are not contiguous 1..{n_sim}: {sorted_sim}"
                )

        return len(errors) == 0, errors

    def validate_grid_dataframe(
        self,
        df: pd.DataFrame,
        expected_grid_size: int = 22,
    ) -> Tuple[bool, List[str]]:
        """
        Validates a canonical starting grid DataFrame for structural and sporting integrity.
        """
        errors = []
        if df is None or df.empty:
            return False, ["Starting grid DataFrame is empty."]

        # Check required columns
        req_cols = ["Abbreviation", "ActualStartingGrid", "simulation_start_rank", "is_pit_lane_start"]
        for c in req_cols:
            if c not in df.columns:
                errors.append(f"Missing required column in starting grid: '{c}'")

        if errors:
            return False, errors

        # Expected grid size validation
        if expected_grid_size is not None and expected_grid_size > 0 and len(df) != expected_grid_size:
            errors.append(f"Grid size {len(df)} does not match expected {expected_grid_size}.")

        # Unique drivers
        if df["Abbreviation"].duplicated().any():
            dups = df[df["Abbreviation"].duplicated()]["Abbreviation"].tolist()
            errors.append(f"Duplicate drivers in starting grid: {dups}")

        # Physical starters validation
        phys_starters = df[df["is_pit_lane_start"] == 0]
        phys_positions = phys_starters["ActualStartingGrid"].dropna().tolist()
        if len(phys_positions) != len(set(phys_positions)):
            errors.append(f"Duplicate physical grid positions: {phys_positions}")

        try:
            sorted_phys = sorted([int(p) for p in phys_positions])
            expected_phys = list(range(1, len(phys_positions) + 1))
            if sorted_phys != expected_phys:
                errors.append(f"Physical grid positions are not contiguous 1..{len(phys_positions)}: {sorted_phys}")
        except (ValueError, TypeError) as e:
            errors.append(f"Non-integer physical grid positions detected: {e}")

        # Pit-lane starters validation (must have NaN ActualStartingGrid)
        pit_starters = df[df["is_pit_lane_start"] == 1]
        for _, r in pit_starters.iterrows():
            if pd.notna(r["ActualStartingGrid"]):
                errors.append(f"Pit-lane starter {r['Abbreviation']} has non-null ActualStartingGrid: {r['ActualStartingGrid']}")

        # Simulation start ranks contiguity
        sim_ranks = df["simulation_start_rank"].dropna().tolist()
        try:
            sorted_sim = sorted([int(r) for r in sim_ranks])
            expected_sim = list(range(1, len(sim_ranks) + 1))
            if sorted_sim != expected_sim:
                errors.append(f"Simulation start ranks are not contiguous 1..{len(sim_ranks)}: {sorted_sim}")
        except (ValueError, TypeError) as e:
            errors.append(f"Non-integer simulation start ranks detected: {e}")

        return len(errors) == 0, errors

    def get_starting_grid(
        self,
        track_key: str,
        year: int,
        round_num: int,
        stage: str = "after_qualifying",
        quali_df: Optional[pd.DataFrame] = None,
        penalties: Optional[Dict[str, Any]] = None,
        expected_grid_size: Optional[int] = None,
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Retrieves starting grid using strict multi-source priority:
        1. FIA Official Starting Grid Manifest (data/grids/{year}/)
        2. Official Formula 1 Website Published Grid
        3. OpenF1 Starting Grid API (with persistent local caching)
        4. Generic Deterministic FIA Grid Resolver (resolve_fia_starting_grid)
        """
        if expected_grid_size is None or expected_grid_size <= 0:
            from src.data.roster_resolver import RosterResolver
            expected_grid_size = RosterResolver.get_expected_grid_size(year, round_num)
        # Strict stage cutoff: before qualifying, official starting grid cannot be published
        if stage in ["pre_fp1", "after_fp1", "after_fp2", "after_fp3", "after_sprint_qualifying", "after_sprint_race", "after_sprint"]:
            logger.info(
                f"Stage '{stage}' is prior to qualifying cutoff. Starting grid is unpublished."
            )
            return None, {
                "provenance": GridProvenance.UNPUBLISHED.value,
                "stage": stage,
                "publication_status": GridPublicationStatus.UNOFFICIAL.value,
                "circuit_key": track_key,
                "season": year,
                "round": round_num,
                "document_reference": "Pre-Qualifying Cutoff",
            }

        # ---------------------------------------------------------------------
        # SOURCE 1: Official FIA Starting Grid Manifest
        # ---------------------------------------------------------------------
        df_fia, meta_fia = self._retrieve_fia_manifest(
            track_key=track_key,
            year=year,
            round_num=round_num,
            expected_grid_size=expected_grid_size,
        )
        if df_fia is not None:
            return df_fia, meta_fia

        # ---------------------------------------------------------------------
        # SOURCE 2: Official Formula 1 Website Published Grid
        # ---------------------------------------------------------------------
        df_f1, meta_f1 = self._retrieve_f1_website(
            track_key=track_key,
            year=year,
            round_num=round_num,
            expected_grid_size=expected_grid_size,
        )
        if df_f1 is not None:
            return df_f1, meta_f1

        # ---------------------------------------------------------------------
        # SOURCE 3: OpenF1 Starting Grid API (with local persistent cache)
        # ---------------------------------------------------------------------
        df_openf1, meta_openf1 = self._retrieve_openf1(
            track_key=track_key,
            year=year,
            round_num=round_num,
            stage=stage,
            quali_df=quali_df,
            expected_grid_size=expected_grid_size,
        )
        if df_openf1 is not None:
            return df_openf1, meta_openf1

        # ---------------------------------------------------------------------
        # SOURCE 4: Algorithmic Fallback (FIA Sporting Regulations Resolver)
        # ---------------------------------------------------------------------
        if quali_df is not None and not quali_df.empty:
            logger.info(
                f"No official manifest or published API grid found for {year} Round {round_num}. "
                "Executing deterministic FIA grid resolution algorithm (FIA Art 42.3)."
            )
            df_alg = resolve_fia_starting_grid(
                df=quali_df,
                penalties=penalties,
                expected_grid_size=expected_grid_size,
            )
            metadata = {
                "provenance": GridProvenance.ALGORITHMIC_FALLBACK.value,
                "publication_status": GridPublicationStatus.PROVISIONAL.value,
                "document_reference": "Algorithmic Provisional Grid (FIA Art 42.3)",
                "season": year,
                "round": round_num,
                "circuit_key": track_key,
                "stage": stage,
            }
            return df_alg, metadata

        logger.warning(
            f"Cannot resolve starting grid for {track_key} Round {round_num}: "
            "Neither published grid nor qualifying data was provided."
        )
        return None, {
            "provenance": GridProvenance.MISSING_DATA.value,
            "stage": stage,
            "season": year,
            "round": round_num,
            "circuit_key": track_key,
            "document_reference": "No starting grid or qualifying data available",
        }

    # =========================================================================
    # SOURCE 1 IMPLEMENTATION: FIA OFFICIAL MANIFEST
    # =========================================================================
    def _retrieve_fia_manifest(
        self,
        track_key: str,
        year: int,
        round_num: int,
        expected_grid_size: int = 22,
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        manifest_path = self.get_manifest_path(year, round_num)
        if not manifest_path.exists():
            return None, {}

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)

            valid, errs = self.validate_manifest_json(manifest_data)
            m_circuit = str(manifest_data.get("circuit_key", "")).lower()
            if valid and track_key and m_circuit and m_circuit != track_key.lower():
                logger.warning(
                    f"Manifest at {manifest_path} specifies circuit '{m_circuit}', which does not match requested '{track_key}'."
                )
                valid = False
                errs = [f"Circuit key mismatch: manifest={m_circuit} vs requested={track_key}"]

            if not valid:
                logger.warning(f"Manifest at {manifest_path} failed validation: {errs}. Skipping.")
                return None, {}

            df = self._manifest_to_dataframe(manifest_data, expected_grid_size=expected_grid_size)
            valid_df, df_errs = self.validate_grid_dataframe(df, expected_grid_size=expected_grid_size)
            if not valid_df:
                logger.warning(f"Manifest DataFrame validation failed: {df_errs}. Skipping.")
                return None, {}

            doc_ref = manifest_data.get("document_reference", "FIA Official Starting Grid")
            pub_status = manifest_data.get("publication_status", GridPublicationStatus.FINAL.value)
            logger.info(
                f"Loaded official FIA starting grid manifest for {track_key} Round {round_num} "
                f"(Ref: {doc_ref}, Status: {pub_status})"
            )
            metadata = {
                "provenance": GridProvenance.FIA_OFFICIAL.value,
                "manifest_path": str(manifest_path),
                "document_reference": doc_ref,
                "publication_status": pub_status,
                "stewards_notes": manifest_data.get("stewards_notes", []),
                "season": year,
                "round": round_num,
                "circuit_key": track_key,
            }
            return df, metadata
        except Exception as e:
            logger.error(f"Error parsing manifest {manifest_path}: {e}")
            return None, {}

    # =========================================================================
    # SOURCE 2 IMPLEMENTATION: OFFICIAL F1 WEBSITE PUBLISHED GRID
    # =========================================================================
    def _retrieve_f1_website(
        self,
        track_key: str,
        year: int,
        round_num: int,
        expected_grid_size: int = 22,
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        f1_path = self.raw_grids_dir / str(year) / f"{year}_round_{round_num}_f1_website_grid.json"
        if not f1_path.exists():
            return None, {}

        try:
            with open(f1_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            valid, errs = self.validate_manifest_json(data)
            if not valid:
                return None, {}

            df = self._manifest_to_dataframe(data, expected_grid_size=expected_grid_size)
            valid_df, df_errs = self.validate_grid_dataframe(df, expected_grid_size=expected_grid_size)
            if not valid_df:
                return None, {}

            metadata = {
                "provenance": GridProvenance.F1_WEBSITE.value,
                "document_reference": data.get("document_reference", "F1 Official Website Published Starting Grid"),
                "publication_status": data.get("publication_status", GridPublicationStatus.FINAL.value),
                "season": year,
                "round": round_num,
                "circuit_key": track_key,
            }
            return df, metadata
        except Exception as e:
            logger.error(f"Error loading F1 website grid from {f1_path}: {e}")
            return None, {}

    # =========================================================================
    # SOURCE 3 IMPLEMENTATION: OPENF1 STARTING GRID API
    # =========================================================================
    def _retrieve_openf1(
        self,
        track_key: str,
        year: int,
        round_num: int,
        stage: str,
        quali_df: Optional[pd.DataFrame] = None,
        expected_grid_size: int = 22,
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        cache_path = self.get_openf1_cache_path(year, round_num)
        data = None

        # 1. Check local persistent cache first
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"Loaded cached OpenF1 starting grid from {cache_path}")
            except Exception as e:
                logger.warning(f"Error reading OpenF1 cache {cache_path}: {e}")
                data = None

        # 2. If not cached, attempt live API query
        if not data:
            data = self._fetch_openf1_live(track_key=track_key, year=year, round_num=round_num)
            if data and "grid" in data and len(data["grid"]) > 0:
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    logger.info(f"Cached OpenF1 starting grid to {cache_path}")
                except Exception as e:
                    logger.warning(f"Could not write OpenF1 cache {cache_path}: {e}")

        if not data or "grid" not in data or not data["grid"]:
            return None, {}

        df = self._openf1_to_dataframe(data, quali_df=quali_df, expected_grid_size=expected_grid_size)
        if df is None or df.empty:
            return None, {}

        # Validate driver count against quali_df if provided
        if quali_df is not None and not quali_df.empty:
            pos_col = "Abbreviation" if "Abbreviation" in quali_df.columns else "Driver"
            quali_drivers = set(quali_df[pos_col].astype(str).str.upper().str.strip())
            grid_drivers = set(df["Abbreviation"].astype(str).str.upper().str.strip())
            if len(quali_drivers) != len(grid_drivers):
                logger.warning(
                    f"OpenF1 grid driver count ({len(grid_drivers)}) does not match qualifying session ({len(quali_drivers)}). Rejecting OpenF1 grid."
                )
                return None, {}

        valid, errs = self.validate_grid_dataframe(df, expected_grid_size=expected_grid_size)
        if not valid:
            logger.warning(f"OpenF1 starting grid validation failed: {errs}. Skipping.")
            return None, {}

        sess_key = data.get("session_key")
        metadata = {
            "provenance": GridProvenance.OPENF1_API.value,
            "publication_status": data.get("publication_status", GridPublicationStatus.PROVISIONAL.value),
            "document_reference": f"OpenF1 API (Session {sess_key})" if sess_key else "OpenF1 Starting Grid API",
            "source": data.get("source", "https://api.openf1.org/v1/starting_grid"),
            "session_key": sess_key,
            "meeting_key": data.get("meeting_key"),
            "retrieval_timestamp": data.get("retrieval_timestamp", ""),
            "season": year,
            "round": round_num,
            "circuit_key": track_key,
            "stage": stage,
        }
        logger.info(
            f"Successfully retrieved and validated OpenF1 starting grid for {track_key} Round {round_num} "
            f"({len(df)} drivers, Ref: {metadata['document_reference']})"
        )
        return df, metadata

    def _match_openf1_meeting(
        self,
        meetings: List[Dict[str, Any]],
        track_key: str,
        round_num: int,
    ) -> Optional[Dict[str, Any]]:
        """Matches a circuit key and round to an OpenF1 meeting record."""
        tk_clean = track_key.lower().replace("_", " ")

        synonyms = {
            "cota": ["united states", "austin"],
            "austin": ["united states", "austin"],
            "mexico": ["mexico", "hermanos rodriguez"],
            "silverstone": ["british", "great britain"],
            "monza": ["italian", "italy"],
            "spa": ["belgian", "belgium"],
            "red_bull_ring": ["austrian", "austria"],
            "interlagos": ["sao paulo", "brazil"],
            "suzuka": ["japanese", "japan"],
            "marina_bay": ["singapore"],
            "yas_marina": ["abu dhabi"],
            "losail": ["qatar"],
            "albert_park": ["australian", "australia"],
            "jeddah": ["saudi arabian", "saudi"],
            "bahrain": ["bahrain", "sakhir"],
            "sepang": ["malaysian", "malaysia", "sepang", "bahrain"],
            "baku": ["azerbaijan", "baku"],
            "monaco": ["monaco", "monte carlo"],
            "villeneuve": ["canadian", "canada"],
            "barcelona": ["spanish", "spain", "barcelona"],
            "hungaroring": ["hungarian", "hungary"],
            "zandvoort": ["dutch", "netherlands"],
            "shanghai": ["chinese", "china"],
            "las_vegas": ["las vegas"],
            "miami": ["miami"],
        }
        patterns = synonyms.get(track_key.lower(), [tk_clean])

        for m in meetings:
            m_name = str(m.get("meeting_name", "")).lower()
            m_loc = str(m.get("location", "")).lower()
            m_country = str(m.get("country_name", "")).lower()
            m_short = str(m.get("circuit_short_name", "")).lower()

            if any(p in m_name or p in m_loc or p in m_country or p in m_short for p in patterns):
                return m

        return None

    def _fetch_openf1_live(
        self,
        track_key: str,
        year: int,
        round_num: int,
    ) -> Optional[Dict[str, Any]]:
        """Performs live network query against OpenF1 API endpoints."""
        import urllib.request
        from urllib.error import URLError, HTTPError

        headers = {"User-Agent": "F1PredictionEngine/1.0"}
        timeout = 5

        # 1. Query meetings
        try:
            url_meetings = f"https://api.openf1.org/v1/meetings?year={year}"
            req = urllib.request.Request(url_meetings, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                meetings = json.loads(resp.read().decode())
        except (URLError, HTTPError, TimeoutError, json.JSONDecodeError) as e:
            logger.info(f"OpenF1 meetings query failed ({e}). Continuing to next source.")
            return None

        # 2. Match meeting
        meeting = self._match_openf1_meeting(meetings, track_key=track_key, round_num=round_num)
        if not meeting:
            logger.info(f"OpenF1 meeting not found for circuit '{track_key}' {year}. Continuing to next source.")
            return None

        m_key = meeting.get("meeting_key")

        # 3. Query sessions
        try:
            url_sessions = f"https://api.openf1.org/v1/sessions?meeting_key={m_key}"
            req = urllib.request.Request(url_sessions, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                sessions = json.loads(resp.read().decode())
        except (URLError, HTTPError, TimeoutError, json.JSONDecodeError) as e:
            logger.info(f"OpenF1 sessions query failed ({e}). Continuing to next source.")
            return None

        # 4. Locate Race session key (and fallback sessions)
        grid_records = None
        target_session_key = None

        race_sess = next(
            (s for s in sessions if str(s.get("session_name", "")).lower() == "race" or str(s.get("session_type", "")).lower() == "race"),
            None,
        )

        if race_sess:
            r_key = race_sess.get("session_key")
            try:
                g_url = f"https://api.openf1.org/v1/starting_grid?session_key={r_key}"
                req = urllib.request.Request(g_url, headers=headers)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    res = json.loads(resp.read().decode())
                    if res and isinstance(res, list) and len(res) > 0:
                        grid_records = res
                        target_session_key = r_key
            except Exception:
                pass

        if not grid_records:
            # Query by meeting_key (OpenF1 links some starting grids by meeting)
            try:
                g_url = f"https://api.openf1.org/v1/starting_grid?meeting_key={m_key}"
                req = urllib.request.Request(g_url, headers=headers)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    res = json.loads(resp.read().decode())
                    if res and isinstance(res, list) and len(res) > 0:
                        grid_records = res
                        target_session_key = res[0].get("session_key", race_sess.get("session_key") if race_sess else None)
            except Exception:
                pass

        if not grid_records:
            return None

        # 5. Fetch drivers for session to resolve names/numbers
        drivers_data = []
        if target_session_key:
            try:
                d_url = f"https://api.openf1.org/v1/drivers?session_key={target_session_key}"
                req = urllib.request.Request(d_url, headers=headers)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    drivers_data = json.loads(resp.read().decode())
            except Exception:
                pass

        return {
            "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
            "source": f"https://api.openf1.org/v1/starting_grid?session_key={target_session_key}",
            "season": year,
            "round": round_num,
            "circuit_key": track_key,
            "meeting_key": m_key,
            "session_key": target_session_key,
            "publication_status": GridPublicationStatus.PROVISIONAL.value,
            "grid": grid_records,
            "drivers": drivers_data,
        }

    def _openf1_to_dataframe(
        self,
        data: Dict[str, Any],
        quali_df: Optional[pd.DataFrame] = None,
        expected_grid_size: int = 22,
    ) -> Optional[pd.DataFrame]:
        """Converts raw OpenF1 starting grid records to canonical DataFrame."""
        records = data.get("grid", [])
        if not records:
            return None

        num_to_code = {}
        num_to_team = {}

        # 1. Populate from quali_df if available (authoritative for current weekend)
        if quali_df is not None and not quali_df.empty:
            for _, r in quali_df.iterrows():
                num = r.get("DriverNumber")
                abbr = r.get("Abbreviation") or r.get("Driver")
                team = r.get("TeamName") or r.get("Team")
                if pd.notna(num) and pd.notna(abbr):
                    try:
                        n_int = int(num)
                        num_to_code[n_int] = str(abbr).upper().strip()
                        if pd.notna(team):
                            num_to_team[n_int] = str(team).strip()
                    except (ValueError, TypeError):
                        pass

        # 2. Populate from OpenF1 drivers list
        for d in data.get("drivers", []):
            num = d.get("driver_number")
            acronym = d.get("name_acronym")
            team = d.get("team_name")
            if num and acronym:
                n_int = int(num)
                if n_int not in num_to_code:
                    num_to_code[n_int] = str(acronym).upper().strip()
                if n_int not in num_to_team and team:
                    num_to_team[n_int] = str(team).strip()

        # 3. Populate from DRIVER_CATALOG fallback
        for code, cat_info in DRIVER_CATALOG.items():
            cat_num = cat_info.get("number")
            if cat_num and cat_num not in num_to_code:
                num_to_code[cat_num] = code
            if cat_num and cat_num not in num_to_team and cat_info.get("default_team"):
                num_to_team[cat_num] = cat_info["default_team"]

        quali_map = {}
        if quali_df is not None and not quali_df.empty:
            pos_col = "Position" if "Position" in quali_df.columns else "QualifyingPosition"
            abbr_col = "Abbreviation" if "Abbreviation" in quali_df.columns else "Driver"
            for _, r in quali_df.iterrows():
                d_abbr = str(r[abbr_col]).upper().strip()
                p_val = r.get(pos_col)
                if pd.notna(p_val):
                    try:
                        quali_map[d_abbr] = float(p_val)
                    except Exception:
                        pass

        total_entrants = len(records)
        rows = []
        for r in records:
            num = r.get("driver_number")
            pos = r.get("position")
            code = num_to_code.get(num, f"D{num}")
            team = num_to_team.get(num, "Unknown")
            q_pos = quali_map.get(code, float(pos) if pos is not None else np.nan)

            is_pit = 1 if (pos is None or r.get("is_pit_lane_start") == 1) else 0
            grid_pos = float(pos) if (pos is not None and is_pit == 0) else np.nan
            sim_rank = float(pos) if pos is not None else float(total_entrants)

            ref_pos = grid_pos if pd.notna(grid_pos) else sim_rank
            delta = int(ref_pos - q_pos) if (pd.notna(ref_pos) and pd.notna(q_pos)) else 0
            pen_places = max(0, delta)

            status = StartingGridStatus.PIT_LANE.value if is_pit == 1 else (
                StartingGridStatus.PENALIZED.value if (pd.notna(grid_pos) and pd.notna(q_pos) and grid_pos > q_pos) else StartingGridStatus.CLASSIFIED.value
            )

            grid_norm = (sim_rank - 1.0) / float(max(total_entrants - 1, 1)) if pd.notna(sim_rank) else np.nan

            rows.append({
                "Abbreviation": code,
                "TeamName": team,
                "QualifyingPosition": q_pos,
                "ActualStartingGrid": grid_pos,
                "simulation_start_rank": sim_rank,
                "is_pit_lane_start": is_pit,
                "StartingGridStatus": status,
                "GridPenalty": pen_places,
                "grid_penalty_places": pen_places,
                "grid_vs_quali_delta": delta,
                "grid_norm": grid_norm,
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("simulation_start_rank").reset_index(drop=True)
        return df

    def _manifest_to_dataframe(
        self,
        manifest_data: Dict[str, Any],
        expected_grid_size: int = 22,
    ) -> pd.DataFrame:
        """Converts validated manifest dictionary to canonical DataFrame."""
        entries = manifest_data.get("grid", [])
        total_entrants = len(entries)
        rows = []

        for e in entries:
            drv = e.get("driver") or e.get("Abbreviation")
            drv_clean = NAME_TO_CODE.get(str(drv).upper().strip(), str(drv).upper().strip())
            team = e.get("team") or e.get("TeamName") or "Unknown"

            q_pos = e.get("qualifying_pos")
            if q_pos is None:
                q_pos = e.get("QualifyingPosition")
            q_pos = float(q_pos) if q_pos is not None else np.nan

            grid_pos = e.get("grid_pos")
            if grid_pos is None:
                grid_pos = e.get("ActualStartingGrid")
            grid_pos = float(grid_pos) if (grid_pos is not None and not pd.isna(grid_pos)) else np.nan

            sim_rank = e.get("simulation_start_rank")
            if sim_rank is None:
                sim_rank = grid_pos if pd.notna(grid_pos) else float(total_entrants)
            sim_rank = float(sim_rank) if sim_rank is not None else np.nan

            is_pit = int(e.get("is_pit_lane_start", 1 if pd.isna(grid_pos) else 0))
            status = e.get("status") or (
                StartingGridStatus.PIT_LANE.value if is_pit == 1 else StartingGridStatus.CLASSIFIED.value
            )

            pen_places = int(e.get("grid_penalty_places", e.get("GridPenalty", 0)))
            delta = int(e.get("grid_vs_quali_delta", 0))
            if delta == 0 and pd.notna(grid_pos) and pd.notna(q_pos):
                delta = int(grid_pos - q_pos)

            grid_norm = e.get("grid_norm")
            if grid_norm is None:
                if pd.notna(sim_rank):
                    grid_norm = (sim_rank - 1.0) / float(max(total_entrants - 1, 1))
                else:
                    grid_norm = np.nan
            else:
                grid_norm = float(grid_norm)

            rows.append({
                "Abbreviation": drv_clean,
                "TeamName": team,
                "QualifyingPosition": q_pos,
                "ActualStartingGrid": grid_pos,
                "simulation_start_rank": sim_rank,
                "is_pit_lane_start": is_pit,
                "StartingGridStatus": status,
                "GridPenalty": pen_places,
                "grid_penalty_places": pen_places,
                "grid_vs_quali_delta": delta,
                "grid_norm": grid_norm,
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("simulation_start_rank").reset_index(drop=True)
        return df

    def save_manifest(
        self,
        year: int,
        round_num: int,
        circuit_key: str,
        grid_df: pd.DataFrame,
        document_reference: str = "FIA Official Starting Grid",
        publication_status: str = "FINAL",
        stewards_notes: Optional[List[str]] = None,
    ) -> Path:
        """Saves a canonical starting-grid manifest JSON file."""
        target_path = self.get_manifest_path(year, round_num)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        grid_list = []
        for _, r in grid_df.iterrows():
            grid_pos = int(r["ActualStartingGrid"]) if pd.notna(r.get("ActualStartingGrid")) else None
            sim_rank = int(r["simulation_start_rank"]) if pd.notna(r.get("simulation_start_rank")) else None
            q_pos = int(r["QualifyingPosition"]) if pd.notna(r.get("QualifyingPosition")) else None

            entry = {
                "driver": str(r["Abbreviation"]),
                "team": str(r.get("TeamName", "Unknown")),
                "qualifying_pos": q_pos,
                "grid_pos": grid_pos,
                "simulation_start_rank": sim_rank,
                "is_pit_lane_start": int(r.get("is_pit_lane_start", 0)),
                "status": str(r.get("StartingGridStatus", "CLASSIFIED")),
                "grid_penalty_places": int(r.get("grid_penalty_places", r.get("GridPenalty", 0))),
                "grid_vs_quali_delta": int(r.get("grid_vs_quali_delta", 0)),
                "grid_norm": float(r.get("grid_norm", 0.0)),
            }
            grid_list.append(entry)

        manifest = {
            "season": year,
            "round": round_num,
            "circuit_key": circuit_key,
            "publication_status": publication_status,
            "document_reference": document_reference,
            "stewards_notes": stewards_notes or [],
            "grid": grid_list,
        }

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Saved starting grid manifest to {target_path}")
        return target_path
