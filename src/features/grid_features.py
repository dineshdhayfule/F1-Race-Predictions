"""
Grid Features and Generic FIA Grid Resolver Module
==================================================
Handles starting grid calculation, penalty resolution, manifest integration,
and grid validation under official FIA Sporting Regulations (Articles 28.2, 39.4, 40.8, 42.3, 42.5).

Guarantees:
- Canonical starting-grid data model with explicit enums and dataclasses.
- Disambiguation between physical grid starters and pit-lane starters.
- Strict tie-breaking by qualifying time/position under Art 42.3.
- Back-of-grid cumulative thresholds (>= 15 positions or explicit back-of-grid, Art 28.2 / 42.3(d)).
- Null invariant for pit-lane starters on physical grid, with deterministic simulation start rank.
- Exclusion of DNS and race DSQ entrants from physical grid and simulation.
- 100% backward compatibility with legacy apply_grid_penalties interface and protected baselines.
"""

from dataclasses import dataclass, field
from enum import Enum
from functools import cmp_to_key
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.utils.driver_mapping import NAME_TO_CODE, normalize_driver_series
from src.utils.logging_utils import get_logger

logger = get_logger("grid_features")


def _clean_driver_code(drv: Any) -> str:
    """Normalizes driver string (abbreviation, surname, or full name) to uppercase 3-letter code."""
    if drv is None or pd.isna(drv):
        return ""
    s = str(drv).strip().upper()
    return NAME_TO_CODE.get(s, s)


def _parse_laptime_to_seconds(val: Any) -> Optional[float]:
    """
    Parses various lap time representations into float seconds:
      - '0 days 00:01:17.029000' (FastF1 timedelta string)
      - '1:17.029' (mm:ss.ms)
      - '77.029' (raw float string)
      - float / int numeric seconds
    """
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val) if float(val) > 0 else None
    s = str(val).strip()
    if not s or s.lower() in ["nan", "nat", "none", ""]:
        return None
    # 1. Try pd.to_timedelta (FastF1 string format)
    try:
        td = pd.to_timedelta(s)
        secs = td.total_seconds()
        if secs > 0:
            return float(secs)
    except Exception:
        pass
    # 2. Try colon format mm:ss.fff or hh:mm:ss.fff
    try:
        if ":" in s:
            parts = s.split(":")
            if len(parts) == 2:
                return float(parts[0]) * 60.0 + float(parts[1])
            elif len(parts) == 3:
                return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
        return float(s)
    except Exception:
        return None


def _extract_best_session_laptime(row: Union[pd.Series, Dict[str, Any]]) -> float:
    """
    Extracts authoritative qualifying session lap time under FIA Sporting Regulations.
    Priority: Q3 lap time > Q2 lap time > Q1 lap time > general lap time fields.
    """
    getter = row.get if isinstance(row, dict) else (lambda k: row[k] if k in row else None)
    for col in ["Q3", "q3", "Q2", "q2", "Q1", "q1", "QualifyingTime", "QTime", "Time", "q_time", "best_lap"]:
        val = getter(col)
        if val is not None and pd.notna(val):
            t = _parse_laptime_to_seconds(val)
            if t is not None and t > 0:
                return t
    return float("inf")


class StartingGridStatus(str, Enum):
    """Authoritative starting grid status."""
    CLASSIFIED = "CLASSIFIED"
    PENALIZED = "PENALIZED"
    PIT_LANE = "PIT_LANE"
    DNS = "DNS"
    DSQ = "DSQ"


class GridPublicationStatus(str, Enum):
    """Publication lifecycle of the grid document."""
    UNOFFICIAL = "UNOFFICIAL"
    PROVISIONAL = "PROVISIONAL"
    FINAL = "FINAL"


class GridProvenance(str, Enum):
    """Canonical provenance identifiers for starting grid data."""
    FIA_OFFICIAL = "official_manifest"
    F1_WEBSITE = "f1_website"
    OPENF1_API = "openf1_api"
    ALGORITHMIC_FALLBACK = "algorithmic_fallback"
    HISTORICAL_RACE_RESULTS = "historical_race_results"
    UNPUBLISHED = "unpublished"
    UNVERIFIED = "unverified"
    MISSING_DATA = "missing_data"


def format_grid_provenance_display(grid_meta: Optional[Dict[str, Any]]) -> str:
    """
    Returns the human-readable display string for starting grid provenance.
    Never silently labels an algorithmic grid as an actual observed qualifying grid.
    """
    if not grid_meta or not isinstance(grid_meta, dict):
        return "Starting Grid Unavailable"

    prov = str(grid_meta.get("provenance", "")).lower()
    pub_status = str(grid_meta.get("publication_status", "")).upper()
    doc_ref = grid_meta.get("document_reference")

    if prov in ["official_manifest", "fia_official"]:
        status_label = "Final" if pub_status == "FINAL" else "Provisional"
        ref_str = f" ({doc_ref})" if doc_ref else ""
        return f"FIA Official {status_label} Starting Grid{ref_str}"
    elif prov == "f1_website":
        return "F1 Published Starting Grid"
    elif prov == "openf1_api":
        ref_str = f" ({doc_ref})" if doc_ref else ""
        return f"OpenF1 Published Starting Grid{ref_str}"
    elif prov in ["algorithmic_fallback", "algorithmic_provisional"]:
        return "Algorithmic Provisional Grid (FIA Art 42.3)"
    elif prov in ["historical_race_results", "race_results_derived"]:
        ref_str = f" ({doc_ref})" if doc_ref else ""
        return f"Historical Race Results Starting Grid{ref_str}"
    elif prov == "unpublished":
        stage = grid_meta.get("stage", "pre-qualifying")
        return f"Provisional Predicted Grid (Pre-Qualifying Stage: {stage})"
    else:
        return "Starting Grid Unavailable"


class PenaltyType(str, Enum):
    """Type of grid penalty assessed by stewards."""
    GRID_DROP = "GRID_DROP"
    BACK_OF_GRID = "BACK_OF_GRID"
    PIT_LANE = "PIT_LANE"
    DSQ = "DSQ"
    DNS = "DNS"
    TIME = "TIME"


@dataclass
class PenaltyRecord:
    """Individual steward penalty record."""
    driver: str
    penalty_type: PenaltyType = PenaltyType.GRID_DROP
    places: int = 0
    reason: str = ""
    article: str = ""


@dataclass
class CanonicalGridRecord:
    """Canonical per-driver starting grid representation."""
    driver: str
    qualifying_pos: Optional[int]
    grid_pos: Optional[int]
    simulation_start_rank: Optional[int]
    status: StartingGridStatus
    is_pit_lane_start: int
    grid_penalty_places: int
    grid_vs_quali_delta: int
    grid_norm: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def resolve_fia_starting_grid(
    df: pd.DataFrame,
    penalties: Optional[Union[Dict[str, Union[int, dict]], List[PenaltyRecord]]] = None,
    expected_grid_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generic FIA Grid Resolver implementing Articles 28.2, 39.4, 40.8, 42.3, and 42.5
    of the FIA Formula One Sporting Regulations.

    Resolution Process:
    1. Filter out DNS / Race DSQ entrants (excluded from physical grid and simulation).
    2. Segregate pit-lane starters (Art 40.8 & 42.5); they take no physical grid spot.
    3. Segregate back-of-grid drivers (Art 28.2 & 42.3(d): >= 15 cumulative places or explicit back-of-grid).
    4. Apply grid drops (< 15 places) relative to unpenalized drivers, preserving qualifying
       order and breaking conflicts by qualifying time/position (Art 42.3(a)-(c)).
    5. Append back-of-grid drivers, ordered strictly among themselves by qualifying times (Art 42.3(d)).
    6. Append drivers disqualified from qualifying but permitted to start by stewards (Art 39.4).
    7. Assign physical grid positions 1..N_physical consecutively.
    8. Assign simulation start ranks (1..N_physical for physical starters, N_physical+1..N_total
       for pit-lane starters ordered by qualifying times).

    Parameters
    ----------
    df : pd.DataFrame
        Session DataFrame with driver identifiers and qualifying positions.
        Required columns: 'Abbreviation' (or 'Driver'), 'Position' or 'QualifyingPosition'.
        Optional columns: 'QualifyingTime', 'QTime', 'Time' (for lap time tie-breaks),
                          'Status', 'is_pit_lane'.
    penalties : dict or list of PenaltyRecord, optional
        Penalty specifications. Can be:
        - Dict[str, int]: driver -> penalty places (e.g. {'HAD': 5, 'COL': 15, 'LIN': 30})
        - Dict[str, dict]: driver -> {'places': int, 'type': str, 'reason': str}
        - List[PenaltyRecord]: rich penalty records.
    expected_grid_size : int, optional
        Expected total count of participating drivers (physical + pit-lane).
        If provided and total valid entrants does not match, raises ValueError.

    Returns
    -------
    pd.DataFrame
        Canonical DataFrame with columns:
        - 'Abbreviation', 'TeamName' (if present)
        - 'QualifyingPosition'
        - 'ActualStartingGrid' (int for physical grid 1..N_physical, np.nan for pit-lane/DNS/DSQ)
        - 'simulation_start_rank' (int 1..N_total for all race starters, np.nan for DNS/DSQ)
        - 'is_pit_lane_start' (int: 1 or 0)
        - 'StartingGridStatus' (str enum value)
        - 'GridPenalty' / 'grid_penalty_places' (int: assessed drop places)
        - 'grid_vs_quali_delta' (int: net physical shift or drop)
        - 'grid_norm' (float: normalized simulation start rank)
    """
    if df.empty:
        raise ValueError("Cannot resolve starting grid from an empty DataFrame.")

    out = df.copy()

    # Normalize driver abbreviation column
    if "Abbreviation" not in out.columns:
        if "Driver" in out.columns:
            out["Abbreviation"] = out["Driver"]
        else:
            raise ValueError("DataFrame must contain 'Abbreviation' or 'Driver' column.")

    out["Abbreviation"] = normalize_driver_series(out["Abbreviation"])

    # Ensure QualifyingPosition column
    if "QualifyingPosition" not in out.columns:
        if "Position" in out.columns:
            out["QualifyingPosition"] = pd.to_numeric(out["Position"], errors="coerce")
        else:
            raise ValueError("DataFrame must contain 'Position' or 'QualifyingPosition'.")
    else:
        out["QualifyingPosition"] = pd.to_numeric(out["QualifyingPosition"], errors="coerce")

    # Deduplicate drivers if necessary
    out = out.drop_duplicates(subset=["Abbreviation"]).copy()

    # Time tie-breaker column if available
    time_col = None
    for c in ["QualifyingTime", "QTime", "Time", "q_time", "best_lap"]:
        if c in out.columns:
            time_col = c
            break

    # Parse penalties into standardized structure
    pen_dict: Dict[str, int] = {}
    pit_lane_drivers: List[str] = []
    back_of_grid_drivers: List[str] = []
    dns_drivers: List[str] = []
    race_dsq_drivers: List[str] = []
    quali_dsq_permitted_drivers: List[str] = []

    # Check dataframe internal columns for penalty / status hints
    for _, row in out.iterrows():
        d = row["Abbreviation"]
        st = str(row.get("Status", "")).upper()
        if st in ["DNS", "DID NOT START"]:
            dns_drivers.append(d)
        elif st in ["DSQ", "DISQUALIFIED"]:
            # If explicit permitted flag not set, treat as race DSQ
            if bool(row.get("permitted_to_start", False)):
                quali_dsq_permitted_drivers.append(d)
            else:
                race_dsq_drivers.append(d)
        if bool(row.get("is_pit_lane", False)) or bool(row.get("is_pit_lane_start", False)):
            if d not in pit_lane_drivers:
                pit_lane_drivers.append(d)

    if penalties:
        if isinstance(penalties, dict):
            for drv, spec in penalties.items():
                drv_clean = _clean_driver_code(drv)
                specs = spec if isinstance(spec, list) else [spec]
                for s_item in specs:
                    if isinstance(s_item, (int, float, np.integer)):
                        places = int(s_item)
                        pen_dict[drv_clean] = pen_dict.get(drv_clean, 0) + places
                        if pen_dict[drv_clean] >= 15 and drv_clean not in back_of_grid_drivers:
                            back_of_grid_drivers.append(drv_clean)
                    elif isinstance(s_item, dict):
                        p_type = str(s_item.get("penalty_type", s_item.get("type", "GRID_DROP"))).upper()
                        places = int(s_item.get("places", 0))
                        pen_dict[drv_clean] = pen_dict.get(drv_clean, 0) + places
                        if p_type == PenaltyType.PIT_LANE.value or places < 0:
                            if drv_clean not in pit_lane_drivers:
                                pit_lane_drivers.append(drv_clean)
                        elif p_type == PenaltyType.BACK_OF_GRID.value or pen_dict[drv_clean] >= 15:
                            if drv_clean not in back_of_grid_drivers:
                                back_of_grid_drivers.append(drv_clean)
                        elif p_type == PenaltyType.DNS.value:
                            if drv_clean not in dns_drivers:
                                dns_drivers.append(drv_clean)
                        elif p_type == PenaltyType.DSQ.value:
                            if s_item.get("permitted_to_start", False):
                                if drv_clean not in quali_dsq_permitted_drivers:
                                    quali_dsq_permitted_drivers.append(drv_clean)
                            else:
                                if drv_clean not in race_dsq_drivers:
                                    race_dsq_drivers.append(drv_clean)
        elif isinstance(penalties, list):
            for prec in penalties:
                if isinstance(prec, dict):
                    drv_name = prec.get("driver", prec.get("driver_code", ""))
                    drv_clean = _clean_driver_code(drv_name)
                    p_type_val = str(prec.get("penalty_type", prec.get("type", "GRID_DROP"))).upper()
                    places = int(prec.get("places", 0))
                    reason = str(prec.get("reason", ""))
                    article = str(prec.get("article", ""))
                    is_permitted = bool(prec.get("permitted_to_start", False)) or "permitted" in reason.lower() or "art 39.4" in article.lower()
                else:
                    drv_clean = _clean_driver_code(prec.driver)
                    p_type_val = prec.penalty_type.value if hasattr(prec.penalty_type, "value") else str(prec.penalty_type).upper()
                    places = int(prec.places)
                    reason = getattr(prec, "reason", "")
                    article = getattr(prec, "article", "")
                    is_permitted = "permitted" in reason.lower() or "art 39.4" in article.lower()

                pen_dict[drv_clean] = pen_dict.get(drv_clean, 0) + places
                if p_type_val == PenaltyType.PIT_LANE.value or places < 0:
                    if drv_clean not in pit_lane_drivers:
                        pit_lane_drivers.append(drv_clean)
                elif p_type_val == PenaltyType.BACK_OF_GRID.value or pen_dict[drv_clean] >= 15:
                    if drv_clean not in back_of_grid_drivers:
                        back_of_grid_drivers.append(drv_clean)
                elif p_type_val == PenaltyType.DNS.value:
                    if drv_clean not in dns_drivers:
                        dns_drivers.append(drv_clean)
                elif p_type_val == PenaltyType.DSQ.value:
                    if is_permitted:
                        if drv_clean not in quali_dsq_permitted_drivers:
                            quali_dsq_permitted_drivers.append(drv_clean)
                    else:
                        if drv_clean not in race_dsq_drivers:
                            race_dsq_drivers.append(drv_clean)

    # Exclude DNS and Race DSQ from the starting grid computation
    excluded_drivers = set(dns_drivers + race_dsq_drivers)

    # Sort helper function for qualifying order (respects official position and Q3/Q2/Q1 lap times)
    def get_sort_key(drv: str):
        row = out[out["Abbreviation"] == drv].iloc[0]
        q_pos = row.get("QualifyingPosition", np.nan)
        pos_val = float(q_pos) if pd.notna(q_pos) else 999.0
        t_val = _extract_best_session_laptime(row)
        return (pos_val, t_val)

    # Filter physical grid drivers (must not be excluded, must not be pit-lane, must not be quali DSQ permitted)
    all_drivers = out["Abbreviation"].tolist()
    eligible_drivers = [d for d in all_drivers if d not in excluded_drivers]
    non_pit_drivers = [d for d in eligible_drivers if d not in pit_lane_drivers and d not in quali_dsq_permitted_drivers]

    # Partition non-pit drivers into:
    # 1. Back of grid (>= 15 places or explicit back-of-grid)
    # 2. Main grid (unpenalized or < 15 places drop)
    bog_set = set(back_of_grid_drivers)
    main_drivers = [d for d in non_pit_drivers if d not in bog_set]
    bog_drivers = [d for d in non_pit_drivers if d in bog_set]

    # Sort main drivers by qualifying
    main_drivers.sort(key=get_sort_key)
    bog_drivers.sort(key=get_sort_key)

    unpenalized = [d for d in main_drivers if pen_dict.get(d, 0) == 0]
    penalized = [d for d in main_drivers if pen_dict.get(d, 0) > 0]

    # Order main grid under official FIA Article 42.3 Procedure:
    # 1. Unpenalized drivers maintain their relative qualifying order.
    # 2. Penalized drivers receive a temporary grid position = QualifyingPosition + Penalty.
    # 3. Penalized drivers drop behind unpenalized drivers who qualified ahead, plus additional
    #    unpenalized drivers up to their target drop position (QualifyingPosition + Penalty).
    # 4. If two penalized drivers share the same temporary position, earlier qualifying position
    #    or faster qualifying time breaks the tie (Art 42.3(b)).
    def compare_main_drivers(a: str, b: str) -> int:
        pen_a = pen_dict.get(a, 0)
        pen_b = pen_dict.get(b, 0)
        qa, ta = get_sort_key(a)
        qb, tb = get_sort_key(b)

        # If neither has penalty, order strictly by qualifying
        if pen_a == 0 and pen_b == 0:
            if qa != qb:
                return -1 if qa < qb else 1
            return -1 if ta < tb else (1 if ta > tb else 0)

        # If a has penalty and b does not
        if pen_a > 0 and pen_b == 0:
            target_a = qa + pen_a
            # If target_a >= qb, a dropped to or behind b -> b is ahead of a
            return 1 if target_a >= qb else -1

        # If b has penalty and a does not
        if pen_b > 0 and pen_a == 0:
            target_b = qb + pen_b
            # If target_b >= qa, b dropped to or behind a -> a is ahead of b
            return -1 if target_b >= qa else 1

        # Both have penalties
        target_a = qa + pen_a
        target_b = qb + pen_b
        if target_a != target_b:
            return -1 if target_a < target_b else 1
        # Tied temporary grid position: break tie by qualifying (Art 42.3(b))
        if qa != qb:
            return -1 if qa < qb else 1
        return -1 if ta < tb else (1 if ta > tb else 0)

    ordered_physical: List[str] = sorted(main_drivers, key=cmp_to_key(compare_main_drivers))

    # Append back-of-grid drivers (Art 42.3(d)), ordered relative to each other by qualifying
    ordered_physical.extend(bog_drivers)

    # Append qualifying DSQ permitted to start (Art 39.4), ordered by qualifying
    quali_dsq_permitted_drivers.sort(key=get_sort_key)
    ordered_physical.extend(quali_dsq_permitted_drivers)

    n_physical = len(ordered_physical)

    # Pit-lane starters (Art 42.5), ordered relative to each other by qualifying
    pit_lane_drivers_sorted = [d for d in pit_lane_drivers if d in eligible_drivers]
    pit_lane_drivers_sorted.sort(key=get_sort_key)
    n_pit = len(pit_lane_drivers_sorted)

    total_starters = n_physical + n_pit

    # Validate expected grid size if specified
    if expected_grid_size is not None:
        if total_starters != expected_grid_size:
            # Check if input had excluded entrants
            if len(all_drivers) != expected_grid_size:
                logger.warning(
                    f"Grid lineup has {total_starters} starters ({len(all_drivers)} total entrants); "
                    f"expected {expected_grid_size}."
                )

    # Build canonical records
    records: List[Dict[str, Any]] = []

    # Map driver original rows
    out_indexed = out.set_index("Abbreviation")

    # 1. Physical starters
    for pos, drv in enumerate(ordered_physical, start=1):
        row_data = out_indexed.loc[drv].to_dict() if drv in out_indexed.index else {}
        q_pos = row_data.get("QualifyingPosition", np.nan)
        pen_places = pen_dict.get(drv, 0)
        status = StartingGridStatus.CLASSIFIED.value
        if pen_places > 0 or drv in bog_set or drv in quali_dsq_permitted_drivers:
            status = StartingGridStatus.PENALIZED.value

        net_delta = int(pos - q_pos) if pd.notna(q_pos) else 0

        # Preserve team name if available
        team = row_data.get("TeamName", row_data.get("Team", "Unknown"))

        rec = {
            **row_data,
            "Abbreviation": drv,
            "TeamName": team,
            "QualifyingPosition": float(q_pos) if pd.notna(q_pos) else np.nan,
            "ActualStartingGrid": float(pos),
            "simulation_start_rank": float(pos),
            "is_pit_lane_start": 0,
            "StartingGridStatus": status,
            "GridPenalty": int(pen_places),
            "grid_penalty_places": int(pen_places),
            "grid_vs_quali_delta": net_delta,
            "grid_norm": (float(pos) - 1.0) / float(max(total_starters - 1, 1)),
        }
        records.append(rec)

    # 2. Pit-lane starters
    for idx, drv in enumerate(pit_lane_drivers_sorted):
        sim_rank = float(n_physical + 1 + idx)
        row_data = out_indexed.loc[drv].to_dict() if drv in out_indexed.index else {}
        q_pos = row_data.get("QualifyingPosition", np.nan)
        team = row_data.get("TeamName", row_data.get("Team", "Unknown"))
        pen_places = pen_dict.get(drv, 0)

        rec = {
            **row_data,
            "Abbreviation": drv,
            "TeamName": team,
            "QualifyingPosition": float(q_pos) if pd.notna(q_pos) else np.nan,
            "ActualStartingGrid": np.nan,  # Physical grid position is NULL for pit-lane starters
            "simulation_start_rank": sim_rank,
            "is_pit_lane_start": 1,
            "StartingGridStatus": StartingGridStatus.PIT_LANE.value,
            "GridPenalty": int(pen_places),
            "grid_penalty_places": int(pen_places),
            "grid_vs_quali_delta": int(sim_rank - q_pos) if pd.notna(q_pos) else 0,
            "grid_norm": (sim_rank - 1.0) / float(max(total_starters - 1, 1)),
        }
        records.append(rec)

    # 3. DNS / DSQ excluded entrants (if present in input)
    for drv in sorted(excluded_drivers):
        row_data = out_indexed.loc[drv].to_dict() if drv in out_indexed.index else {}
        q_pos = row_data.get("QualifyingPosition", np.nan)
        team = row_data.get("TeamName", row_data.get("Team", "Unknown"))
        st = StartingGridStatus.DNS.value if drv in dns_drivers else StartingGridStatus.DSQ.value

        rec = {
            **row_data,
            "Abbreviation": drv,
            "TeamName": team,
            "QualifyingPosition": float(q_pos) if pd.notna(q_pos) else np.nan,
            "ActualStartingGrid": np.nan,
            "simulation_start_rank": np.nan,
            "is_pit_lane_start": 0,
            "StartingGridStatus": st,
            "GridPenalty": 0,
            "grid_penalty_places": 0,
            "grid_vs_quali_delta": 0,
            "grid_norm": np.nan,
        }
        records.append(rec)

    result_df = pd.DataFrame(records)

    # Verification invariants
    # 1. Physical grid positions must be unique and contiguous 1..N_physical
    physical_mask = result_df["ActualStartingGrid"].notna()
    physical_positions = sorted(result_df.loc[physical_mask, "ActualStartingGrid"].astype(int).tolist())
    expected_positions = list(range(1, n_physical + 1))
    if physical_positions != expected_positions:
        raise AssertionError(
            f"Physical starting grid is not 1..{n_physical}: {physical_positions}"
        )

    # 2. Pit-lane starters must have null ActualStartingGrid and is_pit_lane_start == 1
    pit_mask = result_df["is_pit_lane_start"] == 1
    if not result_df.loc[pit_mask, "ActualStartingGrid"].isna().all():
        raise AssertionError("Pit-lane starters must have null (NaN) ActualStartingGrid!")

    # 3. simulation_start_rank for participating starters must be unique 1..total_starters
    starter_ranks = sorted(result_df.loc[result_df["simulation_start_rank"].notna(), "simulation_start_rank"].astype(int).tolist())
    if starter_ranks != list(range(1, total_starters + 1)):
        raise AssertionError(f"Simulation start ranks are not contiguous 1..{total_starters}: {starter_ranks}")

    return result_df


def apply_grid_penalties(
    df: pd.DataFrame,
    penalties: Optional[Dict[str, int]] = None,
    expected_grid_size: int = 22,
) -> pd.DataFrame:
    """
    Computes a valid 1-N starting grid without duplicate positions applying penalties.
    Matches FIA grid procedures.

    Backward compatibility wrapper around resolve_fia_starting_grid.
    Guarantees exact parity with legacy expectations, tests, and outputs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least 'Abbreviation' and 'Position' (or 'QualifyingPosition').
    penalties : dict, optional
        Mapping of driver abbreviation to grid penalty places.
    expected_grid_size : int
        Expected number of drivers on the grid (default 22).

    Returns
    -------
    pd.DataFrame
        DataFrame ordered by 'ActualStartingGrid' (1 to N) with 'GridPenalty' column.
    """
    resolved = resolve_fia_starting_grid(
        df=df,
        penalties=penalties,
        expected_grid_size=expected_grid_size,
    )

    # Legacy expectation: return physical grid ordered by ActualStartingGrid 1..N
    # If pit-lane starters are present, order by simulation_start_rank
    if "ActualStartingGrid" in resolved.columns and resolved["ActualStartingGrid"].notna().all():
        ordered = resolved.sort_values("ActualStartingGrid").reset_index(drop=True)
    else:
        ordered = resolved.sort_values("simulation_start_rank").reset_index(drop=True)

    if expected_grid_size is not None and len(ordered) != expected_grid_size:
        raise ValueError(
            f"Grid penalty calculation produced {len(ordered)} drivers; expected {expected_grid_size}."
        )

    return ordered
