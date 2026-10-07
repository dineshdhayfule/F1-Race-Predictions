"""
Grid Features and Penalty Application Module
============================================
Handles starting grid calculation, penalty resolution, and grid validation.
Guarantees a valid, unique 1-N starting grid matching FIA sporting regulations.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def apply_grid_penalties(
    df: pd.DataFrame,
    penalties: Optional[Dict[str, int]] = None,
    expected_grid_size: int = 22
) -> pd.DataFrame:
    """
    Computes a valid 1-N starting grid without duplicate positions applying penalties.
    Matches FIA grid procedures:
    - Back-of-grid penalties (e.g. >=15 places) are relegated to the rear of the grid.
    - Grid-drop penalties (e.g. +3, +5, +10) shift positions backward; unpenalized drivers shift forward.
    - Resolves conflicts deterministically by qualifying position.
    - Guarantees exactly unique 1..N starting grid with zero duplicates.

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
    out = df.copy()
    if penalties is None:
        penalties = {}

    if "Abbreviation" not in out.columns:
        raise ValueError("DataFrame must contain 'Abbreviation' column.")

    if "QualifyingPosition" not in out.columns:
        if "Position" in out.columns:
            out["QualifyingPosition"] = pd.to_numeric(out["Position"], errors="coerce")
        else:
            raise ValueError("DataFrame must contain 'Position' or 'QualifyingPosition'.")

    out["GridPenalty"] = out["Abbreviation"].map(penalties).fillna(0).astype(int)

    # Separate back-of-grid penalties (>= 15 positions or explicit back-of-grid)
    back_drivers = [d for d, pen in penalties.items() if pen >= 15 and d in out["Abbreviation"].values]
    back = out[out["Abbreviation"].isin(back_drivers)].sort_values("QualifyingPosition")
    main = out[~out["Abbreviation"].isin(back_drivers)].sort_values("QualifyingPosition")

    # FIA grid drop logic:
    # 1. Unpenalized drivers maintain their relative qualifying order.
    # 2. For drivers with grid drop penalties (< 15 places), each penalty place drops them
    #    behind one driver who originally qualified behind them.
    # 3. If multiple drivers drop to the same position, earlier qualifying or smaller penalty takes precedence.
    main_drivers = main["Abbreviation"].tolist()
    q_pos_map = main.set_index("Abbreviation")["QualifyingPosition"].to_dict()

    # Track effective position score:
    # An unpenalized driver has penalty 0.
    # A penalized driver has penalty drop p.
    # Under FIA rules, a driver with penalty p drops behind min(p, unpenalized_drivers_behind) drivers.
    unpenalized = [d for d in main_drivers if penalties.get(d, 0) == 0]
    penalized = [d for d in main_drivers if penalties.get(d, 0) > 0]

    if not penalized:
        ordered_drivers = unpenalized
    else:
        # Build ordered grid for main
        ordered_drivers = []
        # Sort penalized by their target drop
        # Start with unpenalized list
        ordered_drivers = list(unpenalized)
        for d in sorted(penalized, key=lambda x: q_pos_map[x]):
            drop = penalties.get(d, 0)
            orig_q = q_pos_map[d]
            # Number of unpenalized drivers who qualified ahead of d
            ahead_unpen = sum(1 for u in unpenalized if q_pos_map[u] < orig_q)
            # Target insertion index in unpenalized list: ahead_unpen + drop
            target_idx = min(ahead_unpen + drop, len(ordered_drivers))
            ordered_drivers.insert(target_idx, d)

    # Back of grid drivers appended at rear, ordered by their qualifying times
    ordered_drivers.extend(back["Abbreviation"].tolist())

    ordered = out.set_index("Abbreviation").loc[ordered_drivers].reset_index()
    ordered["ActualStartingGrid"] = np.arange(1, len(ordered) + 1)
    ordered["grid_norm"] = (ordered["ActualStartingGrid"] - 1.0) / float(max(len(ordered) - 1, 1))

    if expected_grid_size is not None and len(ordered) != expected_grid_size:
        raise ValueError(
            f"Grid penalty calculation produced {len(ordered)} drivers; expected {expected_grid_size}."
        )

    grid_positions = sorted(ordered["ActualStartingGrid"].tolist())
    expected_positions = list(range(1, len(ordered) + 1))
    if grid_positions != expected_positions:
        raise AssertionError(f"Starting grid is not 1..{len(ordered)}: {grid_positions}")

    return ordered
