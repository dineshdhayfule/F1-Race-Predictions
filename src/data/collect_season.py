"""
Authoritative FastF1 Full Season Data Collector.
================================================
Discovers all championship rounds in a season, automatically detects Sprint vs.
Standard Grand Prix formats, and collects only the required sessions for race prediction:

Standard Weekend:
- Qualifying
- Race

Sprint Weekend:
- Qualifying
- Race
- Sprint Qualifying / Sprint Shootout
- Sprint Race

Practice sessions (FP1, FP2, FP3) are strictly excluded.
Supports idempotency, resumption, incremental updates, and fault-tolerant collection.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import fastf1
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple, Union

from src.utils.driver_mapping import filter_competing_drivers
from src.utils.logging_utils import get_logger

logger = get_logger("collect_season")


def detect_event_format_and_sessions(event_row: pd.Series) -> Dict[str, Any]:
    """
    Analyzes event schedule metadata to determine if the event is a Sprint or Standard weekend,
    and dynamically resolves the session candidate names without hardcoding rounds.

    Supports historical and evolving sprint naming conventions:
      - 'sprint_qualifying' / 'Sprint Qualifying'
      - 'sprint_shootout' / 'Sprint Shootout'
      - 'sprint' / 'Sprint Race'

    Explicitly excludes FP1, FP2, and FP3.
    """
    fmt_raw = str(event_row.get("EventFormat", "")).strip().lower()
    
    # Collect all declared session names from schedule columns (Session1 .. Session5)
    session_names = []
    for i in range(1, 6):
        col = f"Session{i}"
        if col in event_row and pd.notna(event_row[col]):
            val = str(event_row[col]).strip()
            if val:
                session_names.append(val)

    # Determine if sprint format
    is_sprint = (
        "sprint" in fmt_raw
        or any("sprint" in s.lower() for s in session_names)
    )

    # Discover specific sprint session names from schedule
    sq_declared = None
    sr_declared = None

    for s in session_names:
        s_lower = s.lower()
        if "shootout" in s_lower or "sprint qualifying" in s_lower or "sprint quali" in s_lower or s_lower == "sq":
            sq_declared = s
        elif "sprint" in s_lower and "shootout" not in s_lower and "qualifying" not in s_lower and "quali" not in s_lower:
            sr_declared = s

    # Qualifying declared name
    q_declared = next((s for s in session_names if "qualifying" in s.lower() and "sprint" not in s.lower()), "Qualifying")
    # Race declared name
    r_declared = next((s for s in session_names if s.lower() == "race" or ("race" in s.lower() and "sprint" not in s.lower())), "Race")

    session_plan = []
    if is_sprint:
        # 1. Sprint Qualifying / Shootout
        sq_candidates = [sq_declared] if sq_declared else []
        for default_candidate in ("Sprint Qualifying", "Sprint Shootout", "SQ", "Sprint Quali"):
            if default_candidate not in sq_candidates:
                sq_candidates.append(default_candidate)
        session_plan.append({
            "canonical": "Sprint Qualifying",
            "slug": "sprint_qualifying",
            "candidates": sq_candidates,
        })

        # 2. Sprint Race
        sr_candidates = [sr_declared] if sr_declared else []
        for default_candidate in ("Sprint", "Sprint Race", "SR"):
            if default_candidate not in sr_candidates:
                sr_candidates.append(default_candidate)
        session_plan.append({
            "canonical": "Sprint Race",
            "slug": "sprint_race",
            "candidates": sr_candidates,
        })

    # Standard / GP sessions for all weekends:
    # 3. GP Qualifying
    q_candidates = [q_declared] if q_declared else []
    for default_candidate in ("Qualifying", "Q", "Quali"):
        if default_candidate not in q_candidates:
            q_candidates.append(default_candidate)
    session_plan.append({
        "canonical": "Qualifying",
        "slug": "qualifying",
        "candidates": q_candidates,
    })

    # 4. GP Race
    r_candidates = [r_declared] if r_declared else []
    for default_candidate in ("Race", "R"):
        if default_candidate not in r_candidates:
            r_candidates.append(default_candidate)
    session_plan.append({
        "canonical": "Race",
        "slug": "race",
        "candidates": r_candidates,
    })

    return {
        "is_sprint": is_sprint,
        "format": "sprint" if is_sprint else "standard",
        "format_name": fmt_raw or ("sprint" if is_sprint else "conventional"),
        "session_plan": session_plan,
        "sq_declared": sq_declared,
        "sr_declared": sr_declared,
    }


def validate_and_format_session_results(
    session,
    expected_year: int,
    expected_round: int,
    event_name: str,
    circuit_name: str,
    country: str,
    canonical_type: str,
    original_session_name: str = ""
) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Validates FastF1 session results and enriches with standard metadata columns.

    Checks:
    - Session results are non-empty.
    - Valid position and driver identifiers exist.
    - Session round matches expected round if present on session event.
    - Resolves duplicates if present.

    Enriches with metadata:
    - Year, Round, Race, Circuit, Country, Session, SessionDate, DataSource, CollectedAt
    """
    res = session.results.copy() if hasattr(session, "results") and session.results is not None else pd.DataFrame()
    if res.empty:
        return False, "Session results table is empty.", None

    # Check position column
    pos_col = "Position" if "Position" in res.columns else ("ClassifiedPosition" if "ClassifiedPosition" in res.columns else None)
    if not pos_col:
        return False, "Missing Position or ClassifiedPosition column.", None

    valid_positions = res[pos_col].dropna()
    if len(valid_positions) == 0:
        return False, "Session results contain no classified or finishing positions.", None

    # Check driver identifier column
    driver_col = next((c for c in ["Abbreviation", "Driver", "BroadcastName", "DriverNumber"] if c in res.columns), None)
    if not driver_col:
        return False, "No valid driver identifier column found in session results.", None

    # Check session round identity if available on session.event
    if hasattr(session, "event") and session.event is not None:
        if hasattr(session.event, "RoundNumber") and pd.notna(session.event.RoundNumber):
            try:
                sess_r = int(session.event.RoundNumber)
                if sess_r > 0 and sess_r != expected_round:
                    return False, f"Round identity mismatch: session event is round {sess_r}, expected {expected_round}.", None
            except (ValueError, TypeError):
                pass

    # Exclude non-driver FIA course/service vehicles
    res = filter_competing_drivers(res)

    # Deduplicate drivers if duplicate records exist
    if "Abbreviation" in res.columns:
        if res["Abbreviation"].dropna().duplicated().any():
            dups = res[res["Abbreviation"].duplicated()]["Abbreviation"].tolist()
            logger.warning(f"Deduplicating multiple rows for drivers in {canonical_type}: {dups}")
            res = res.drop_duplicates(subset=["Abbreviation"]).copy()

    # Session date resolution
    session_date = None
    if hasattr(session, "date") and session.date is not None:
        session_date = str(session.date)
    elif hasattr(session, "event") and session.event is not None and hasattr(session.event, "EventDate"):
        session_date = str(session.event.EventDate)

    # Attach canonical metadata columns
    res["Year"] = expected_year
    res["Round"] = expected_round
    res["Race"] = event_name
    res["Circuit"] = circuit_name
    res["Country"] = country
    res["Session"] = canonical_type
    res["OriginalSessionName"] = original_session_name or canonical_type
    res["SessionDate"] = session_date or ""
    res["DataSource"] = "FastF1"
    res["CollectedAt"] = datetime.now(timezone.utc).isoformat()

    return True, "VALID", res


def collect_round_data(
    year: int,
    round_num: int,
    event_row: pd.Series,
    output_dir: Path,
    track_dir: Optional[Path] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Collects required session data for a single Grand Prix round.
    Strictly excludes FP1, FP2, and FP3.
    """
    event_name = str(event_row.get("EventName", f"Round {round_num}"))
    circuit_name = str(event_row.get("Location", event_row.get("Circuit", "")))
    country = str(event_row.get("Country", ""))

    format_info = detect_event_format_and_sessions(event_row)
    is_sprint = format_info["is_sprint"]
    session_plan = format_info["session_plan"]

    round_successes = []
    round_skips = []
    round_failures = []
    round_unavailable = []

    for item in session_plan:
        c_type = item["canonical"]
        slug = item["slug"]
        candidates = item["candidates"]

        target_file = output_dir / f"{year}_round_{round_num}_{slug}.csv"
        track_target_file = (track_dir / f"{slug}.csv") if track_dir else None

        # Check if already present and valid (Incremental / Resumable)
        if target_file.exists() and not force:
            try:
                if target_file.stat().st_size > 0:
                    existing_df = pd.read_csv(target_file)
                    if not existing_df.empty:
                        skip_msg = f"{c_type} (file exists: {target_file.name}, {len(existing_df)} drivers)"
                        round_skips.append(skip_msg)
                        logger.info(f"    [SKIP] {skip_msg}")
                        # Also sync track dir if requested
                        if track_target_file and not track_target_file.exists():
                            track_target_file.parent.mkdir(parents=True, exist_ok=True)
                            existing_df.to_csv(track_target_file, index=False)
                        continue
            except Exception as e:
                logger.warning(f"Existing file {target_file.name} corrupted ({e}). Re-collecting.")

        # Attempt download across candidates
        collected = False
        last_error = None

        for cand in candidates:
            try:
                sess = fastf1.get_session(year, round_num, cand)
                sess.load(telemetry=False, weather=False, messages=False)
                valid, msg, df = validate_and_format_session_results(
                    session=sess,
                    expected_year=year,
                    expected_round=round_num,
                    event_name=event_name,
                    circuit_name=circuit_name,
                    country=country,
                    canonical_type=c_type,
                    original_session_name=cand,
                )
                if valid and df is not None and not df.empty:
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(target_file, index=False)

                    if track_target_file:
                        track_target_file.parent.mkdir(parents=True, exist_ok=True)
                        df.to_csv(track_target_file, index=False)

                    success_msg = f"{c_type} ({len(df)} drivers via '{cand}')"
                    round_successes.append(success_msg)
                    logger.info(f"    [OK] Saved {success_msg}")
                    collected = True
                    break
                else:
                    last_error = msg
            except Exception as e:
                last_error = str(e)

        if not collected:
            err_summary = f"{c_type}: {last_error or 'Session results not available'}"
            if "not available" in err_summary.lower() or "empty" in err_summary.lower():
                round_unavailable.append(err_summary)
                logger.info(f"    [INFO] {err_summary}")
            else:
                round_failures.append(err_summary)
                logger.warning(f"    [FAIL] {err_summary}")

    return {
        "round": round_num,
        "event_name": event_name,
        "circuit_name": circuit_name,
        "country": country,
        "is_sprint": is_sprint,
        "successes": round_successes,
        "skips": round_skips,
        "failures": round_failures,
        "unavailable": round_unavailable,
    }


def collect_season_data(
    year: int,
    rounds: Optional[List[int]] = None,
    output_dir: Optional[Path] = None,
    force: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Collects full season race and qualifying sessions for any championship year.
    Discovers all rounds, detects Sprint vs. Standard format, and collects:
      - Standard Grand Prix: Qualifying + Race
      - Sprint Grand Prix:   Qualifying + Race + Sprint Qualifying + Sprint Race

    Practice sessions (FP1, FP2, FP3) are strictly excluded.

    Parameters
    ----------
    year : int
        Championship year (e.g. 2026, 2024).
    rounds : list of int, optional
        Explicit round numbers to collect. If None, dynamically discovers all positive rounds.
    output_dir : Path, optional
        Target directory. Defaults to data/raw/season/{year}/.
    force : bool, default False
        If True, re-downloads and overwrites existing valid CSV files.
    verbose : bool, default True
        If True, prints formatted terminal output.

    Returns
    -------
    dict with full summary statistics:
      - 'year', 'rounds_discovered', 'standard_count', 'sprint_count',
      - 'successes', 'skips', 'failures', 'unavailable', 'round_summaries'
    """
    root = Path(__file__).resolve().parents[2]
    out_dir = output_dir or (root / "data" / "raw" / "season" / str(year))
    cache_dir = root / "data" / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    fastf1.Cache.enable_cache(str(cache_dir))

    if verbose:
        print("\n" + "=" * 80)
        print(f"FASTF1 FULL SEASON DATA COLLECTION: {year}")
        print("=" * 80)
        print("Required sessions:")
        print("  - Standard GP: Qualifying + Race")
        print("  - Sprint GP:   Qualifying + Race + Sprint Qualifying + Sprint Race")
        print("Practice sessions (FP1, FP2, FP3) are strictly excluded.")
        print("=" * 80 + "\n")

    logger.info(f"Discovering schedule for championship season {year}...")
    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as e:
        err_msg = f"Failed to retrieve season schedule for {year}: {e}"
        logger.error(err_msg)
        if verbose:
            print(f"[ERROR] {err_msg}\n")
        return {
            "year": year,
            "rounds_discovered": [],
            "standard_count": 0,
            "sprint_count": 0,
            "successes": [],
            "skips": [],
            "failures": [err_msg],
            "unavailable": [],
            "round_summaries": [],
        }

    # Discover valid championship rounds
    if rounds is None:
        if not schedule.empty and "RoundNumber" in schedule.columns:
            valid_rounds = pd.to_numeric(schedule["RoundNumber"], errors="coerce").dropna()
            rounds = sorted([int(r) for r in valid_rounds.unique() if int(r) > 0])
        else:
            rounds = []

    if not rounds:
        warn_msg = f"Could not discover any positive championship rounds for {year}."
        logger.warning(warn_msg)
        if verbose:
            print(f"[WARNING] {warn_msg}\n")
        return {
            "year": year,
            "rounds_discovered": [],
            "standard_count": 0,
            "sprint_count": 0,
            "successes": [],
            "skips": [],
            "failures": [warn_msg],
            "unavailable": [],
            "round_summaries": [],
        }

    if verbose:
        print(f"Discovered {len(rounds)} championship rounds for season {year}.\n")

    total_successes = []
    total_skips = []
    total_failures = []
    total_unavailable = []
    round_summaries = []
    standard_count = 0
    sprint_count = 0

    for r in rounds:
        match = schedule[schedule["RoundNumber"] == r]
        if match.empty:
            msg = f"Round {r} not found in {year} schedule."
            logger.info(msg)
            total_unavailable.append(msg)
            continue

        event_row = match.iloc[0]
        event_name = str(event_row.get("EventName", f"Round {r}"))
        circuit_loc = str(event_row.get("Location", event_row.get("Country", "")))

        format_info = detect_event_format_and_sessions(event_row)
        is_sprint = format_info["is_sprint"]

        if is_sprint:
            sprint_count += 1
            fmt_label = "[SPRINT WEEKEND]"
        else:
            standard_count += 1
            fmt_label = "[STANDARD]"

        if verbose:
            loc_str = f" ({circuit_loc})" if circuit_loc else ""
            print(f"Round {r:2d}: {event_name}{loc_str} {fmt_label}")

        r_summary = collect_round_data(
            year=year,
            round_num=r,
            event_row=event_row,
            output_dir=out_dir,
            force=force,
        )

        round_summaries.append(r_summary)
        for s in r_summary["successes"]:
            total_successes.append(f"R{r} {s}")
            if verbose:
                print(f"  -> {s}")
        for s in r_summary["skips"]:
            total_skips.append(f"R{r} {s}")
            if verbose:
                print(f"  -> [SKIPPED] {s}")
        for u in r_summary["unavailable"]:
            total_unavailable.append(f"R{r} {u}")
            if verbose:
                print(f"  -> [UNAVAILABLE] {u}")
        for f in r_summary["failures"]:
            total_failures.append(f"R{r} {f}")
            if verbose:
                print(f"  -> [FAILED] {f}")
        if verbose:
            print()

    # Final summary report
    if verbose:
        print("=" * 80)
        print(f"SEASON {year} DATA COLLECTION SUMMARY")
        print("=" * 80)
        print(f"Total Rounds Discovered  : {len(rounds)}")
        print(f"Standard Grand Prix      : {standard_count}")
        print(f"Sprint Grand Prix        : {sprint_count}")
        print(f"Sessions Collected (New) : {len(total_successes)}")
        print(f"Sessions Skipped (Exist) : {len(total_skips)}")
        print(f"Sessions Unavailable    : {len(total_unavailable)}")
        print(f"Sessions Failed          : {len(total_failures)}")
        print(f"Storage Directory        : {out_dir}")
        print("=" * 80 + "\n")

    return {
        "year": year,
        "rounds_discovered": rounds,
        "standard_count": standard_count,
        "sprint_count": sprint_count,
        "successes": total_successes,
        "skips": total_skips,
        "failures": total_failures,
        "unavailable": total_unavailable,
        "round_summaries": round_summaries,
    }


def collect_grand_prix(
    track_id: str,
    year: int = 2026,
    force: bool = False,
    output_dir: Optional[Path] = None,
    season_output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Collects required session data for an individual Grand Prix circuit.
    Strictly collects Qualifying + Race (plus Sprint Qualifying + Sprint Race if Sprint).
    Explicitly excludes FP1, FP2, and FP3.
    """
    root = Path(__file__).resolve().parents[2]
    config_file = root / "config" / "tracks.json"
    with open(config_file, "r", encoding="utf-8") as f:
        tracks_config = json.load(f).get("tracks", {})

    from src.data.collect_track import get_event_by_track_id

    event_row = get_event_by_track_id(year, track_id, tracks_config)
    round_num = int(event_row["RoundNumber"])
    event_name = event_row["EventName"]
    circuit_loc = event_row.get("Location", "")

    s_dir = season_output_dir or (root / "data" / "raw" / "season" / str(year))
    t_dir = output_dir or (root / "data" / "raw" / "tracks" / track_id.lower().strip() / str(year))
    cache_dir = root / "data" / "cache"

    s_dir.mkdir(parents=True, exist_ok=True)
    t_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    fastf1.Cache.enable_cache(str(cache_dir))

    format_info = detect_event_format_and_sessions(event_row)
    is_sprint = format_info["is_sprint"]

    if verbose:
        print("\n" + "=" * 80)
        print(f"INDIVIDUAL GRAND PRIX DATA COLLECTION: {event_name.upper()} ({track_id.upper()}) {year}")
        print("=" * 80)
        print(f"Championship Round: Round {round_num}")
        print(f"Circuit Location  : {circuit_loc}")
        print(f"Weekend Format    : {'SPRINT WEEKEND' if is_sprint else 'STANDARD GRAND PRIX'}")
        print("Required Sessions : Qualifying, Race" + (", Sprint Qualifying, Sprint Race" if is_sprint else ""))
        print("Practice Sessions : FP1, FP2, FP3 are strictly excluded.")
        print("=" * 80 + "\n")

    r_summary = collect_round_data(
        year=year,
        round_num=round_num,
        event_row=event_row,
        output_dir=s_dir,
        track_dir=t_dir,
        force=force,
    )

    if verbose:
        for s in r_summary["successes"]:
            print(f"  [OK] {s}")
        for s in r_summary["skips"]:
            print(f"  [SKIP] {s}")
        for u in r_summary["unavailable"]:
            print(f"  [INFO] {u}")
        for f in r_summary["failures"]:
            print(f"  [FAIL] {f}")
        print("\n" + "=" * 80)
        print(f"COLLECTION SUMMARY: {track_id.upper()} {year}")
        print("=" * 80)
        print(f"Sessions Collected (New) : {len(r_summary['successes'])}")
        print(f"Sessions Skipped (Exist) : {len(r_summary['skips'])}")
        print(f"Sessions Unavailable    : {len(r_summary['unavailable'])}")
        print(f"Sessions Failed          : {len(r_summary['failures'])}")
        print(f"Track Data Directory     : {t_dir}")
        print(f"Season Data Directory    : {s_dir}")
        print("=" * 80 + "\n")

    return r_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect FastF1 Season Data (Qualifying & Race)")
    parser.add_argument("--year", type=int, default=2026, help="Season championship year")
    parser.add_argument("--rounds", type=int, nargs="+", default=None, help="Explicit round numbers to collect")
    parser.add_argument("--force", action="store_true", help="Force re-collection of existing session files")
    args = parser.parse_args()

    collect_season_data(args.year, args.rounds, force=args.force)
