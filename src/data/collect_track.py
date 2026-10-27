"""
Reusable Track Weekend Data Collector.
Collects current weekend sessions (FP1, FP2, FP3, Qualifying) for any configured track.
Strictly excludes target race result during pre-race data collection.
"""

from pathlib import Path
import json
import argparse
import fastf1
import pandas as pd
from typing import Optional, List

from src.utils.logging_utils import get_logger

logger = get_logger("collect_track")


def format_year_range(years: Optional[List[int]]) -> str:
    """
    Format a list of years into readable ranges (e.g., '2022–2025' or '2018–2019, 2022–2025').
    """
    if not years:
        return "None configured"

    valid_years = []
    for y in years:
        try:
            valid_years.append(int(y))
        except (ValueError, TypeError):
            continue

    if not valid_years:
        return "None configured"

    sorted_years = sorted(set(valid_years))
    ranges = []
    start = sorted_years[0]
    prev = start

    for y in sorted_years[1:]:
        if y == prev + 1:
            prev = y
        else:
            if start == prev:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}–{prev}")
            start = y
            prev = y

    if start == prev:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}–{prev}")

    return ", ".join(ranges)


def get_event_by_track_id(year: int, track_id: str, tracks_config: dict):
    schedule = fastf1.get_event_schedule(year)
    info = tracks_config.get(track_id.lower().strip())
    if not info:
        raise ValueError(f"Unknown track_id '{track_id}'.")

    # Match by round if configured for this year or 2026 default
    if f"round_{year}" in info:
        r_num = info[f"round_{year}"]
        match = schedule[schedule["RoundNumber"] == r_num]
        if not match.empty:
            return match.iloc[0]
    elif "round_2026" in info and year == 2026:
        r_num = info["round_2026"]
        match = schedule[schedule["RoundNumber"] == r_num]
        if not match.empty:
            return match.iloc[0]

    # Match by event name patterns
    patterns = "|".join(info.get("event_name_patterns", [track_id]))
    match = schedule[schedule["EventName"].str.contains(patterns, case=False, na=False)]
    if not match.empty:
        return match.iloc[0]

    match = schedule[schedule["Location"].str.contains(patterns, case=False, na=False)]
    if not match.empty:
        return match.iloc[0]

    raise RuntimeError(f"Could not locate {track_id} in {year} schedule.")


def save_practice_laps(session, target_path: Path, force: bool = False):
    if target_path.exists() and not force:
        logger.info(f"  [SKIP] {target_path.name} already exists.")
        return pd.read_csv(target_path)

    try:
        session.load(telemetry=False, weather=False, messages=False)
        laps = session.laps
        if laps is None or len(laps) == 0:
            logger.info(f"  [INFO] No lap data found for {session.name}.")
            return None

        laps = laps.copy().reset_index(drop=True)
        if "LapTime" in laps.columns:
            laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
        else:
            return None

        # Filter keep columns
        keep = [
            "Driver", "DriverNumber", "LapTime", "LapTimeSeconds", "LapNumber",
            "Stint", "Compound", "TyreLife", "Team", "Deleted"
        ]
        available = [c for c in keep if c in laps.columns]
        practice_data = laps[available].copy()
        practice_data["Year"] = session.event.EventDate.year
        practice_data["Race"] = session.event.EventName
        practice_data["Session"] = session.name

        target_path.parent.mkdir(parents=True, exist_ok=True)
        practice_data.to_csv(target_path, index=False)
        logger.info(f"  [OK] Saved {target_path.name}: {practice_data['Driver'].nunique()} drivers, {len(practice_data)} laps.")
        return practice_data
    except Exception as e:
        logger.info(f"  [INFO] {session.name} not available: {e}")
        return None


def collect_track_data(track_id: str, year: int, force: bool = False):
    root = Path(__file__).resolve().parents[2]
    config_file = root / "config" / "tracks.json"
    with open(config_file, "r", encoding="utf-8") as f:
        tracks_config = json.load(f)["tracks"]

    event = get_event_by_track_id(year, track_id, tracks_config)
    round_num = int(event["RoundNumber"])
    event_name = event["EventName"]

    track_dir = root / "data" / "raw" / "tracks" / track_id.lower() / str(year)
    cache_dir = root / "data" / "cache"
    track_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    fastf1.Cache.enable_cache(str(cache_dir))
    logger.info(f"Processing {year} Round {round_num}: {event_name} ({track_id.upper()})")

    # 1. FP1, FP2, FP3
    for s_name in ["FP1", "FP2", "FP3"]:
        s_file = track_dir / f"{s_name.lower()}.csv"
        try:
            sess = fastf1.get_session(year, round_num, s_name)
            save_practice_laps(sess, s_file, force=force)
        except Exception as e:
            logger.info(f"  [INFO] {s_name} not available: {e}")

    # 2. Qualifying
    q_file = track_dir / "qualifying.csv"
    if not q_file.exists() or force:
        try:
            q_sess = fastf1.get_session(year, round_num, "Q")
            q_sess.load(telemetry=False, weather=False, messages=False)
            res = q_sess.results.copy()
            if not res.empty and "Position" in res.columns and res["Position"].dropna().shape[0] > 0:
                res["Year"] = year
                res["Race"] = event_name
                res["Session"] = "Qualifying"
                res.to_csv(q_file, index=False)
                logger.info(f"  [OK] Saved Qualifying: {len(res)} drivers.")
            else:
                logger.info("  [INFO] Qualifying results not yet populated.")
        except Exception as e:
            logger.info(f"  [INFO] Qualifying not available: {e}")



def collect_historical_track_data(
    track_id: str,
    years: Optional[List[int]] = None,
    output_dir: Optional[Path] = None,
    force: bool = False,
    tracks_config: Optional[dict] = None,
) -> dict:
    """
    Collects historical Race and Qualifying session data for a configured track
    using the year ranges specified in config/tracks.json (or passed explicitly).

    Saves individual session files:
      {track_id}_{year}_race.csv
      {track_id}_{year}_qualifying.csv
    and combined historical files:
      {track_id}_historical_races.csv
      {track_id}_historical_qualifying.csv
    compatible with DataLoader.load_historical_track_data.

    Safely skips existing files when force=False.
    Returns a dict with successes, skips, and failures.
    """
    root = Path(__file__).resolve().parents[2]
    if tracks_config is None:
        config_file = root / "config" / "tracks.json"
        with open(config_file, "r", encoding="utf-8") as f:
            tracks_config = json.load(f).get("tracks", {})

    t_key = track_id.lower().strip()
    info = tracks_config.get(t_key)
    display_name = info.get("display_name", t_key.title()) if info else t_key.title()

    successes = []
    skips = []
    failures = []

    if not info:
        msg = f"Unknown circuit key '{track_id}'."
        failures.append(msg)
        logger.error(msg)
        return {
            "track": t_key,
            "circuit_name": display_name,
            "successes": successes,
            "skips": skips,
            "failures": failures,
        }

    # Determine years to collect
    if years is None:
        hist_avail = info.get("historical_data_available", False)
        configured_years = info.get("historical_years", [])
        if not hist_avail or not configured_years:
            msg = f"No historical years configured for {display_name} (historical_data_available: {hist_avail})."
            skips.append(msg)
            logger.info(f"  [SKIP] {msg}")
            return {
                "track": t_key,
                "circuit_name": display_name,
                "successes": successes,
                "skips": skips,
                "failures": failures,
            }
        years = configured_years

    # Target directory setup
    hist_dir = output_dir or (root / "data" / "raw" / "tracks" / t_key / "history")
    cache_dir = hist_dir.parent / "cache" if output_dir else (root / "data" / "cache")
    hist_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    fastf1.Cache.enable_cache(str(cache_dir))
    logger.info(f"Collecting historical data for {display_name} ({t_key}) across years: {years}")

    newly_saved_race = False
    newly_saved_quali = False

    for y in sorted(years):
        # 1. Locate event in season schedule
        try:
            event = get_event_by_track_id(y, t_key, tracks_config)
            round_num = int(event["RoundNumber"])
            event_name = event["EventName"]
        except Exception as e:
            msg = f"{y}: Could not locate event in schedule ({e})"
            skips.append(msg)
            logger.info(f"  [SKIP] {msg}")
            continue

        # 2. Race session
        race_file = hist_dir / f"{t_key}_{y}_race.csv"
        if race_file.exists() and not force:
            skips.append(f"{y} Race: file already exists ({race_file.name})")
            logger.info(f"  [SKIP] {y} Race already exists ({race_file.name}).")
        else:
            try:
                r_sess = fastf1.get_session(y, round_num, "R")
                r_sess.load(telemetry=False, weather=False, messages=False)
                res = r_sess.results.copy()
                if not res.empty and "Position" in res.columns and res["Position"].dropna().shape[0] > 0:
                    res["Year"] = y
                    res["Race"] = event_name
                    res["Session"] = "Race"
                    res.to_csv(race_file, index=False)
                    successes.append(f"{y} Race ({len(res)} drivers)")
                    newly_saved_race = True
                    logger.info(f"  [OK] Saved {y} Race: {len(res)} drivers.")
                else:
                    msg = f"{y} Race: session results empty or no classified positions"
                    skips.append(msg)
                    logger.info(f"  [SKIP] {msg}")
            except Exception as e:
                msg = f"{y} Race: {e}"
                failures.append(msg)
                logger.warning(f"  [FAIL] {msg}")

        # 3. Qualifying session
        quali_file = hist_dir / f"{t_key}_{y}_qualifying.csv"
        if quali_file.exists() and not force:
            skips.append(f"{y} Qualifying: file already exists ({quali_file.name})")
            logger.info(f"  [SKIP] {y} Qualifying already exists ({quali_file.name}).")
        else:
            try:
                q_sess = fastf1.get_session(y, round_num, "Q")
                q_sess.load(telemetry=False, weather=False, messages=False)
                res = q_sess.results.copy()
                if not res.empty and "Position" in res.columns and res["Position"].dropna().shape[0] > 0:
                    res["Year"] = y
                    res["Race"] = event_name
                    res["Session"] = "Qualifying"
                    res.to_csv(quali_file, index=False)
                    successes.append(f"{y} Qualifying ({len(res)} drivers)")
                    newly_saved_quali = True
                    logger.info(f"  [OK] Saved {y} Qualifying: {len(res)} drivers.")
                else:
                    msg = f"{y} Qualifying: session results empty or no positions"
                    skips.append(msg)
                    logger.info(f"  [SKIP] {msg}")
            except Exception as e:
                msg = f"{y} Qualifying: {e}"
                failures.append(msg)
                logger.warning(f"  [FAIL] {msg}")

    # Build or update combined historical files
    combined_r_file = hist_dir / f"{t_key}_historical_races.csv"
    combined_q_file = hist_dir / f"{t_key}_historical_qualifying.csv"

    # Only regenerate combined files if new files were saved, if combined file is missing, or force=True
    if newly_saved_race or not combined_r_file.exists() or force:
        r_files = sorted(hist_dir.glob(f"{t_key}_*_race.csv"))
        r_files = [f for f in r_files if f.name != combined_r_file.name]
        dfs = []
        for rf in r_files:
            try:
                df = pd.read_csv(rf)
                if not df.empty:
                    dfs.append(df)
            except Exception:
                pass
        if dfs:
            combined_r = pd.concat(dfs, ignore_index=True)
            combined_r.to_csv(combined_r_file, index=False)
            logger.info(f"  [OK] Updated {combined_r_file.name}: {len(combined_r)} rows across {len(dfs)} seasons.")

    if newly_saved_quali or not combined_q_file.exists() or force:
        q_files = sorted(hist_dir.glob(f"{t_key}_*_qualifying.csv"))
        q_files = [f for f in q_files if f.name != combined_q_file.name]
        dfs = []
        for qf in q_files:
            try:
                df = pd.read_csv(qf)
                if not df.empty:
                    dfs.append(df)
            except Exception:
                pass
        if dfs:
            combined_q = pd.concat(dfs, ignore_index=True)
            combined_q.to_csv(combined_q_file, index=False)
            logger.info(f"  [OK] Updated {combined_q_file.name}: {len(combined_q)} rows across {len(dfs)} seasons.")

    return {
        "track": t_key,
        "circuit_name": display_name,
        "successes": successes,
        "skips": skips,
        "failures": failures,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Weekend Session Data for Track")
    parser.add_argument("--track", type=str, required=True)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--historical", action="store_true", help="Collect historical race & qualifying sessions")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.historical:
        collect_historical_track_data(args.track, force=args.force)
    else:
        collect_track_data(args.track, args.year, force=args.force)
