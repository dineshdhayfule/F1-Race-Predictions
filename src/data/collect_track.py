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
from typing import Optional

from src.utils.logging_utils import get_logger

logger = get_logger("collect_track")


def get_event_by_track_id(year: int, track_id: str, tracks_config: dict):
    schedule = fastf1.get_event_schedule(year)
    info = tracks_config.get(track_id.lower().strip())
    if not info:
        raise ValueError(f"Unknown track_id '{track_id}'.")

    # Match by round if configured
    if "round_2026" in info and year == 2026:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Weekend Session Data for Track")
    parser.add_argument("--track", type=str, required=True)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    collect_track_data(args.track, args.year, force=args.force)
