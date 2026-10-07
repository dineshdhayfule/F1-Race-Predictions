"""
Reusable FastF1 Season Collector.
Collects completed season rounds (Race, Qualifying, FP1-3) for any year.
"""

from pathlib import Path
import argparse
import fastf1
import pandas as pd
from typing import Optional, List

from src.utils.logging_utils import get_logger

logger = get_logger("collect_season")


def collect_season_data(year: int, rounds: Optional[List[int]] = None, output_dir: Optional[Path] = None, force: bool = False):
    root = Path(__file__).resolve().parents[2]
    out_dir = output_dir or (root / "data" / "raw" / "season" / str(year))
    cache_dir = root / "data" / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    fastf1.Cache.enable_cache(str(cache_dir))

    schedule = fastf1.get_event_schedule(year)
    if rounds is None:
        rounds = list(range(1, 16))

    for r in rounds:
        match = schedule[schedule["RoundNumber"] == r]
        if match.empty:
            logger.info(f"Round {r} not found in {year} schedule.")
            continue

        event_name = match.iloc[0]["EventName"]
        logger.info(f"Collecting {year} Round {r}: {event_name}...")

        # 1. Race session
        race_file = out_dir / f"{year}_round_{r}_race.csv"
        if not race_file.exists() or force:
            try:
                r_sess = fastf1.get_session(year, r, "R")
                r_sess.load(telemetry=False, weather=False, messages=False)
                res = r_sess.results.copy()
                if not res.empty:
                    res["Year"] = year
                    res["Race"] = event_name
                    res["Session"] = "Race"
                    res.to_csv(race_file, index=False)
                    logger.info(f"  [OK] Saved Race: {len(res)} drivers.")
            except Exception as e:
                logger.info(f"  [INFO] Race not available for round {r}: {e}")

        # 2. Qualifying session
        q_file = out_dir / f"{year}_round_{r}_qualifying.csv"
        if not q_file.exists() or force:
            try:
                q_sess = fastf1.get_session(year, r, "Q")
                q_sess.load(telemetry=False, weather=False, messages=False)
                res = q_sess.results.copy()
                if not res.empty and "Position" in res.columns and res["Position"].dropna().shape[0] > 0:
                    res["Year"] = year
                    res["Race"] = event_name
                    res["Session"] = "Qualifying"
                    res.to_csv(q_file, index=False)
                    logger.info(f"  [OK] Saved Qualifying: {len(res)} drivers.")
            except Exception as e:
                logger.info(f"  [INFO] Qualifying not available for round {r}: {e}")

        # 3. Sprint Qualifying session (if held)
        sq_file = out_dir / f"{year}_round_{r}_sprint_qualifying.csv"
        if not sq_file.exists() or force:
            for sq_name in ("Sprint Qualifying", "Sprint Shootout", "SQ"):
                try:
                    sq_sess = fastf1.get_session(year, r, sq_name)
                    sq_sess.load(telemetry=False, weather=False, messages=True)
                    res = sq_sess.results.copy()
                    if not res.empty and "Position" in res.columns and res["Position"].dropna().shape[0] > 0:
                        res["Year"] = year
                        res["Race"] = event_name
                        res["Session"] = "Sprint Qualifying"
                        res.to_csv(sq_file, index=False)
                        logger.info(f"  [OK] Saved Sprint Qualifying: {len(res)} drivers.")
                        break
                except Exception:
                    pass

        # 4. Sprint Race session (if held)
        sr_file = out_dir / f"{year}_round_{r}_sprint_race.csv"
        if not sr_file.exists() or force:
            for sr_name in ("Sprint", "Sprint Race"):
                try:
                    sr_sess = fastf1.get_session(year, r, sr_name)
                    sr_sess.load(telemetry=False, weather=False, messages=False)
                    res = sr_sess.results.copy()
                    if not res.empty and "Position" in res.columns and res["Position"].dropna().shape[0] > 0:
                        res["Year"] = year
                        res["Race"] = event_name
                        res["Session"] = "Sprint Race"
                        res.to_csv(sr_file, index=False)
                        logger.info(f"  [OK] Saved Sprint Race: {len(res)} drivers.")
                        break
                except Exception:
                    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect FastF1 Season Data")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--rounds", type=int, nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    collect_season_data(args.year, args.rounds, force=args.force)
