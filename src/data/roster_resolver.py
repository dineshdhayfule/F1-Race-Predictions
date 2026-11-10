"""
Dynamic Event Roster Resolver.
Builds authoritative, stage-aware driver lineups for any Grand Prix event across multiple seasons.
Strictly prevents missing entrants caused by previous-race DNS, driver illness, reserve substitutions,
or multi-year grid-size differences.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.utils.driver_mapping import (
    DRIVER_CATALOG,
    filter_competing_drivers,
    is_course_vehicle,
    normalize_driver_code,
)

logger = logging.getLogger("apexgrid.roster_resolver")


class RosterResolver:
    """
    Resolves the driver entry lineup for a target Grand Prix weekend and stage
    using hierarchical evidence:
      1. Official Event Entry List / Manifest (FIA / Event document)
      2. Starting Grid / Qualifying Entries (when stage >= after_qualifying)
      3. Practice Session Participation (FP1, FP2, FP3, SQ, SR)
      4. Season Driver Pool (ensures 2 cars per active team; repairs previous-race DNS/absences)
      5. Prior Race Starters (last-resort fallback)
    """

    def __init__(self, root_dir: Optional[Path] = None):
        self.root_dir = Path(root_dir) if root_dir else Path(__file__).resolve().parents[2]

    @staticmethod
    def get_expected_grid_size(year: int, target_round: Optional[int] = None) -> int:
        """
        Determines the regulation grid size for a given championship season.
        - 2026+: 22 cars (11 teams, including Cadillac)
        - 2017-2025: 20 cars (10 teams)
        - 2010-2012: 24 cars (12 teams)
        - 2013-2016: 22 cars (11 teams)
        - Earlier eras: 20 cars default
        """
        if year >= 2026:
            return 22
        elif 2017 <= year <= 2025:
            return 20
        elif 2010 <= year <= 2012:
            return 24
        elif 2013 <= year <= 2016:
            return 22
        return 20

    def resolve_roster(
        self,
        year: int,
        round_num: int,
        track_id: Optional[str] = None,
        stage: str = "after_qualifying",
        weekend_sessions: Optional[Dict[str, Optional[pd.DataFrame]]] = None,
        quali_df: Optional[pd.DataFrame] = None,
        starting_grid_df: Optional[pd.DataFrame] = None,
        entry_list_df: Optional[pd.DataFrame] = None,
        season_races_df: Optional[pd.DataFrame] = None,
        season_qualis_df: Optional[pd.DataFrame] = None,
        expected_grid_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Resolves the target event driver roster.

        Returns a DataFrame containing:
          - Abbreviation (canonical 3-letter code)
          - FullName
          - TeamName
          - DriverNumber
          - EntryStatus ("active", "substitute", "withdrawn", "dns")
          - Source ("official_entry_list", "starting_grid", "qualifying", "practice_sessions", "season_pool", "prior_race_fallback")
          - Confidence ("verified", "high", "provisional", "fallback")
        """
        if expected_grid_size is None or expected_grid_size <= 0:
            expected_grid_size = self.get_expected_grid_size(year, round_num)

        stage_clean = str(stage).lower().strip()
        if stage_clean == "after_sprint":
            stage_clean = "after_sprint_race"

        # -------------------------------------------------------------
        # PRIORITY 1: Official Entry List / Manifest
        # -------------------------------------------------------------
        if entry_list_df is not None and not entry_list_df.empty:
            roster = self._extract_from_dataframe(
                entry_list_df,
                source_label="official_entry_list",
                confidence="verified",
                year=year,
            )
            if len(roster) >= max(expected_grid_size - 2, 10):
                logger.info(
                    f"Roster resolved from Official Entry List ({len(roster)} drivers, expected {expected_grid_size})."
                )
                return self._finalize_roster(roster, expected_grid_size, year)

        # -------------------------------------------------------------
        # PRIORITY 2: Authoritative Starting Grid or Qualifying Session
        # Only permissible when stage has observed qualifying
        # -------------------------------------------------------------
        if stage_clean in ["after_qualifying", "before_race", "post_race"]:
            if starting_grid_df is not None and not starting_grid_df.empty:
                roster = self._extract_from_dataframe(
                    starting_grid_df,
                    source_label="starting_grid",
                    confidence="verified",
                    year=year,
                )
                if len(roster) >= max(expected_grid_size - 2, 10):
                    logger.info(
                        f"Roster resolved from Starting Grid ({len(roster)} drivers, expected {expected_grid_size})."
                    )
                    return self._finalize_roster(roster, expected_grid_size, year)

            if quali_df is not None and not quali_df.empty:
                roster = self._extract_from_dataframe(
                    quali_df,
                    source_label="qualifying",
                    confidence="high",
                    year=year,
                )
                if len(roster) >= max(expected_grid_size - 2, 10):
                    logger.info(
                        f"Roster resolved from Qualifying Session ({len(roster)} drivers, expected {expected_grid_size})."
                    )
                    return self._finalize_roster(roster, expected_grid_size, year)

        # -------------------------------------------------------------
        # PRIORITY 3: Practice Session Participation
        # For stages like after_fp1, after_fp2, after_fp3, after_sprint_qualifying, after_sprint_race
        # -------------------------------------------------------------
        practice_roster = self._extract_from_practice_sessions(
            weekend_sessions=weekend_sessions,
            stage=stage_clean,
            year=year,
        )
        if practice_roster is not None and len(practice_roster) >= expected_grid_size:
            logger.info(
                f"Roster resolved from Weekend Practice Sessions ({len(practice_roster)} drivers, expected {expected_grid_size})."
            )
            return self._finalize_roster(practice_roster, expected_grid_size, year)

        # -------------------------------------------------------------
        # PRIORITY 4: Season Driver Pool (Repairs previous-race DNS/absences)
        # -------------------------------------------------------------
        season_roster = self._build_from_season_pool(
            year=year,
            round_num=round_num,
            season_races_df=season_races_df,
            season_qualis_df=season_qualis_df,
            practice_seed=practice_roster,
            expected_grid_size=expected_grid_size,
        )
        if season_roster is not None and not season_roster.empty:
            logger.info(
                f"Roster resolved from Season Driver Pool ({len(season_roster)} drivers, expected {expected_grid_size})."
            )
            return self._finalize_roster(season_roster, expected_grid_size, year)

        # -------------------------------------------------------------
        # PRIORITY 5: Prior Round Starters Fallback (Last Resort)
        # -------------------------------------------------------------
        fallback_roster = self._build_from_prior_race_fallback(
            year=year,
            round_num=round_num,
            season_races_df=season_races_df,
            expected_grid_size=expected_grid_size,
        )
        if fallback_roster is not None and not fallback_roster.empty:
            logger.warning(
                f"Roster resolved from Prior Race Fallback ({len(fallback_roster)} drivers). "
                f"Lineup may omit drivers who did not start the prior event."
            )
            return self._finalize_roster(fallback_roster, expected_grid_size, year)

        # Emergency catalog fallback if no session or season data exists (e.g. Round 1 pre_fp1 synthetic test)
        logger.warning(
            f"No session or season history found. Generating catalog baseline for year {year}."
        )
        catalog_roster = self._build_catalog_baseline(year, expected_grid_size)
        return self._finalize_roster(catalog_roster, expected_grid_size, year)

    def _extract_from_dataframe(
        self,
        df: pd.DataFrame,
        source_label: str,
        confidence: str,
        year: int,
    ) -> pd.DataFrame:
        """Standardizes a session or grid DataFrame into the canonical roster format."""
        if df is None or df.empty:
            return pd.DataFrame()

        clean_df = filter_competing_drivers(df.copy())
        if clean_df.empty:
            return pd.DataFrame()

        abbr_col = "Abbreviation" if "Abbreviation" in clean_df.columns else (
            "Driver" if "Driver" in clean_df.columns else "DriverId"
        )
        team_col = "TeamName" if "TeamName" in clean_df.columns else (
            "Team" if "Team" in clean_df.columns else "Constructor"
        )
        num_col = "DriverNumber" if "DriverNumber" in clean_df.columns else (
            "Number" if "Number" in clean_df.columns else None
        )
        name_col = "FullName" if "FullName" in clean_df.columns else (
            "BroadcastName" if "BroadcastName" in clean_df.columns else "Driver"
        )

        rows = []
        for _, r in clean_df.iterrows():
            raw_abbr = str(r.get(abbr_col, "")).strip().upper()
            abbr = normalize_driver_code(raw_abbr)
            if not abbr or is_course_vehicle(abbr):
                continue

            cat_info = DRIVER_CATALOG.get(abbr, {})
            team = str(r.get(team_col, "")).strip() if team_col and pd.notna(r.get(team_col)) else cat_info.get("default_team", "Unknown")
            if team in ["nan", "None", "", "null"]:
                team = cat_info.get("default_team", "Unknown")

            full_name = str(r.get(name_col, "")).strip() if name_col and pd.notna(r.get(name_col)) else cat_info.get("full_name", abbr)
            if full_name in ["nan", "None", "", "null"]:
                full_name = cat_info.get("full_name", abbr)

            num = None
            if num_col and pd.notna(r.get(num_col)):
                try:
                    num = int(r[num_col])
                except (ValueError, TypeError):
                    num = cat_info.get("number")
            else:
                num = cat_info.get("number")

            # Entry status from classification/status column if present
            raw_status = str(r.get("Status", "active")).strip().lower()
            if "dns" in raw_status or "did not start" in raw_status:
                entry_status = "dns"
            elif "withdr" in raw_status:
                entry_status = "withdrawn"
            elif "dsq" in raw_status or "disqual" in raw_status:
                entry_status = "dsq"
            else:
                entry_status = "active"

            rows.append({
                "Abbreviation": abbr,
                "FullName": full_name,
                "TeamName": team,
                "DriverNumber": num,
                "EntryStatus": entry_status,
                "Source": source_label,
                "Confidence": confidence,
            })

        roster_df = pd.DataFrame(rows)
        if not roster_df.empty:
            roster_df = roster_df.drop_duplicates(subset=["Abbreviation"]).reset_index(drop=True)
        return roster_df

    def _extract_from_practice_sessions(
        self,
        weekend_sessions: Optional[Dict[str, Optional[pd.DataFrame]]],
        stage: str,
        year: int,
    ) -> Optional[pd.DataFrame]:
        """Extracts confirmed active drivers who ran laps in weekend sessions up to stage cutoff."""
        if not weekend_sessions:
            return None

        # Determine sessions permissible at or before stage cutoff
        allowed = []
        if stage in ["after_fp1", "after_fp2", "after_fp3", "after_sprint_qualifying", "after_sprint_race", "after_qualifying", "post_race"]:
            allowed.append("fp1")
        if stage in ["after_fp2", "after_fp3", "after_qualifying", "post_race"]:
            allowed.append("fp2")
        if stage in ["after_fp3", "after_qualifying", "post_race"]:
            allowed.append("fp3")
        if stage in ["after_sprint_qualifying", "after_sprint_race", "after_qualifying", "post_race"]:
            allowed.append("sprint_qualifying")
        if stage in ["after_sprint_race", "after_qualifying", "post_race"]:
            allowed.append("sprint_race")

        combined_rows = []
        for s in allowed:
            sess_df = weekend_sessions.get(s)
            if sess_df is not None and not sess_df.empty:
                extracted = self._extract_from_dataframe(
                    sess_df,
                    source_label=f"practice_{s}",
                    confidence="high",
                    year=year,
                )
                if not extracted.empty:
                    combined_rows.append(extracted)

        if not combined_rows:
            return None

        all_prac = pd.concat(combined_rows, ignore_index=True)
        # Later practice sessions take precedence
        all_prac = all_prac.drop_duplicates(subset=["Abbreviation"], keep="last").reset_index(drop=True)
        return all_prac

    def _build_from_season_pool(
        self,
        year: int,
        round_num: int,
        season_races_df: Optional[pd.DataFrame],
        season_qualis_df: Optional[pd.DataFrame],
        practice_seed: Optional[pd.DataFrame],
        expected_grid_size: int,
    ) -> Optional[pd.DataFrame]:
        """
        Builds the driver lineup from the season driver pool, ensuring every active team
        maintains two cars even if one driver recorded a DNS, DNF, or was absent in the prior race.
        Integrates any confirmed drivers from earlier practice sessions.
        """
        entries_by_driver: Dict[str, Dict[str, Any]] = {}

        # 1. Seed with confirmed drivers from current weekend practice
        if practice_seed is not None and not practice_seed.empty:
            for _, r in practice_seed.iterrows():
                abbr = r["Abbreviation"]
                entries_by_driver[abbr] = r.to_dict()

        # 2. Inspect qualifying from prior rounds (qualifying includes DNS drivers who participated in quali)
        prior_qualis = pd.DataFrame()
        if season_qualis_df is not None and not season_qualis_df.empty:
            if "Round" in season_qualis_df.columns:
                prior_qualis = season_qualis_df[season_qualis_df["Round"] < round_num].copy()
            else:
                prior_qualis = season_qualis_df.copy()

        # 3. Inspect races from prior rounds
        prior_races = pd.DataFrame()
        if season_races_df is not None and not season_races_df.empty:
            if "Round" in season_races_df.columns:
                prior_races = season_races_df[season_races_df["Round"] < round_num].copy()
            else:
                prior_races = season_races_df.copy()

        if prior_qualis.empty and prior_races.empty:
            if entries_by_driver:
                return pd.DataFrame(list(entries_by_driver.values()))
            return None

        # Determine all teams active in the season
        all_prior = pd.concat([prior_qualis, prior_races], ignore_index=True)
        all_prior = filter_competing_drivers(all_prior)
        if "TeamName" not in all_prior.columns:
            if "Team" in all_prior.columns:
                all_prior["TeamName"] = all_prior["Team"]
            else:
                all_prior["TeamName"] = "Unknown"

        active_teams = [t for t in all_prior["TeamName"].dropna().unique() if t not in ["Unknown", "nan", ""]]

        # For each team, ensure exactly 2 drivers are resolved
        team_slots: Dict[str, List[str]] = {t: [] for t in active_teams}

        # Place practice seed drivers into their respective team slots
        for abbr, dinfo in entries_by_driver.items():
            team = dinfo.get("TeamName")
            if team in team_slots and abbr not in team_slots[team]:
                team_slots[team].append(abbr)

        # Get latest rounds in reverse chronological order
        rounds_seen = sorted(all_prior["Round"].dropna().unique().astype(int), reverse=True) if "Round" in all_prior.columns else [1]

        for team in active_teams:
            if len(team_slots[team]) >= 2:
                continue

            # Look across recent rounds for this team's drivers
            for r in rounds_seen:
                if len(team_slots[team]) >= 2:
                    break

                # Check qualifying first (preserves drivers who suffered race DNS like Hirakawa in Round 19)
                if not prior_qualis.empty and "Round" in prior_qualis.columns:
                    q_round = prior_qualis[(prior_qualis["Round"] == r) & (prior_qualis["TeamName"] == team)]
                    for _, qrow in q_round.iterrows():
                        abbr = normalize_driver_code(str(qrow.get("Abbreviation", qrow.get("Driver", ""))))
                        if abbr and not is_course_vehicle(abbr) and abbr not in team_slots[team]:
                            team_slots[team].append(abbr)
                            if abbr not in entries_by_driver:
                                cat_info = DRIVER_CATALOG.get(abbr, {})
                                entries_by_driver[abbr] = {
                                    "Abbreviation": abbr,
                                    "FullName": cat_info.get("full_name", abbr),
                                    "TeamName": team,
                                    "DriverNumber": cat_info.get("number"),
                                    "EntryStatus": "active",
                                    "Source": "season_pool",
                                    "Confidence": "provisional",
                                }
                            if len(team_slots[team]) >= 2:
                                break

                # Then check race
                if len(team_slots[team]) < 2 and not prior_races.empty and "Round" in prior_races.columns:
                    r_round = prior_races[(prior_races["Round"] == r) & (prior_races["TeamName"] == team)]
                    for _, rrow in r_round.iterrows():
                        abbr = normalize_driver_code(str(rrow.get("Abbreviation", rrow.get("Driver", ""))))
                        if abbr and not is_course_vehicle(abbr) and abbr not in team_slots[team]:
                            team_slots[team].append(abbr)
                            if abbr not in entries_by_driver:
                                cat_info = DRIVER_CATALOG.get(abbr, {})
                                entries_by_driver[abbr] = {
                                    "Abbreviation": abbr,
                                    "FullName": cat_info.get("full_name", abbr),
                                    "TeamName": team,
                                    "DriverNumber": cat_info.get("number"),
                                    "EntryStatus": "active",
                                    "Source": "season_pool",
                                    "Confidence": "provisional",
                                }
                            if len(team_slots[team]) >= 2:
                                break

            # If team still has only 1 driver (e.g., both recent drivers unavailable), check catalog defaults
            if len(team_slots[team]) < 2:
                for cat_abbr, cat_meta in DRIVER_CATALOG.items():
                    if cat_meta.get("default_team") == team and cat_abbr not in team_slots[team]:
                        team_slots[team].append(cat_abbr)
                        if cat_abbr not in entries_by_driver:
                            entries_by_driver[cat_abbr] = {
                                "Abbreviation": cat_abbr,
                                "FullName": cat_meta.get("full_name", cat_abbr),
                                "TeamName": team,
                                "DriverNumber": cat_meta.get("number"),
                                "EntryStatus": "active",
                                "Source": "season_pool",
                                "Confidence": "provisional",
                            }
                        if len(team_slots[team]) >= 2:
                            break

        roster_df = pd.DataFrame(list(entries_by_driver.values()))
        return roster_df

    def _build_from_prior_race_fallback(
        self,
        year: int,
        round_num: int,
        season_races_df: Optional[pd.DataFrame],
        expected_grid_size: int,
    ) -> Optional[pd.DataFrame]:
        """Last-resort fallback to the previous completed race."""
        if season_races_df is None or season_races_df.empty:
            return None

        clean_races = filter_competing_drivers(season_races_df.copy())
        if "Round" in clean_races.columns:
            prior_races = clean_races[clean_races["Round"] < round_num]
            if prior_races.empty:
                return None
            latest_r = prior_races["Round"].max()
            latest_race = prior_races[prior_races["Round"] == latest_r]
        else:
            latest_race = clean_races

        return self._extract_from_dataframe(
            latest_race,
            source_label="prior_race_fallback",
            confidence="fallback",
            year=year,
        )

    def _build_catalog_baseline(self, year: int, expected_grid_size: int) -> pd.DataFrame:
        """Constructs a deterministic catalog baseline for a season when no files exist."""
        rows = []
        count = 0
        for abbr, meta in DRIVER_CATALOG.items():
            team = meta.get("default_team", "Unknown")
            # Exclude historical inactive drivers
            if team in ["Alfa Romeo", "AlphaTauri", "Kick Sauber", "RB", "Renault", "Toro Rosso", "Sauber"]:
                if year >= 2025:
                    continue
            rows.append({
                "Abbreviation": abbr,
                "FullName": meta.get("full_name", abbr),
                "TeamName": team,
                "DriverNumber": meta.get("number"),
                "EntryStatus": "active",
                "Source": "catalog_baseline",
                "Confidence": "provisional",
            })
            count += 1
            if count >= expected_grid_size:
                break
        return pd.DataFrame(rows)

    def _finalize_roster(
        self,
        df: pd.DataFrame,
        expected_grid_size: int,
        year: int,
    ) -> pd.DataFrame:
        """Ensures uniqueness, canonical column ordering, and logs grid size verification."""
        if df.empty:
            return df

        # Deduplicate on Abbreviation
        final_df = df.drop_duplicates(subset=["Abbreviation"]).copy()

        # Check grid count
        actual_count = len(final_df)
        if actual_count != expected_grid_size:
            logger.warning(
                f"Resolved roster for year {year} contains {actual_count} entrants (expected {expected_grid_size})."
            )
        else:
            logger.info(
                f"Resolved complete roster for year {year} ({actual_count}/{expected_grid_size} entrants verified)."
            )

        cols = ["Abbreviation", "FullName", "TeamName", "DriverNumber", "EntryStatus", "Source", "Confidence"]
        for c in cols:
            if c not in final_df.columns:
                final_df[c] = None

        return final_df[cols].reset_index(drop=True)
