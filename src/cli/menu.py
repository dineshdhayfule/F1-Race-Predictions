"""
Interactive Terminal Menu for F1 Unified Race Prediction Engine.
Provides complete guided workflows for:
1. Predict a Race
2. Load / Collect Data
3. View Previous Predictions & Reports
4. Evaluate a Prediction
5. System Health & Data Readiness
6. Configuration & Supported Circuits
"""

from pathlib import Path
import sys
import json
import subprocess
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from src.cli.terminal_ui import (
    clear_screen,
    print_banner,
    print_header,
    print_success,
    print_warning,
    print_error,
    print_info,
    prompt_choice,
    prompt_int,
    prompt_str,
    prompt_confirm,
    print_table,
    paginate_text,
    Colors,
)
from src.data.data_loader import DataLoader
from src.data.data_validator import DataValidator
from src.features.build_features import FeatureEngineer
from src.pipeline.race_prediction_pipeline import RacePredictionPipeline

ROOT = Path(__file__).resolve().parents[2]


class InteractiveMenu:
    """Master controller for the interactive CLI prediction engine menu."""

    def __init__(self, root_dir: Optional[Path] = None):
        self.root_dir = root_dir or ROOT
        self.loader = DataLoader(root_dir=self.root_dir)
        self.validator = DataValidator()
        self.tracks_cfg = self._load_tracks_config()
        self.model_cfg = self._load_model_config()
        self.pipeline: Optional[RacePredictionPipeline] = None

    def _get_pipeline(self) -> RacePredictionPipeline:
        if self.pipeline is None:
            self.pipeline = RacePredictionPipeline(root_dir=self.root_dir)
        return self.pipeline

    def _load_tracks_config(self) -> Dict[str, Any]:
        cfg_path = self.root_dir / "config" / "tracks.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                return json.load(f).get("tracks", {})
        return {}

    def _load_model_config(self) -> Dict[str, Any]:
        cfg_path = self.root_dir / "config" / "model_config.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def run(self) -> None:
        """Main event loop."""
        while True:
            try:
                clear_screen()
                print_banner(season=2026, status_text="Ready")
                print(f"{Colors.BOLD}MAIN MENU{Colors.RESET}\n")
                print("  1. Predict a Race")
                print("  2. Load / Collect Data")
                print("  3. View Previous Predictions & Reports")
                print("  4. Evaluate a Prediction")
                print("  5. System Health & Data Readiness")
                print("  6. Configuration & Supported Circuits")
                print("\n  0. Exit\n")

                choice = prompt_choice("Enter your choice", ["1", "2", "3", "4", "5", "6", "0"], default="1")

                if choice == "0":
                    print("\nExiting F1 Race Prediction Engine. Goodbye!\n")
                    sys.exit(0)
                elif choice == "1":
                    self.menu_predict_race()
                elif choice == "2":
                    self.menu_collect_data()
                elif choice == "3":
                    self.menu_view_outputs()
                elif choice == "4":
                    self.menu_evaluate_prediction()
                elif choice == "5":
                    self.menu_system_health()
                elif choice == "6":
                    self.menu_configuration()

            except (KeyboardInterrupt, EOFError):
                print("\n\nOperation interrupted by user. Exiting.\n")
                sys.exit(0)
            except Exception as e:
                print_error(f"Unexpected error: {e}")
                prompt_str("\nPress [Enter] to return to Main Menu", allow_empty=True)

    # =========================================================================
    # OPTION 1: PREDICT A RACE (Guided Workflow)
    # =========================================================================
    def menu_predict_race(self) -> None:
        clear_screen()
        print_banner(season=2026)
        print_header("Step 1: Select Season & Championship Year")
        year = prompt_int("Enter Season Year", min_val=2018, max_val=2030, default=2026)
        if year is None:
            return

        # Step 2: Select Circuit
        print_header("Step 2: Select Circuit")
        track_keys = sorted(
            self.tracks_cfg.keys(),
            key=lambda k: self.tracks_cfg[k].get(f"round_{year}", self.tracks_cfg[k].get("round_2026" if year == 2026 else None, 99))
        )

        rows = []
        key_map = {}
        for idx, k in enumerate(track_keys, start=1):
            info = self.tracks_cfg[k]
            r_num = info.get(f"round_{year}", info.get("round_2026" if year == 2026 else None, "-"))
            d_name = info.get("display_name", k.title())
            country = info.get("country", "")
            is_sprint = info.get(f"sprint_weekend_{year}", info.get("sprint_weekend", False))
            sprint_tag = "[SPRINT]" if is_sprint else "Normal"
            rows.append([idx, f"Round {r_num}", d_name, country, sprint_tag, k])
            key_map[str(idx)] = k
            key_map[k] = k

        print_table(["#", "Round", "Grand Prix", "Country", "Format", "Key"], rows)

        selected_input = prompt_choice("Select Circuit (# or key)", list(key_map.keys()) + ["0"], default="18")
        if selected_input == "0":
            return
        track_key = key_map[selected_input]
        track_info = self.tracks_cfg[track_key]
        track_display = track_info.get("display_name", track_key.title())
        is_sprint = track_info.get(f"sprint_weekend_{year}", track_info.get("sprint_weekend", False))
        round_num = track_info.get(f"round_{year}", track_info.get("round_2026" if year == 2026 else None, None))
        if round_num is None:
            print_error(f"Calendar round for track '{track_key}' is not configured for year {year} in tracks.json.")
            prompt_str("Press [Enter] to return", allow_empty=True)
            return

        # Step 3: Select Prediction Stage
        print_header("Step 3: Select Prediction Timeline Stage", f"{track_display} (Round {round_num})")
        if is_sprint:
            available_stages = [
                ("1", "pre_fp1", "Before weekend start (Zero telemetry, rolling form only)"),
                ("2", "after_fp1", "After Free Practice 1 telemetry observed"),
                ("3", "after_sprint_qualifying", "After Sprint Shootout results observed"),
                ("4", "after_sprint_race", "After Sprint Race classification observed"),
                ("5", "after_qualifying", "After Grand Prix Qualifying (Actual Starting Grid)"),
                ("6", "post_race", "Post-Race Evaluation (Verified actual finish)"),
            ]
        else:
            available_stages = [
                ("1", "pre_fp1", "Before weekend start (Zero telemetry, rolling form only)"),
                ("2", "after_fp1", "After Free Practice 1 telemetry observed"),
                ("3", "after_fp2", "After Free Practice 2 telemetry observed"),
                ("4", "after_fp3", "After Free Practice 3 telemetry observed"),
                ("5", "after_qualifying", "After Grand Prix Qualifying (Actual Starting Grid)"),
                ("6", "post_race", "Post-Race Evaluation (Verified actual finish)"),
            ]

        for num, stg, desc in available_stages:
            print(f"  {num}. {stg:<24} - {desc}")
        print("  0. Cancel / Back\n")

        stage_choice_keys = [s[0] for s in available_stages] + [s[1] for s in available_stages] + ["0"]
        chosen_stage_val = prompt_choice("Select Stage (# or name)", stage_choice_keys, default="5")
        if chosen_stage_val == "0":
            return

        stage = next((s[1] for s in available_stages if s[0] == chosen_stage_val or s[1] == chosen_stage_val), "after_qualifying")

        # Step 4: Data Readiness Check
        clear_screen()
        print_banner(season=year)
        print_header("Step 4: Data Readiness Audit", f"Circuit: {track_display} | Stage: {stage}")

        ready, details = self._audit_prediction_readiness(track_key, year, round_num, stage, is_sprint)

        for status, msg in details:
            if status == "OK":
                print_success(msg)
            elif status == "WARN":
                print_warning(msg)
            elif status == "MISSING":
                print_error(msg)
            else:
                print_info(msg)

        print(f"\n{Colors.BOLD}Prediction Readiness:{Colors.RESET} " + (f"{Colors.GREEN}YES{Colors.RESET}" if ready else f"{Colors.RED}NO (Action Needed){Colors.RESET}"))

        # Step 5: Missing-data Handling
        if not ready:
            print("\nRequired session data is missing for the selected stage.")
            print("  1. Collect missing session data via FastF1")
            print("  2. Proceed anyway using fallback provisional models (if permissible)")
            print("  3. Change stage")
            print("  0. Cancel prediction\n")
            action = prompt_choice("Select action", ["1", "2", "3", "0"], default="1")
            if action == "1":
                self._run_data_collection_wizard(track_key, year, round_num)
                # Re-audit
                ready, _ = self._audit_prediction_readiness(track_key, year, round_num, stage, is_sprint)
                if not ready:
                    print_error("Required data remains unverified after collection.")
                    if not prompt_confirm("Proceed with fallback model?", default=False):
                        return
            elif action == "2":
                print_warning("Proceeding with provisional fallback model. Actual starting grid will NOT be used.")
            elif action == "3":
                return self.menu_predict_race()
            else:
                return

        # Step 6: Prediction Options
        print_header("Step 5: Output Display Options")
        print("  1. Top 3 (Podium prediction)")
        print("  2. Top 10 (Points scoring positions)")
        print("  3. Full Grid (All 22 classified positions)")
        print("  0. Back\n")
        disp_choice = prompt_choice("Select display scope", ["1", "2", "3", "0"], default="2")
        if disp_choice == "0":
            return
        top_n = 3 if disp_choice == "1" else (10 if disp_choice == "2" else 22)

        # Step 7: Execute Prediction
        print_header("Executing Prediction Pipeline", f"Running 25,000 Monte Carlo Simulations...")
        pipeline = self._get_pipeline()
        try:
            results = pipeline.run(track_key=track_key, year=year, stage=stage)
        except Exception as e:
            print_error(f"Pipeline execution error: {e}")
            prompt_str("Press [Enter] to return", allow_empty=True)
            return

        # Display Results
        self._display_prediction_results(results, top_n=top_n)

        # Step 8: Post-Prediction Actions
        self._handle_post_prediction_actions(results)

    def _audit_prediction_readiness(
        self, track_key: str, year: int, round_num: int, stage: str, is_sprint: bool
    ) -> (bool, List[tuple]):
        """Inspects disk files and validates schemas to determine if the pipeline is ready to run."""
        details = []
        is_ready = True

        # 1. Historical training rounds (1 .. round_num-1)
        train_rounds = list(range(1, round_num))
        season_dir = self.loader.raw_data_dir / "season" / str(year)
        completed_historical = 0
        for r in train_rounds:
            rf = season_dir / f"{year}_round_{r}_race.csv"
            if rf.exists():
                completed_historical += 1

        if round_num <= 1 or len(train_rounds) == 0:
            details.append(("MISSING", f"Round 1 Season Opener: No prior {year} rounds exist to train predictive models. Minimum 1 completed round is required."))
            is_ready = False
        elif completed_historical == len(train_rounds):
            details.append(("OK", f"Historical Training Rounds: {completed_historical}/{len(train_rounds)} completed rounds available ({completed_historical*22} driver-race observations)."))
        elif completed_historical > 0:
            details.append(("WARN", f"Historical Training Rounds: {completed_historical}/{len(train_rounds)} completed rounds found in season directory."))
        else:
            details.append(("MISSING", f"No completed season rounds found in data/raw/season/{year}/ for Round {round_num}."))
            is_ready = False

        # 2. Stage-specific weekend session readiness
        sessions = self.loader.load_weekend_sessions(track_key, year=year, stage=stage)

        if stage == "pre_fp1":
            details.append(("OK", "Weekend Stage 'pre_fp1': Zero weekend telemetry required (pure pre-event rolling form)."))
        elif stage == "after_fp1":
            fp1 = sessions.get("fp1")
            if fp1 is not None and not fp1.empty:
                details.append(("OK", f"FP1 Practice telemetry available ({len(fp1)} drivers recorded)."))
            else:
                details.append(("MISSING", f"FP1 Practice telemetry NOT found for {track_key.title()} {year}."))
                is_ready = False
        elif stage == "after_fp2":
            for s in ("fp1", "fp2"):
                sess_df = sessions.get(s)
                if sess_df is not None and not sess_df.empty:
                    details.append(("OK", f"{s.upper()} Practice telemetry available ({len(sess_df)} drivers recorded)."))
                else:
                    details.append(("MISSING", f"{s.upper()} Practice telemetry NOT found for {track_key.title()} {year}."))
                    is_ready = False
        elif stage == "after_fp3":
            for s in ("fp1", "fp2", "fp3"):
                sess_df = sessions.get(s)
                if sess_df is not None and not sess_df.empty:
                    details.append(("OK", f"{s.upper()} Practice telemetry available ({len(sess_df)} drivers recorded)."))
                else:
                    details.append(("MISSING", f"{s.upper()} Practice telemetry NOT found for {track_key.title()} {year}."))
                    is_ready = False
        elif stage == "after_sprint_qualifying":
            fp1 = sessions.get("fp1")
            if fp1 is not None and not fp1.empty:
                details.append(("OK", f"FP1 Practice telemetry available ({len(fp1)} drivers recorded)."))
            else:
                details.append(("MISSING", f"FP1 Practice telemetry NOT found for {track_key.title()} {year}."))
                is_ready = False
            sq = sessions.get("sprint_qualifying")
            if sq is None:
                sq = self.loader.load_target_sprint_qualifying(track_key, year)
            if sq is not None and not sq.empty:
                details.append(("OK", f"Sprint Qualifying session available ({len(sq)} drivers recorded)."))
            else:
                details.append(("MISSING", f"Sprint Qualifying NOT found for {track_key.title()} {year}."))
                is_ready = False
        elif stage == "after_sprint_race":
            for s in ("fp1", "sprint_qualifying", "sprint_race"):
                sess_df = sessions.get(s)
                if s == "sprint_race" and sess_df is None:
                    sess_df = self.loader.load_target_sprint_race(track_key, year)
                if sess_df is not None and not sess_df.empty:
                    details.append(("OK", f"{s.replace('_', ' ').title()} available ({len(sess_df)} drivers recorded)."))
                else:
                    details.append(("MISSING", f"{s.replace('_', ' ').title()} NOT found for {track_key.title()} {year}."))
                    is_ready = False
        elif stage in ["after_qualifying", "post_race"]:
            fp1_df = sessions.get("fp1")
            if fp1_df is not None and not fp1_df.empty:
                details.append(("OK", f"Weekend practice telemetry available ({len(fp1_df)} drivers)."))
            else:
                details.append(("INFO", "Weekend practice telemetry absent (optional: pipeline uses historical team pace fallback)."))

        # 3. Target qualifying (strictly required for after_qualifying and post_race)
        if stage in ["after_qualifying", "post_race"]:
            target_q = self.loader.load_target_qualifying(track_key, year)
            if target_q is not None and not target_q.empty:
                if "Position" in target_q.columns and target_q["Position"].notna().sum() > 0:
                    details.append(("OK", f"Official Qualifying Results available ({len(target_q)} drivers; Starting Grid verified)."))
                else:
                    details.append(("INVALID", f"Official Qualifying file exists but contains invalid/missing position data."))
                    is_ready = False
            else:
                details.append(("MISSING", f"Official Qualifying Results NOT found for {track_key.title()} {year}."))
                is_ready = False

        # 4. Target race (strictly required for post_race)
        if stage == "post_race":
            target_r = self.loader.load_target_race(track_key, year)
            if target_r is not None and not target_r.empty:
                pos_col = "Position" if "Position" in target_r.columns else "ActualFinish"
                if pos_col in target_r.columns and target_r[pos_col].notna().sum() > 0:
                    details.append(("OK", f"Official Race Results available ({len(target_r)} drivers for post-race evaluation)."))
                else:
                    details.append(("INVALID", f"Official Race file exists but contains invalid/missing position data."))
                    is_ready = False
            else:
                details.append(("MISSING", f"Official Race Results NOT found for {track_key.title()} {year}."))
                is_ready = False

        return is_ready, details

    def _display_prediction_results(self, results: Dict[str, Any], top_n: int = 10) -> None:
        """Formats and displays prediction tables to terminal."""
        df = results["predictions"]
        track_name = results["track_name"]
        year = results["year"]
        stage = results["stage"]

        if results.get("is_provisional_quali") and stage in ["after_qualifying", "post_race"]:
            print("\n" + Colors.YELLOW + "!" * 76 + Colors.RESET)
            print(Colors.YELLOW + " WARNING: Actual qualifying results were not found. Provisional model used." + Colors.RESET)
            print(Colors.YELLOW + "!" * 76 + Colors.RESET)

        print("\n" + "=" * 76)
        print(Colors.BOLD + Colors.CYAN + f" GRAND PRIX RACE TOP {min(top_n, len(df))} PREDICTION: {track_name.upper()} {year} [{stage.upper()}]" + Colors.RESET)
        print(f" Model Selected  : {results.get('selected_model', 'N/A')}")
        print(f" Selection Method: {results.get('selection_method', 'N/A')} (MAE: {results.get('selected_mae', 0.0):.4f})")
        if stage in ["after_qualifying", "post_race"]:
            grid_source = "Provisional Model (Fallback - Actual Quali Missing)" if results.get("is_provisional_quali") else "Actual Observed Qualifying Session"
        else:
            grid_source = f"Provisional Predicted Grid (Pre-Qualifying Model: {stage})"
        print(f" Starting Grid   : {Colors.BOLD}{grid_source}{Colors.RESET}")
        print("=" * 76)

        sub_df = df.iloc[:top_n]
        headers = ["Pos", "Driver", "Team", "Grid", "ExpFinish", "Win%", "Podium%", "Points%", "Range"]
        rows = []
        for _, r in sub_df.iterrows():
            pos = int(r["PredictedPosition"])
            drv = r["Abbreviation"]
            team = r["TeamName"][:20]
            grid = int(r["ActualStartingGrid"])
            exp = f"{r['ExpectedFinish']:.2f}"
            win = f"{r['WinProbability']:.1f}%"
            pod = f"{r['Top3Probability']:.1f}%"
            pts = f"{r['PointsProbability']:.1f}%"
            rng = f"P{int(r['P5'])}-P{int(r['P95'])}"
            rows.append([pos, drv, team, grid, exp, win, pod, pts, rng])

        align_right = [True, False, False, True, True, True, True, True, False]
        print_table(headers, rows, align_right=align_right)

        if results.get("post_race_metrics"):
            m = results["post_race_metrics"]
            print(Colors.CYAN + "-" * 76 + Colors.RESET)
            print(Colors.BOLD + " POST-RACE OUT-OF-SAMPLE EVALUATION SCORECARD" + Colors.RESET)
            print(Colors.CYAN + "-" * 76 + Colors.RESET)
            print(f"  MAE (Expected Finish) : {Colors.BOLD}{m['mae_expected']:.4f}{Colors.RESET} (Walk-Forward Benchmark: 3.5343)")
            print(f"  MAE (Predicted Pos)   : {m['mae_predicted']:.4f}")
            print(f"  Spearman Rank Corr    : {Colors.BOLD}{m['spearman_rank_corr']:.4f}{Colors.RESET} (Strong ordinal ranking)")
            print(f"  Top 10 Points Overlap : {m['top10_overlap']} / 10 ({m['top10_overlap']*10:.1f}%)")
            print(f"  Top 3 Podium Overlap  : {m['top3_overlap']} / 3 ({m['top3_overlap']/3*100:.1f}%)")
            print(f"  Podium Brier Score    : {m['podium_brier_score']:.4f}")
            print(Colors.CYAN + "-" * 76 + Colors.RESET)

        print(f"Artifacts and Markdown reports written to:\n  {Colors.DIM}{results['output_dir']}{Colors.RESET}\n")

    def _handle_post_prediction_actions(self, results: Dict[str, Any]) -> None:
        """Offers actions after prediction generation."""
        out_dir = Path(results["output_dir"])
        track_key = results["track_key"]
        year = results["year"]
        stage = results["stage"]

        while True:
            print(f"{Colors.BOLD}Post-Prediction Options:{Colors.RESET}")
            print("  1. View full prediction table (All 22 drivers)")
            print("  2. View Top 3 Race Report")
            print("  3. View Top 10 Race Report")
            print("  4. View full Markdown summary report")
            print("  5. View Post-Race Evaluation Report (if available)")
            print("  0. Return to Main Menu\n")

            choice = prompt_choice("Select option", ["1", "2", "3", "4", "5", "0"], default="0")
            if choice == "0":
                break
            elif choice == "1":
                self._display_prediction_results(results, top_n=22)
            elif choice == "2":
                p = out_dir / f"{track_key}_{year}_race_top3_report.md"
                self._view_markdown_file(p)
            elif choice == "3":
                p = out_dir / f"{track_key}_{year}_race_top10_report.md"
                self._view_markdown_file(p)
            elif choice == "4":
                p = out_dir / f"{track_key}_{year}_{stage}_report.md"
                self._view_markdown_file(p)
            elif choice == "5":
                p = out_dir / f"{track_key}_{year}_post_race_evaluation.md"
                self._view_markdown_file(p)

    def _view_markdown_file(self, path: Path) -> None:
        if not path.exists():
            print_warning(f"File not found: {path}")
            return
        content = path.read_text(encoding="utf-8", errors="replace")
        print("\n" + "=" * 76)
        print(f" FILE: {path.name}")
        print("=" * 76 + "\n")
        paginate_text(content)

    # =========================================================================
    # OPTION 2: LOAD / COLLECT DATA
    # =========================================================================
    def menu_collect_data(self) -> None:
        while True:
            clear_screen()
            print_banner(season=2026)
            print_header("Load & Collect Session Data")
            print("  1. Inspect existing dataset inventory")
            print("  2. Collect data for a specific circuit (Practice + Qualifying)")
            print("  3. Collect data for season rounds (Race + Qualifying via FastF1)")
            print("  4. Audit data integrity & validate schemas")
            print("  5. View missing sessions across 2026 calendar")
            print("  0. Back to Main Menu\n")

            c = prompt_choice("Select option", ["1", "2", "3", "4", "5", "0"], default="1")
            if c == "0":
                break
            elif c == "1":
                self._inspect_data_inventory()
            elif c == "2":
                self._run_track_collector()
            elif c == "3":
                self._run_season_collector()
            elif c == "4":
                self._audit_data_integrity()
            elif c == "5":
                self._view_calendar_missing_sessions()

    def _inspect_data_inventory(self) -> None:
        clear_screen()
        print_header("Dataset Inventory")
        raw_season = self.loader.raw_data_dir / "season" / "2026"
        raw_tracks = self.loader.raw_data_dir / "tracks"

        print(f"{Colors.BOLD}Season 2026 Files:{Colors.RESET}")
        if raw_season.exists():
            s_files = sorted(raw_season.glob("*.csv"))
            print(f"  Total CSV files: {len(s_files)}")
            for f in s_files[:15]:
                try:
                    df = pd.read_csv(f)
                    print(f"    - {f.name:<38} ({len(df)} rows, {len(df.columns)} cols)")
                except Exception:
                    print(f"    - {f.name}")
            if len(s_files) > 15:
                print(f"    ... and {len(s_files)-15} more files.")
        else:
            print("  (Season directory not yet created)")

        print(f"\n{Colors.BOLD}Track-Specific Directories:{Colors.RESET}")
        if raw_tracks.exists():
            t_dirs = [d for d in raw_tracks.iterdir() if d.is_dir()]
            for d in sorted(t_dirs):
                c_files = list(d.rglob("*.csv"))
                print(f"    - {d.name:<18} ({len(c_files)} CSV files)")
        print()
        prompt_str("Press [Enter] to continue", allow_empty=True)

    def _run_track_collector(self) -> None:
        print_header("Collect Track Weekend Telemetry (FastF1)")
        t_key = prompt_str("Enter circuit key (e.g. austin, sepang, baku)", default="austin")
        if t_key not in self.tracks_cfg:
            print_error(f"Track '{t_key}' not in configuration.")
            prompt_str("Press [Enter] to return", allow_empty=True)
            return

        year = prompt_int("Enter Year", default=2026)
        force = prompt_confirm("Force refresh cache?", default=False)

        print_info(f"Running collect_track for {t_key.title()} {year}...")
        try:
            from src.data.collect_track import collect_track_data
            collect_track_data(t_key, year=year, force=force)
            print_success(f"Track collection completed for {t_key.title()}.")
        except Exception as e:
            print_error(f"Collection failed: {e}")
        prompt_str("Press [Enter] to continue", allow_empty=True)

    def _run_season_collector(self) -> None:
        print_header("Collect Season Rounds (Race & Qualifying)")
        year = prompt_int("Enter Year", default=2026)
        rounds_str = prompt_str("Enter round numbers separated by space (e.g. '18' or '1 2 3')", default="18")
        try:
            r_list = [int(x) for x in rounds_str.split()]
        except ValueError:
            print_error("Invalid round numbers.")
            prompt_str("Press [Enter] to return", allow_empty=True)
            return

        force = prompt_confirm("Force redownload?", default=False)
        print_info(f"Collecting rounds {r_list} for season {year}...")
        try:
            from src.data.collect_season import collect_season_data
            collect_season_data(year=year, rounds=r_list, force=force)
            print_success("Season data collection completed.")
        except Exception as e:
            print_error(f"Season collection failed: {e}")
        prompt_str("Press [Enter] to continue", allow_empty=True)

    def _audit_data_integrity(self) -> None:
        print_header("Dataset Integrity & Anti-Leakage Audit")
        t_key = prompt_str("Enter circuit key to audit against", default="austin")
        if t_key not in self.tracks_cfg:
            print_error(f"Unknown circuit '{t_key}'.")
            prompt_str("Press [Enter] to return", allow_empty=True)
            return

        pipeline = self._get_pipeline()
        r_num = self.tracks_cfg[t_key].get("round_2026", 18)
        train_rounds = list(range(1, r_num))

        races_df, qualis_df = pipeline.data_loader.load_season_rounds(2026, train_rounds)
        sessions_by_round = pipeline.data_loader.load_all_practice_sessions(2026, train_rounds)
        weekend_sessions = pipeline.data_loader.load_weekend_sessions(t_key, 2026, stage="after_qualifying")
        target_q = pipeline.data_loader.load_target_qualifying(t_key, 2026)

        if target_q is None:
            print_warning(f"Qualifying data for {t_key.title()} not found. Generating provisional...")
            meta = pipeline.predict_qualifying(t_key, 2026)
            target_q = meta["predictions"][["Abbreviation", "TeamName", "PredictedPosition"]].copy()
            target_q.rename(columns={"PredictedPosition": "Position"}, inplace=True)

        train_feats = pipeline.feature_engineer.build_training_dataset(
            races_df=races_df,
            qualis_df=qualis_df,
            practice_sessions_by_round=sessions_by_round,
            train_rounds=train_rounds
        )
        target_feats = pipeline.feature_engineer.build_target_dataset(
            target_round=r_num,
            track_name=self.tracks_cfg[t_key].get("display_name", t_key.title()),
            quali_df=target_q,
            practice_sessions=weekend_sessions,
            season_races_df=races_df,
            season_qualis_df=qualis_df,
            penalties=self.tracks_cfg[t_key].get("known_penalties_2026", {})
        )

        audit = pipeline.data_validator.audit_leakage(train_feats, target_feats, target_round=r_num)
        print("\nAudit Results:")
        print(f"  Clean of Leakage: {audit['is_clean']}")
        print(f"  Training Samples: {audit['training_samples']}")
        print(f"  Target Drivers  : {audit['target_drivers']}")
        if audit["violations"]:
            for v in audit["violations"]:
                print_error(v)
        else:
            print_success("Zero future-race or post-race leakage detected.")
        prompt_str("\nPress [Enter] to continue", allow_empty=True)

    def _view_calendar_missing_sessions(self) -> None:
        clear_screen()
        print_header("2026 Calendar Session Availability")
        rows = []
        season_dir = self.loader.raw_data_dir / "season" / "2026"
        for k, info in sorted(self.tracks_cfg.items(), key=lambda x: x[1].get("round_2026", 99)):
            r = info.get("round_2026", 0)
            name = info.get("display_name", k.title())
            is_sprint = info.get("sprint_weekend_2026", False)

            # Check track dir
            td = self.loader.raw_data_dir / "tracks" / k / "2026"
            has_fp = (td / "fp1.csv").exists()
            # Check season dir
            has_q = (season_dir / f"2026_round_{r}_qualifying.csv").exists() or (td / "qualifying.csv").exists()
            has_r = (season_dir / f"2026_round_{r}_race.csv").exists()

            fp_s = "Yes" if has_fp else "No"
            q_s = "Yes" if has_q else "No"
            r_s = "Yes" if has_r else "No"
            rows.append([r, name, "Sprint" if is_sprint else "Normal", fp_s, q_s, r_s])

        print_table(["Round", "Grand Prix", "Format", "Practice", "Qualifying", "Race"], rows)
        prompt_str("Press [Enter] to continue", allow_empty=True)

    def _run_data_collection_wizard(self, track_key: str, year: int, round_num: int) -> None:
        print_info(f"Attempting automated data collection for {track_key.title()} (Round {round_num})...")
        try:
            from src.data.collect_season import collect_season_data
            from src.data.collect_track import collect_track_data
            collect_season_data(year=year, rounds=[round_num])
            collect_track_data(track_id=track_key, year=year)
            print_success("Automated collection routine finished.")
        except Exception as e:
            print_error(f"Automated collection encountered an error: {e}")

    # =========================================================================
    # OPTION 3: VIEW PREVIOUS PREDICTIONS & REPORTS
    # =========================================================================
    def menu_view_outputs(self) -> None:
        while True:
            clear_screen()
            print_banner(season=2026)
            print_header("Inspect Saved Predictions & Reports")
            out_root = self.root_dir / "outputs"
            if not out_root.exists():
                print_warning("No outputs/ directory detected.")
                prompt_str("Press [Enter] to return", allow_empty=True)
                return

            track_dirs = sorted([d.name for d in out_root.iterdir() if d.is_dir()])
            if not track_dirs:
                print_info("No track-specific output folders found in outputs/.")
                prompt_str("Press [Enter] to return", allow_empty=True)
                return

            for i, td in enumerate(track_dirs, start=1):
                cnt = len(list((out_root / td).glob("*.*")))
                print(f"  {i}. {td.upper():<16} ({cnt} generated artifacts)")
            print("\n  0. Back to Main Menu\n")

            c = prompt_choice("Select circuit folder", [str(i) for i in range(1, len(track_dirs) + 1)] + ["0"], default="1")
            if c == "0":
                break

            selected_track = track_dirs[int(c) - 1]
            self._browse_track_outputs(out_root / selected_track)

    def _browse_track_outputs(self, track_dir: Path) -> None:
        while True:
            clear_screen()
            print_header(f"Artifacts: {track_dir.name.upper()}", str(track_dir))
            files = sorted(track_dir.glob("*.*"), key=lambda p: p.name)
            if not files:
                print("  (Empty directory)\n")
                prompt_str("Press [Enter] to return", allow_empty=True)
                return

            for i, f in enumerate(files, start=1):
                sz = f"{f.stat().st_size / 1024:.1f} KB"
                print(f"  {i:>2}. {f.name:<45} [{sz}]")
            print("\n   0. Back\n")

            c = prompt_choice("Select file to view", [str(i) for i in range(1, len(files) + 1)] + ["0"], default="1")
            if c == "0":
                break

            target_file = files[int(c) - 1]
            if target_file.suffix == ".csv":
                self._view_csv_file(target_file)
            elif target_file.suffix == ".md":
                self._view_markdown_file(target_file)
            else:
                print_info(f"File: {target_file}")
                prompt_str("Press [Enter] to return", allow_empty=True)

    def _view_csv_file(self, path: Path) -> None:
        try:
            df = pd.read_csv(path)
            print("\n" + "=" * 76)
            print(f" TABLE: {path.name} ({len(df)} rows, {len(df.columns)} columns)")
            print("=" * 76)
            # Show top columns for readability
            cols = [c for c in ["PredictedPosition", "Position", "Abbreviation", "TeamName", "ActualStartingGrid", "ExpectedFinish", "WinProbability", "Top3Probability", "PointsProbability", "ActualFinish"] if c in df.columns]
            if not cols:
                cols = list(df.columns[:7])
            sub = df[cols].head(22)
            rows = sub.values.tolist()
            print_table(cols, rows)
        except Exception as e:
            print_error(f"Could not load CSV: {e}")
        prompt_str("Press [Enter] to return", allow_empty=True)

    # =========================================================================
    # OPTION 4: EVALUATE A PREDICTION
    # =========================================================================
    def menu_evaluate_prediction(self) -> None:
        clear_screen()
        print_banner(season=2026)
        print_header("Out-of-Sample Race Prediction Evaluation")
        print("Select a previously generated prediction CSV to evaluate against official race results.\n")

        out_root = self.root_dir / "outputs"
        pred_files = sorted(out_root.rglob("*_predictions.csv"))
        if not pred_files:
            print_warning("No saved prediction CSV files found in outputs/.")
            prompt_str("Press [Enter] to return", allow_empty=True)
            return

        for idx, pf in enumerate(pred_files, start=1):
            rel = pf.relative_to(out_root)
            print(f"  {idx:>2}. {str(rel)}")
        print("\n   0. Cancel / Back\n")

        c = prompt_choice("Select prediction file to evaluate", [str(i) for i in range(1, len(pred_files) + 1)] + ["0"], default="1")
        if c == "0":
            return

        target_csv = pred_files[int(c) - 1]
        self._evaluate_saved_prediction_file(target_csv)

    def _evaluate_saved_prediction_file(self, pred_csv_path: Path) -> None:
        print_info(f"Loading forecast from {pred_csv_path.name}...")
        try:
            pred_df = pd.read_csv(pred_csv_path)
        except Exception as e:
            print_error(f"Failed to read CSV: {e}")
            prompt_str("Press [Enter] to return", allow_empty=True)
            return

        track_folder = pred_csv_path.parent.name
        if track_folder not in self.tracks_cfg:
            prefix = pred_csv_path.stem.split("_")[0].lower()
            if prefix in self.tracks_cfg:
                track_folder = prefix

        year = 2026

        # Locate actual results via DataLoader
        actual_df = self.loader.load_target_race(track_folder, year=year)
        if actual_df is None or actual_df.empty:
            print_warning(f"Official race results not found for {track_folder.title()} {year}.")
            if prompt_confirm("Attempt to collect official race results via FastF1?", default=True):
                info = self.tracks_cfg.get(track_folder, {})
                r_num = info.get("round_2026", 18)
                from src.data.collect_season import collect_season_data
                collect_season_data(year=year, rounds=[r_num])
                actual_df = self.loader.load_target_race(track_folder, year=year)

        if actual_df is None or actual_df.empty:
            print_error(f"Cannot evaluate: official race results unavailable for {track_folder.title()}.")
            prompt_str("Press [Enter] to return", allow_empty=True)
            return

        # Check driver validity and handle duplicate or missing drivers
        pred_duplicates = pred_df[pred_df["Abbreviation"].duplicated()]
        if not pred_duplicates.empty:
            dups = list(pred_duplicates["Abbreviation"].unique())
            print_warning(f"Detected duplicate drivers in forecast CSV: {dups}. Deduplicating to keep first entry.")
            pred_df = pred_df.drop_duplicates(subset=["Abbreviation"], keep="first").reset_index(drop=True)

        actual_duplicates = actual_df[actual_df["Abbreviation"].duplicated()]
        if not actual_duplicates.empty:
            dups = list(actual_duplicates["Abbreviation"].unique())
            print_warning(f"Detected duplicate drivers in actual race results: {dups}. Deduplicating to keep first entry.")
            actual_df = actual_df.drop_duplicates(subset=["Abbreviation"], keep="first").reset_index(drop=True)

        pred_drivers = set(pred_df["Abbreviation"])
        act_drivers = set(actual_df["Abbreviation"])
        missing_in_act = pred_drivers - act_drivers
        missing_in_pred = act_drivers - pred_drivers
        if missing_in_act:
            print_warning(f"Drivers present in forecast but missing from actual race: {sorted(missing_in_act)}")
        if missing_in_pred:
            print_warning(f"Drivers present in actual race but missing from forecast: {sorted(missing_in_pred)}")

        pipeline = self._get_pipeline()
        metrics = pipeline._evaluate_race_predictions(pred_df, actual_df)

        print("\n" + "=" * 76)
        print(Colors.BOLD + Colors.CYAN + f" OUT-OF-SAMPLE EVALUATION: {track_folder.upper()} {year}" + Colors.RESET)
        print(f" Source Forecast : {pred_csv_path.name}")
        print(f" Actual Results  : {len(actual_df)} drivers verified (Evaluated: {metrics['evaluated_drivers']} matched)")
        print("=" * 76)
        print(f"  Mean Absolute Error (Expected Finish) : {Colors.BOLD}{metrics['mae_expected']:.4f}{Colors.RESET} (Walk-Forward Benchmark: 3.5343)")
        print(f"  Mean Absolute Error (Predicted Pos)   : {metrics['mae_predicted']:.4f}")
        print(f"  Spearman Rank Correlation             : {Colors.BOLD}{metrics['spearman_rank_corr']:.4f}{Colors.RESET}")
        print(f"  Top 10 Points Scorers Overlap         : {metrics['top10_overlap']} / 10 ({metrics['top10_overlap']*10:.1f}%)")
        print(f"  Top 3 Exact Podium Overlap            : {metrics['top3_overlap']} / 3 ({metrics['top3_overlap']/3*100:.1f}%)")
        print(f"  Multi-Driver Podium Brier Score       : {metrics['podium_brier_score']:.4f}")
        print("=" * 76)

        # Merge comparison
        pos_col = "Position" if "Position" in actual_df.columns else "ActualFinish"
        actual_df["ActualFinish"] = pd.to_numeric(actual_df[pos_col], errors="coerce")
        merged = pd.merge(pred_df, actual_df[["Abbreviation", "ActualFinish"]], on="Abbreviation").sort_values("ActualFinish")

        headers = ["ActualPos", "Driver", "Grid", "PredictedPos", "ExpFinish", "Delta"]
        rows = []
        for _, r in merged.iterrows():
            act = int(r["ActualFinish"])
            drv = r["Abbreviation"]
            grid = int(r.get("ActualStartingGrid", 0))
            pred = int(r["PredictedPosition"])
            exp = f"{r['ExpectedFinish']:.2f}"
            delta = pred - act
            delta_str = f"+{delta}" if delta > 0 else f"{delta}"
            rows.append([act, drv, grid, pred, exp, delta_str])

        print_table(headers, rows, align_right=[True, False, True, True, True, True])
        prompt_str("Press [Enter] to continue", allow_empty=True)

    # =========================================================================
    # OPTION 5: SYSTEM HEALTH & DIAGNOSTICS
    # =========================================================================
    def menu_system_health(self) -> None:
        clear_screen()
        print_banner(season=2026)
        print_header("System Health & Diagnostic Center")

        # 1. Environment
        print(f"{Colors.BOLD}Environment & Core Dependencies:{Colors.RESET}")
        print_success(f"Python Runtime: {sys.version.split()[0]} ({sys.platform})")
        for pkg in ["pandas", "numpy", "scipy", "sklearn", "xgboost", "fastf1", "pytest"]:
            try:
                mod = __import__(pkg)
                ver = getattr(mod, "__version__", "installed")
                print_success(f"Library {pkg:<12}: {ver}")
            except ImportError:
                print_error(f"Library {pkg:<12}: NOT INSTALLED")

        # 2. Configuration
        print(f"\n{Colors.BOLD}Configuration Files:{Colors.RESET}")
        for c_file in ["tracks.json", "model_config.json", "settings.json"]:
            p = self.root_dir / "config" / c_file
            if p.exists():
                print_success(f"Config '{c_file}' present ({p.stat().st_size} bytes).")
            else:
                print_error(f"Config '{c_file}' missing.")

        # 3. Pipelines
        print(f"\n{Colors.BOLD}Pipeline Modules:{Colors.RESET}")
        try:
            self._get_pipeline()
            print_success("RacePredictionPipeline initialized without errors.")
        except Exception as e:
            print_error(f"RacePredictionPipeline import failure: {e}")

        # 4. Storage & Directories
        print(f"\n{Colors.BOLD}Data & Storage Directories:{Colors.RESET}")
        for d_name in ["data/raw/season", "data/raw/tracks", "data/cache", "outputs"]:
            d_path = self.root_dir / d_name
            if d_path.exists():
                print_success(f"Directory '{d_name}' accessible.")
            else:
                print_warning(f"Directory '{d_name}' not yet created.")

        # 5. Git Status
        print(f"\n{Colors.BOLD}Git Working Tree Status:{Colors.RESET}")
        try:
            res = subprocess.run(["git", "status", "--short"], cwd=self.root_dir, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                lines = res.stdout.strip().splitlines()
                if not lines:
                    print_success("Working tree clean (zero uncommitted changes).")
                else:
                    print_info(f"Working tree has {len(lines)} modified/untracked files.")
                    for l in lines[:5]:
                        print(f"     {l}")
                    if len(lines) > 5:
                        print(f"     ... and {len(lines)-5} more.")
            else:
                print_warning("Git command exited with non-zero status.")
        except Exception as e:
            print_warning(f"Could not query git: {e}")

        # Action: Option to run tests
        print(f"\n{Colors.BOLD}Diagnostics Options:{Colors.RESET}")
        print("  1. Run Fast Unit Tests (56 tests)")
        print("  2. Run COTA Session Discovery Tests (6 tests)")
        print("  0. Back to Main Menu\n")
        c = prompt_choice("Select action", ["1", "2", "0"], default="0")
        if c == "1":
            print_info("Executing fast unit tests...")
            subprocess.run([
                sys.executable, "-m", "pytest",
                "tests/test_leakage.py", "tests/test_sprint_architecture.py",
                "tests/test_config_driven_circuits.py", "tests/test_grid_conversion.py",
                "tests/test_grid_penalties.py", "tests/test_sprint_features.py",
                "tests/test_teammate_features.py", "tests/test_timing.py"
            ], cwd=self.root_dir)
            prompt_str("\nPress [Enter] to continue", allow_empty=True)
        elif c == "2":
            print_info("Executing COTA session discovery tests...")
            subprocess.run([sys.executable, "-m", "pytest", "tests/test_cota_session_discovery.py"], cwd=self.root_dir)
            prompt_str("\nPress [Enter] to continue", allow_empty=True)

    # =========================================================================
    # OPTION 6: CONFIGURATION & SUPPORTED CIRCUITS
    # =========================================================================
    def menu_configuration(self) -> None:
        clear_screen()
        print_banner(season=2026)
        print_header("System Configuration & 2026 Circuit Registry")

        print(f"{Colors.BOLD}Model Hyperparameters & Configuration:{Colors.RESET}")
        m_sel = self.model_cfg.get("model_selection", {})
        blend = self.model_cfg.get("blending", {})
        mc = self.model_cfg.get("monte_carlo", {})
        print(f"  Random Seed           : {self.model_cfg.get('random_seed', 42)}")
        print(f"  Ensemble Architecture : {m_sel.get('fixed_rf_weight', 0.6)*100:.0f}% Random Forest / {m_sel.get('fixed_xgb_weight', 0.4)*100:.0f}% XGBoost")
        print(f"  Blending Ratio        : {blend.get('ml_weight', 0.65)*100:.0f}% ML Prediction / {blend.get('grid_weight', 0.35)*100:.0f}% Starting Grid")
        print(f"  Monte Carlo Sims      : {mc.get('default_simulations', 25000):,} draws per driver")
        print(f"  EWMA Form Half-Life   : alpha = {self.model_cfg.get('ewma_alpha', 0.3)}")
        print(f"  Expected Grid Size    : {self.model_cfg.get('expected_grid_size', 22)} drivers")

        print(f"\n{Colors.BOLD}2026 Championship Calendar (23 Configured Circuits):{Colors.RESET}")
        rows = []
        for k, info in sorted(self.tracks_cfg.items(), key=lambda x: x[1].get("round_2026", 99)):
            r = info.get("round_2026", "-")
            name = info.get("display_name", k.title())
            c_type = info.get("circuit_type", "permanent")
            km = f"{info.get('track_length_km', 0.0):.3f} km"
            sprint = "Yes" if info.get("sprint_weekend_2026", False) else "No"
            penalties = len(info.get("known_penalties_2026", {}))
            pen_str = f"{penalties} penalties" if penalties > 0 else "None"
            rows.append([r, name, c_type, km, sprint, pen_str, k])

        print_table(["Round", "Grand Prix", "Type", "Length", "Sprint", "Grid Penalties", "Key"], rows)
        print(f"{Colors.DIM}Note: Configuration is read-only. Settings live in config/tracks.json and config/model_config.json.{Colors.RESET}\n")
        prompt_str("Press [Enter] to return to Main Menu", allow_empty=True)
