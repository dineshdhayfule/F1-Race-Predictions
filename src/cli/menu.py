"""
Interactive Terminal Menu for ApexGrid — Motorsport Intelligence & Race Prediction Platform.
Predicting the Race Beyond the Grid.
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
from typing import Dict, Any, List, Optional, Tuple
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
from src.features.grid_features import format_grid_provenance_display, GridProvenance
from src.data.data_loader import DataLoader
from src.data.data_validator import DataValidator
from src.data.season_discovery import SeasonDiscovery
from src.features.build_features import FeatureEngineer
from src.pipeline.race_prediction_pipeline import RacePredictionPipeline
from src.data.data_manager import DataManager
from src.utils.logging_utils import get_logger

ROOT = Path(__file__).resolve().parents[2]
logger = get_logger("menu")


def prompt_custom_weights_only() -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """Prompt for custom RF and XGBoost weights with validation and confirmation."""
    while True:
        try:
            rf_input = input("Enter Random Forest weight (%): ").strip()
            xgb_input = input("Enter XGBoost weight (%): ").strip()
        except (KeyboardInterrupt, EOFError):
            return None, None, None

        try:
            rf_w = float(rf_input)
            xgb_w = float(xgb_input)
        except ValueError:
            print("Error: Both values must be numeric. Please try again.\n")
            continue

        if not (0.0 <= rf_w <= 100.0 and 0.0 <= xgb_w <= 100.0):
            print("Error: Each weight must be between 0 and 100. Please try again.\n")
            continue

        if abs((rf_w + xgb_w) - 100.0) > 1e-4:
            print(f"Error: Weight sum must equal exactly 100 (got {rf_w} + {xgb_w} = {rf_w + xgb_w}%). Please try again.\n")
            continue

        rf_disp = int(rf_w) if rf_w.is_integer() else rf_w
        xgb_disp = int(xgb_w) if xgb_w.is_integer() else xgb_w
        print()
        print(f"Random Forest : {rf_disp}%")
        print(f"XGBoost       : {xgb_disp}%")
        print()
        try:
            confirm = input("Confirm configuration? (Y/N): ").strip().upper()
        except (KeyboardInterrupt, EOFError):
            return None, None, None

        if confirm in ("Y", "YES", ""):
            return ("hybrid", rf_w, xgb_w)
        elif confirm in ("N", "NO"):
            return None, None, None
        else:
            print("Invalid response. Please enter Y or N.\n")
            continue


def prompt_model_selection() -> Tuple[str, Optional[float], Optional[float]]:
    """
    Interactive prompt for model selection when bare --model is supplied:

    ========== MODEL SELECTION ==========

    1. Random Forest (100%)
    2. XGBoost (100%)
    3. Hybrid Ensemble (Custom Weights)

    Select model:
    """
    while True:
        print("\n========== MODEL SELECTION ==========\n")
        print("1. Random Forest (100%)")
        print("2. XGBoost (100%)")
        print("3. Hybrid Ensemble (Custom Weights)")
        print("4. Ridge Regression (100%)")
        print("5. Validated Multi-Model Ensemble (40% RF / 30% XGB / 30% Ridge)\n")
        try:
            choice = input("Select model: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nModel selection cancelled.")
            sys.exit(0)

        if choice == "1":
            return ("rf", None, None)
        elif choice == "2":
            return ("xgb", None, None)
        elif choice == "3":
            model, rf_w, xgb_w = prompt_custom_weights_only()
            if model is None:
                # N returns to model selection
                continue
            return (model, rf_w, xgb_w)
        elif choice == "4":
            return ("ridge", None, None)
        elif choice == "5":
            return ("ensemble", None, None)
        else:
            print("Invalid choice. Please select 1, 2, 3, 4, or 5.\n")


class InteractiveMenu:
    """Master controller for the interactive CLI prediction engine menu."""

    def __init__(self, root_dir: Optional[Path] = None):
        self.root_dir = root_dir or ROOT
        self.loader = DataLoader(root_dir=self.root_dir)
        self.validator = DataValidator()
        self.tracks_cfg = self._load_tracks_config()
        self.model_cfg = self._load_model_config()
        self.pipeline: Optional[RacePredictionPipeline] = None
        self.data_manager = DataManager(root_dir=self.root_dir)

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
                print("  7. Data Management & Deletion")
                print("\n  0. Exit\n")

                choice = prompt_choice("Enter your choice", ["1", "2", "3", "4", "5", "6", "7", "0"], default="1")

                if choice == "0":
                    print("\nExiting ApexGrid. Goodbye!\n")
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
                elif choice == "7":
                    self.menu_data_management()

            except (KeyboardInterrupt, EOFError):
                print("\n\nOperation interrupted by user. Exiting.\n")
                sys.exit(0)
            except Exception as e:
                print_error(f"Unexpected error: {e}")
                prompt_str("\nPress [Enter] to return to Main Menu", allow_empty=True)

    # =========================================================================
    # OPTION 1: PREDICT A RACE (Guided Workflow)
    # =========================================================================
    def _get_supported_seasons(self, workflow: str = "prediction") -> List[int]:
        """Discovers verified and available championship seasons from disk."""
        discovery = SeasonDiscovery(root_dir=self.root_dir)
        return discovery.discover_seasons(workflow=workflow)

    def _get_default_circuit_for_season(self, year: int) -> Tuple[Optional[str], Optional[int]]:
        """
        Calculates the dynamic default circuit for a season:
        Finds the first uncompleted round in chronological order.
        If all rounds are completed, returns the last completed round.
        If no rounds completed, returns round 1.
        Returns (track_key, round_num).
        """
        circuits = []
        for k, info in self.tracks_cfg.items():
            r = info.get(f"round_{year}" if year != 2026 else "round_2026")
            if r is not None and isinstance(r, int):
                circuits.append((r, k))

        if not circuits:
            return None, None

        circuits.sort(key=lambda x: x[0])

        season_raw = self.loader.raw_data_dir / "season" / str(year)
        season_proc = self.loader.processed_data_dir / "season" / str(year)

        completed_rounds = set()
        for r_num, k in circuits:
            has_race = (
                (season_raw / f"{year}_round_{r_num}_race.csv").exists()
                or (season_proc / f"{year}_round_{r_num}_race.csv").exists()
                or (self.loader.raw_data_dir / "tracks" / k / str(year) / "race.csv").exists()
            )
            if has_race:
                completed_rounds.add(r_num)

        for r_num, k in circuits:
            if r_num not in completed_rounds:
                return k, r_num

        return circuits[-1][1], circuits[-1][0]

    def menu_predict_race(self) -> None:
        clear_screen()
        supported_seasons = self._get_supported_seasons(workflow="prediction")
        if not supported_seasons:
            print_banner(season=None, status_text="No Data")
            print_header("Step 1: Select Season & Championship Year")
            print_error("No usable championship seasons found with required prediction datasets.")
            print_info("Please run Option 2 (Load / Collect Data) first to download season telemetry.")
            prompt_str("Press [Enter] to return to Main Menu", allow_empty=True)
            return

        default_yr = SeasonDiscovery.get_default_season(supported_seasons, preferred=2026)

        print_banner(season=default_yr)
        print_header("Step 1: Select Season & Championship Year")
        formatted = SeasonDiscovery.format_available_seasons(supported_seasons)
        print_info(f"Available seasons: {formatted}")
        year = prompt_int("Enter Season Year", allowed_values=supported_seasons, default=default_yr)
        if year is None:
            return

        # Step 2: Select Circuit
        print_header("Step 2: Select Circuit")
        track_keys = [
            k for k in self.tracks_cfg.keys()
            if self.tracks_cfg[k].get(f"round_{year}" if year != 2026 else "round_2026") is not None
        ]
        if not track_keys:
            print_error(f"Season {year} has no configured calendar in tracks.json. Configured seasons: {supported_seasons}")
            prompt_str("Press [Enter] to return", allow_empty=True)
            return

        track_keys.sort(
            key=lambda k: self.tracks_cfg[k].get(f"round_{year}" if year != 2026 else "round_2026", 99)
        )

        default_track_key, default_round = self._get_default_circuit_for_season(year)

        rows = []
        key_map = {}
        default_choice = "1"
        for idx, k in enumerate(track_keys, start=1):
            info = self.tracks_cfg[k]
            r_num = info.get(f"round_{year}" if year != 2026 else "round_2026", "-")
            d_name = info.get("display_name", k.title())
            country = info.get("country", "")
            is_sprint = info.get(f"sprint_weekend_{year}", info.get("sprint_weekend", False))
            sprint_tag = "[SPRINT]" if is_sprint else "Normal"
            rows.append([idx, f"Round {r_num}", d_name, country, sprint_tag, k])
            key_map[str(idx)] = k
            key_map[k] = k
            if k == default_track_key:
                default_choice = str(idx)

        print_table(["#", "Round", "Grand Prix", "Country", "Format", "Key"], rows)

        selected_input = prompt_choice("Select Circuit (# or key)", list(key_map.keys()) + ["0"], default=default_choice)
        if selected_input == "0":
            return
        track_key = key_map[selected_input]
        track_info = self.tracks_cfg[track_key]
        track_display = track_info.get("display_name", track_key.title())
        is_sprint = track_info.get(f"sprint_weekend_{year}", track_info.get("sprint_weekend", False))
        round_num = track_info.get(f"round_{year}" if year != 2026 else "round_2026", None)
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

        # Step 5: Select Prediction Model
        print_header("Step 5: Select Prediction Model", f"Circuit: {track_display} | Stage: {stage}")
        print("  1. Random Forest (100%)")
        print("  2. XGBoost (100%)")
        print("  3. Ridge Regression (100%)")
        print("  4. Hybrid Ensemble (60% RF / 40% XGB) — Default")
        print("  5. Hybrid Ensemble (Custom Weights)")
        print("  6. Validated Multi-Model Ensemble (40% RF / 30% XGB / 30% Ridge)")
        print("  0. Cancel / Back\n")
        model_choice_input = prompt_choice("Select Model", ["1", "2", "3", "4", "5", "6", "0"], default="4")
        if model_choice_input == "0":
            return
        selected_custom_rf = None
        selected_custom_xgb = None
        if model_choice_input == "1":
            selected_model_opt = "rf"
        elif model_choice_input == "2":
            selected_model_opt = "xgb"
        elif model_choice_input == "3":
            selected_model_opt = "ridge"
        elif model_choice_input == "4":
            selected_model_opt = "hybrid"
        elif model_choice_input == "5":
            selected_model_opt, selected_custom_rf, selected_custom_xgb = prompt_custom_weights_only()
            if selected_model_opt is None:
                return self.menu_predict_race()
        elif model_choice_input == "6":
            selected_model_opt = "ensemble"
        else:
            selected_model_opt = "hybrid"

        # Step 6: Enhanced Simulation Capabilities
        print_header("Step 6: Enhanced Simulation Capabilities")
        print("  • Race Pace Intelligence (Long-run multi-lap stint extraction & fuel correction)")
        print("  • Tire Degradation Modelling (Compound wear slopes & thermal scaling)")
        print("  • Pit Stop Strategy Simulator (1/2-stop optimization & FIA 2-dry-compound rule)")
        print("  • Race Control Probability Model (Safety Car / VSC pack bunching & cheap pit stops)")
        print("  • Weather-Aware Race Predictions (Session telemetry, rain variance & wet driver skill)\n")
        enable_enhanced = prompt_confirm("Enable Five Major ML Capabilities?", default=True)
        caps = {
            "race_pace_intelligence": True,
            "tire_degradation_modeling": True,
            "pit_strategy_simulator": True,
            "race_control_model": True,
            "weather_aware_prediction": True,
        } if enable_enhanced else None

        # Step 7: Output Display Options
        print_header("Step 7: Output Display Options")
        print("  1. Top 3 (Podium prediction)")
        print("  2. Top 10 (Points scoring positions)")
        print("  3. Full Grid (All 22 classified positions)")
        print("  0. Back\n")
        disp_choice = prompt_choice("Select display scope", ["1", "2", "3", "0"], default="2")
        if disp_choice == "0":
            return
        top_n = 3 if disp_choice == "1" else (10 if disp_choice == "2" else 22)

        # Step 8: Execute Prediction
        print_header("Executing Prediction Pipeline", f"Running 25,000 Monte Carlo Simulations...")
        pipeline = self._get_pipeline()
        try:
            results = pipeline.run(
                track_key=track_key,
                year=year,
                stage=stage,
                model=selected_model_opt,
                rf_weight=selected_custom_rf,
                xgb_weight=selected_custom_xgb,
                capabilities=caps,
            )
        except Exception as e:
            logger.error(f"Pipeline execution error: {e}", exc_info=True)
            print_error(f"Pipeline execution error [{type(e).__name__}]: {e}")
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

        # 3. Target qualifying and starting grid verification (for after_qualifying and post_race)
        if stage in ["after_qualifying", "post_race"]:
            target_q = self.loader.load_target_qualifying(track_key, year)
            if target_q is not None and not target_q.empty:
                if "Position" in target_q.columns and target_q["Position"].notna().sum() > 0:
                    details.append(("OK", f"Qualifying Results available ({len(target_q)} drivers recorded)."))

                    # Starting grid verification
                    grid_df, grid_meta = self.loader.load_starting_grid(
                        track_id=track_key,
                        year=year,
                        round_num=round_num,
                        stage=stage,
                        quali_df=target_q,
                    )
                    prov = grid_meta.get("provenance", "")
                    if prov in ["official_manifest", "fia_official"]:
                        doc_ref = grid_meta.get("document_reference", "FIA Document")
                        details.append(("OK", f"Official FIA Starting Grid verified ({doc_ref})."))
                    elif prov == "f1_website":
                        details.append(("OK", "Official F1 Website Published Starting Grid verified."))
                    elif prov == "openf1_api":
                        doc_ref = grid_meta.get("document_reference", "OpenF1 API")
                        details.append(("INFO", f"OpenF1 Published Starting Grid available ({doc_ref})."))
                    elif prov in ["algorithmic_fallback", "algorithmic_provisional"]:
                        details.append(("INFO", "No official manifest found; Algorithmic Provisional Grid (FIA Art 42.3) available."))
                    else:
                        details.append(("WARN", "Starting grid unverified (provisional resolution only)."))
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
        mae_val = results.get('selected_mae')
        mae_str = f"{mae_val:.4f}" if isinstance(mae_val, (int, float)) else str(mae_val)
        print(f" Model Selected  : {results.get('selected_model', 'N/A')}")
        print(f" Selection Method: {results.get('selection_method', 'N/A')} (MAE: {mae_str})")
        grid_meta = results.get("grid_meta", {})
        grid_source = format_grid_provenance_display(grid_meta)
        print(f" Starting Grid   : {Colors.BOLD}{grid_source}{Colors.RESET}")
        print("=" * 76)

        sub_df = df.iloc[:top_n]
        headers = ["Pos", "Driver", "Team", "Grid", "ExpFinish", "Win%", "Podium%", "Points%", "Range"]
        rows = []
        for _, r in sub_df.iterrows():
            pos = int(r["PredictedPosition"])
            drv = r["Abbreviation"]
            team = r["TeamName"][:20]
            grid_raw = r.get("ActualStartingGrid")
            raw_pit = r.get("is_pit_lane_start", 0)
            is_pit = pd.notna(raw_pit) and int(round(float(raw_pit))) == 1
            if is_pit:
                grid = "PL"
            elif pd.notna(grid_raw):
                grid = str(int(round(float(grid_raw))))
            else:
                grid = "-"
            exp = f"{r['ExpectedFinish']:.2f}"
            win = f"{r['WinProbability']:.1f}%"
            pod = f"{r['Top3Probability']:.1f}%"
            pts = f"{r['PointsProbability']:.1f}%"
            rng = f"P{int(r['P5'])}-P{int(r['P95'])}"
            rows.append([pos, drv, team, grid, exp, win, pod, pts, rng])

        align_right = [True, False, False, True, True, True, True, True, False]
        print_table(headers, rows, align_right=align_right)

        if "RecommendedStrategy" in df.columns:
            print("\n" + "=" * 76)
            print(Colors.BOLD + Colors.GREEN + " STRATEGIC & ENVIRONMENTAL RACE INTELLIGENCE" + Colors.RESET)
            sim_meta = results.get("simulation_meta", {})
            sc_p = sim_meta.get("p_safety_car", "N/A")
            rain_p = sim_meta.get("rain_probability", "N/A")
            print(f" Safety Car Probability: {Colors.BOLD}{sc_p}{Colors.RESET} | Rain Probability: {Colors.BOLD}{rain_p}{Colors.RESET}")
            print(Colors.GREEN + "-" * 76 + Colors.RESET)
            strat_headers = ["Driver", "Recommended Strategy", "Stops", "Delta to Alt"]
            strat_rows = []
            for _, r in sub_df.iterrows():
                rec_strat = str(r.get("RecommendedStrategy", "1-Stop"))
                stops = str(int(r.get("OptimalStops", 1)))
                delta_sec = f"{float(r.get('StrategyDeltaSec', 0.0)):.1f}s"
                strat_rows.append([r["Abbreviation"], rec_strat, stops, delta_sec])
            print_table(strat_headers, strat_rows, align_right=[False, False, True, True])

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
            print("  2. Collect data for an ENTIRE season (Interactive: Full Weekend, Quali+Race, Practice)")
            print("  3. Collect data for specific season rounds (Interactive: Full Weekend, Quali+Race, Practice)")
            print("  4. Collect data for a specific circuit (Interactive: Full Weekend, Quali+Race, Practice)")
            print("  5. Collect historical circuit data (configured tracks & year ranges)")
            print("  6. Audit data integrity & validate schemas")
            print("  7. View missing sessions across 2026 calendar")
            print("  0. Back to Main Menu\n")

            c = prompt_choice("Select option", ["1", "2", "3", "4", "5", "6", "7", "0"], default="1")
            if c == "0":
                break
            elif c == "1":
                self._inspect_data_inventory()
            elif c == "2":
                self._run_full_season_collector()
            elif c == "3":
                self._run_season_collector()
            elif c == "4":
                self._run_track_collector()
            elif c == "5":
                self._run_historical_circuit_collector()
            elif c == "6":
                self._audit_data_integrity()
            elif c == "7":
                self._view_calendar_missing_sessions()

    def _prompt_session_collection_mode(self) -> str:
        """Prompts user to select which sessions to collect."""
        print(f"\n{Colors.BOLD}Select Sessions to Collect:{Colors.RESET}")
        print("  1. Full Weekend (FP1, FP2, FP3, Qualifying, Race / Sprint)")
        print("  2. Race Prediction Sessions Only (Qualifying + Race / Sprint — faster)")
        print("  3. Practice Sessions Only (FP1, FP2, FP3)")
        c = prompt_choice("Select session mode", ["1", "2", "3"], default="1")
        if c == "1":
            return "full"
        elif c == "2":
            return "prediction_only"
        elif c == "3":
            return "practice_only"
        return "full"

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

    def _run_full_season_collector(self) -> None:
        print_header("Collect Entire Season (FastF1)")
        year = prompt_int("Enter Season Year", default=2026)
        session_mode = self._prompt_session_collection_mode()
        force = prompt_confirm("Force redownload of existing session files?", default=False)
        mode_label = "Full Weekend (Practice, Quali & Race)" if session_mode == "full" else (
            "Race Prediction (Quali & Race)" if session_mode == "prediction_only" else "Practice Sessions Only"
        )
        print_info(f"Initiating full season collection for {year} [{mode_label}]...")
        try:
            from src.data.collect_season import collect_season_data
            collect_season_data(year=year, rounds=None, force=force, session_mode=session_mode)
            print_success(f"Full season data collection for {year} completed.")
        except Exception as e:
            print_error(f"Full season collection failed: {e}")
        prompt_str("Press [Enter] to continue", allow_empty=True)

    def _run_track_collector(self) -> None:
        print_header("Collect Circuit Weekend Sessions (FastF1)")
        t_key = prompt_str("Enter circuit key (e.g. austin, sepang, baku)", default="austin")
        if t_key not in self.tracks_cfg:
            print_error(f"Track '{t_key}' not in configuration.")
            prompt_str("Press [Enter] to return", allow_empty=True)
            return

        year = prompt_int("Enter Year", default=2026)
        session_mode = self._prompt_session_collection_mode()
        force = prompt_confirm("Force refresh cache?", default=False)
        mode_label = "Full Weekend (Practice, Quali & Race)" if session_mode == "full" else (
            "Race Prediction (Quali & Race)" if session_mode == "prediction_only" else "Practice Sessions Only"
        )

        print_info(f"Running collection for {t_key.title()} {year} [{mode_label}]...")
        try:
            from src.data.collect_season import collect_grand_prix
            collect_grand_prix(t_key, year=year, force=force, session_mode=session_mode)
            print_success(f"Track collection completed for {t_key.title()}.")
        except Exception as e:
            print_error(f"Collection failed: {e}")
        prompt_str("Press [Enter] to continue", allow_empty=True)

    def _run_season_collector(self) -> None:
        print_header("Collect Season Rounds (FastF1)")
        year = prompt_int("Enter Year", default=2026)
        rounds_str = prompt_str("Enter round numbers separated by space (or 'all' for entire season)", default="all")
        if rounds_str.strip().lower() in ("all", "*", ""):
            r_list = None
        else:
            try:
                r_list = [int(x) for x in rounds_str.split()]
            except ValueError:
                print_error("Invalid round numbers.")
                prompt_str("Press [Enter] to return", allow_empty=True)
                return

        session_mode = self._prompt_session_collection_mode()
        force = prompt_confirm("Force redownload?", default=False)
        target_desc = f"rounds {r_list}" if r_list else f"entire season {year}"
        mode_label = "Full Weekend (Practice, Quali & Race)" if session_mode == "full" else (
            "Race Prediction (Quali & Race)" if session_mode == "prediction_only" else "Practice Sessions Only"
        )
        print_info(f"Collecting {target_desc} for season {year} [{mode_label}]...")
        try:
            from src.data.collect_season import collect_season_data
            collect_season_data(year=year, rounds=r_list, force=force, session_mode=session_mode)
            print_success("Season data collection completed.")
        except Exception as e:
            print_error(f"Season collection failed: {e}")
        prompt_str("Press [Enter] to continue", allow_empty=True)

    def _run_season_or_historical_collector(self) -> None:
        """Submenu for choosing between full season, single rounds, or historical circuit collection."""
        while True:
            clear_screen()
            print_banner(season=2026)
            print_header("Collect Race & Qualifying Data (FastF1)")
            print("  1. Collect data for an ENTIRE season (all championship rounds)")
            print("  2. Collect data for specific season rounds (specify round numbers)")
            print("  3. Collect historical circuit data (configured tracks & year ranges)")
            print("  0. Back to Data Collection Menu\n")

            sub_choice = prompt_choice("Select collection mode", ["1", "2", "3", "0"], default="1")
            if sub_choice == "0":
                break
            elif sub_choice == "1":
                self._run_full_season_collector()
                break
            elif sub_choice == "2":
                self._run_season_collector()
                break
            elif sub_choice == "3":
                self._run_historical_circuit_collector()
                break

    def _run_historical_circuit_collector(self) -> None:
        from src.data.collect_track import format_year_range, collect_historical_track_data

        clear_screen()
        print_banner(season=2026)
        print_header("HISTORICAL CIRCUIT DATA COLLECTION")

        # Read circuits dynamically from config
        circuits = []
        for k, info in self.tracks_cfg.items():
            d_name = info.get("display_name", k.title())
            hist_years = info.get("historical_years", [])
            hist_avail = info.get("historical_data_available", False)
            circuits.append((k, d_name, hist_years, hist_avail))

        # Sort alphabetically by display name for clear presentation
        circuits.sort(key=lambda x: x[1])

        print(f"{Colors.BOLD}Configured Circuits & Historical Coverage:{Colors.RESET}\n")
        circuit_map = {}
        for idx, (k, d_name, h_years, h_avail) in enumerate(circuits, start=1):
            year_range_str = format_year_range(h_years)
            if not h_avail and h_years:
                coverage_str = f"{year_range_str} (unavailable / disabled)"
            elif not h_years:
                coverage_str = "None configured"
            else:
                coverage_str = year_range_str
            print(f"  {idx:>2}. {d_name:<28} — configured years: {coverage_str}")
            circuit_map[str(idx)] = (k, d_name)

        print("\n  0. Return to menu\n")

        selection_str = prompt_str(
            "Select circuit numbers separated by space (e.g. '1 3 5', or 'all'), or enter 0 to return",
            default="0"
        )
        if selection_str.strip() == "0":
            return

        # Parse selections
        selected_circuits = []
        if selection_str.strip().lower() == "all":
            selected_circuits = [(k, d_name) for (k, d_name, _, _) in circuits]
        else:
            tokens = selection_str.replace(",", " ").split()
            invalid_tokens = []
            seen = set()
            for t in tokens:
                if t in circuit_map:
                    k, d_name = circuit_map[t]
                    if k not in seen:
                        seen.add(k)
                        selected_circuits.append((k, d_name))
                elif t.lower() in self.tracks_cfg:
                    k = t.lower()
                    d_name = self.tracks_cfg[k].get("display_name", k.title())
                    if k not in seen:
                        seen.add(k)
                        selected_circuits.append((k, d_name))
                else:
                    invalid_tokens.append(t)

            if invalid_tokens:
                print_warning(f"Ignored unrecognized selection(s): {', '.join(invalid_tokens)}")

        if not selected_circuits:
            print_error("No valid circuits selected.")
            prompt_str("Press [Enter] to return", allow_empty=True)
            return

        print(f"\n{Colors.BOLD}Selected Circuits ({len(selected_circuits)}):{Colors.RESET}")
        for k, d_name in selected_circuits:
            print(f"  - {d_name} ({k})")

        force = prompt_confirm("\nForce re-download existing files?", default=False)
        print_info(f"\nStarting historical collection for {len(selected_circuits)} circuit(s)...")

        all_results = []
        for k, d_name in selected_circuits:
            print(f"\n{Colors.CYAN}--- Processing {d_name} ({k}) ---{Colors.RESET}")
            try:
                res = collect_historical_track_data(
                    track_id=k,
                    force=force,
                    tracks_config=self.tracks_cfg
                )
                all_results.append(res)
            except Exception as e:
                print_error(f"Failed processing {d_name}: {e}")
                all_results.append({
                    "track": k,
                    "circuit_name": d_name,
                    "successes": [],
                    "skips": [],
                    "failures": [str(e)],
                })

        # Summary report
        print("\n" + "=" * 76)
        print(Colors.BOLD + Colors.CYAN + " HISTORICAL DATA COLLECTION SUMMARY REPORT" + Colors.RESET)
        print("=" * 76)

        total_succ = 0
        total_skip = 0
        total_fail = 0

        for r in all_results:
            c_name = r["circuit_name"]
            c_key = r["track"]
            s_list = r["successes"]
            k_list = r["skips"]
            f_list = r["failures"]
            total_succ += len(s_list)
            total_skip += len(k_list)
            total_fail += len(f_list)

            status_color = Colors.GREEN if not f_list else Colors.YELLOW
            print(f"\n{status_color}{Colors.BOLD}Circuit: {c_name} ({c_key}){Colors.RESET}")
            if s_list:
                print(f"  {Colors.GREEN}Successes ({len(s_list)}):{Colors.RESET}")
                for s in s_list:
                    print(f"    + {s}")
            if k_list:
                print(f"  {Colors.DIM}Skips ({len(k_list)}):{Colors.RESET}")
                for sk in k_list:
                    print(f"    - {sk}")
            if f_list:
                print(f"  {Colors.RED}Failures ({len(f_list)}):{Colors.RESET}")
                for fl in f_list:
                    print(f"    ! {fl}")

        print("\n" + "=" * 76)
        print(
            f"Overall: {total_succ} sessions saved, {total_skip} skipped, {total_fail} failed "
            f"across {len(all_results)} circuit(s)."
        )
        print("=" * 76 + "\n")
        prompt_str("Press [Enter] to continue", allow_empty=True)

    def _audit_data_integrity(self) -> None:
        print_header("Dataset Integrity & Anti-Leakage Audit")
        def_k, def_r = self._get_default_circuit_for_season(2026)
        def_k = def_k or "sepang"
        t_key = prompt_str("Enter circuit key to audit against", default=def_k)
        if t_key not in self.tracks_cfg:
            print_error(f"Unknown circuit '{t_key}'.")
            prompt_str("Press [Enter] to return", allow_empty=True)
            return

        pipeline = self._get_pipeline()
        r_num = self.tracks_cfg[t_key].get("round_2026", def_r or 16)
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
            penalties=self.tracks_cfg[t_key].get("known_penalties_2026", {}),
            track_key=t_key,
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
                r_num = info.get(f"round_{year}", info.get("round_2026"))
                if r_num is not None:
                    from src.data.collect_season import collect_season_data
                    collect_season_data(year=year, rounds=[r_num])
                    actual_df = self.loader.load_target_race(track_folder, year=year)
                else:
                    print_error(f"Round number not configured for {track_folder} in {year}.")

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
            grid_raw = r.get("ActualStartingGrid")
            raw_pit = r.get("is_pit_lane_start", 0)
            is_pit = pd.notna(raw_pit) and int(round(float(raw_pit))) == 1
            if is_pit:
                grid = "PL"
            elif pd.notna(grid_raw):
                grid = str(int(round(float(grid_raw))))
            else:
                grid = "-"
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

        print(f"\n{Colors.BOLD}Five Major ML Capabilities (Modularly Supported):{Colors.RESET}")
        print(f"  • Race Pace Intelligence       : {Colors.GREEN}Available{Colors.RESET} (FP1/FP2/FP3 stint extraction, fuel burn adj)")
        print(f"  • Tire Degradation Modelling   : {Colors.GREEN}Available{Colors.RESET} (Compound slopes, teammate pooling, thermal scale)")
        print(f"  • Pit Stop Strategy Simulator  : {Colors.GREEN}Available{Colors.RESET} (1/2-stop optimization & FIA 2-dry-compound rule)")
        print(f"  • Race Control Incident Model  : {Colors.GREEN}Available{Colors.RESET} (SC/VSC field compression & cheap pit stops)")
        print(f"  • Weather-Aware Predictions    : {Colors.GREEN}Available{Colors.RESET} (Session telemetry, rain variance & wet driver skill)")

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

    # =========================================================================
    # OPTION 7: DATA MANAGEMENT & DELETION
    # =========================================================================
    def menu_data_management(self) -> None:
        """Interactive controller for safe data management and selective deletion."""
        while True:
            clear_screen()
            print_banner(season=2026, status_text="Data Management")
            print(f"{Colors.BOLD}DATA MANAGEMENT & DELETION{Colors.RESET}")
            print(f"{Colors.DIM}Safely manage and selectively delete project-generated data.{Colors.RESET}")
            print(f"{Colors.DIM}Source code, configurations, tests, docs, and audit logs are permanently protected.{Colors.RESET}\n")

            print("  1. Delete Cached Data (FastF1 cache, session cache, temporary files)")
            print("  2. Delete Collected Datasets (Categorized: practice, qualifying, race, grids)")
            print("  3. Delete Prediction Outputs (Generated reports, Monte Carlo CSVs, charts)")
            print("  4. Delete Grand Prix Data (Isolate by season and circuit / round)")
            print("  5. Delete All Project-Generated Data (Comprehensive cleanup with safety guards)")
            print("  6. View Deletion Audit Log (Review past deletion operations)")
            print("\n  0. Back to Main Menu\n")

            choice = prompt_choice("Select an operation", ["1", "2", "3", "4", "5", "6", "0"], default="0")

            if choice == "0":
                break
            elif choice == "1":
                self._handle_delete_cached_data()
            elif choice == "2":
                self._handle_delete_collected_datasets()
            elif choice == "3":
                self._handle_delete_prediction_outputs()
            elif choice == "4":
                self._handle_delete_grand_prix_data()
            elif choice == "5":
                self._handle_delete_all_generated_data()
            elif choice == "6":
                self._view_deletion_audit_log()

    def _render_deletion_preview(self, plan: Dict[str, Any]) -> None:
        """Renders formatted preview table and summary of files targeted for deletion."""
        print_header(f"Deletion Plan Preview: {plan.get('operation', 'Operation')}")
        file_count = plan.get("file_count", 0)
        storage_str = plan.get("storage_formatted", "0 B")
        categories = plan.get("categories", {})

        print(f"{Colors.BOLD}Summary of Targeted Artifacts:{Colors.RESET}")
        print(f"  * Total files to delete  : {Colors.CYAN}{file_count}{Colors.RESET}")
        print(f"  * Estimated space reclaim: {Colors.CYAN}{storage_str}{Colors.RESET}")
        if categories:
            print(f"\n{Colors.BOLD}Breakdown by Category:{Colors.RESET}")
            for cat, count in categories.items():
                print(f"  - {cat:<26}: {count} files")

        if plan.get("excluded_count", 0) > 0:
            print(f"\n{Colors.DIM}Protected / Excluded files: {plan.get('excluded_count')} items safely preserved.{Colors.RESET}")

        target_paths = plan.get("target_paths", [])
        if target_paths:
            print(f"\n{Colors.BOLD}Target Files (showing up to 15):{Colors.RESET}")
            for p in target_paths[:15]:
                print(f"  {Colors.DIM}* {p}{Colors.RESET}")
            if len(target_paths) > 15:
                print(f"  {Colors.DIM}... and {len(target_paths) - 15} more files.{Colors.RESET}")
        else:
            print(f"\n{Colors.YELLOW}No eligible files found for this operation.{Colors.RESET}")

        print(f"\n{Colors.GREEN}[PROTECTION GUARANTEE]{Colors.RESET} Source code, configs, tests, docs, .git, and audit logs are protected.")
        if not plan.get("historical_included", False):
            print(f"{Colors.GREEN}[HISTORICAL SAFETY]{Colors.RESET} Historical training data (seasons prior to 2026) is preserved.\n")

    def _render_deletion_summary(self, result: Dict[str, Any]) -> None:
        """Renders deletion result summary."""
        status = result.get("status", "UNKNOWN")
        if status == "COMPLETED":
            print_success("Operation completed successfully!")
            print(f"  * Files deleted    : {result.get('files_deleted_count', 0)}")
            print(f"  * Storage reclaimed: {result.get('storage_reclaimed_formatted', '0 B')}")
        elif status == "PARTIAL_FAILURE":
            print_warning("Operation completed with partial failures.")
            print(f"  * Files deleted    : {result.get('files_deleted_count', 0)}")
            print(f"  * Failed files     : {result.get('failed_count', 0)}")
            print(f"  * Storage reclaimed: {result.get('storage_reclaimed_formatted', '0 B')}")
            for err in result.get("failed_deletions", [])[:5]:
                print_error(f"  Failed: {err.get('path')} - {err.get('error')}")
        elif status == "CANCELLED":
            print_warning("Operation cancelled by user. No files were deleted.")
        elif status == "REJECTED":
            print_error(f"Operation rejected: {result.get('message')}")
        else:
            print_error(f"Deletion failed: {result.get('message', 'Unknown error')}")

        print(f"\n{Colors.DIM}Persistent audit record appended to: logs/deletion_audit.jsonl{Colors.RESET}\n")
        prompt_str("Press [Enter] to continue", allow_empty=True)

    def _handle_delete_cached_data(self) -> None:
        """Handles deletion of cached telemetry, session data, and temporary files."""
        clear_screen()
        print_banner(season=2026, status_text="Cache Management")
        print(f"{Colors.BOLD}Delete Cached Data{Colors.RESET}\n")
        print("  1. Delete all cached data across all seasons")
        print("  2. Delete cached data for a specific season")
        print("\n  0. Cancel / Back\n")

        scope = prompt_choice("Select scope", ["1", "2", "0"], default="1")
        if scope == "0":
            return

        year_filter = None
        if scope == "2":
            year_filter = prompt_int("Enter season year to clean cache for", min_val=2015, max_val=2030, default=2026)

        cache_files = self.data_manager.discover_cache_data(year=year_filter)
        op_label = f"Delete Cached Data ({year_filter if year_filter else 'All Seasons'})"
        plan = self.data_manager.build_deletion_preview(cache_files, operation_name=op_label, operation_type="cache")

        self._render_deletion_preview(plan)
        if plan.get("file_count", 0) == 0:
            prompt_str("Press [Enter] to return", allow_empty=True)
            return

        if prompt_confirm("Are you sure you want to delete these cached files?", default=False):
            result = self.data_manager.execute_deletion_plan(plan, confirmed=True)
            self._render_deletion_summary(result)
        else:
            result = self.data_manager.execute_deletion_plan(plan, confirmed=False)
            self._render_deletion_summary(result)

    def _handle_delete_collected_datasets(self) -> None:
        """Handles deletion of collected session CSVs with category filtering and historical safety."""
        clear_screen()
        print_banner(season=2026, status_text="Dataset Management")
        print(f"{Colors.BOLD}Delete Collected Datasets{Colors.RESET}\n")
        print("  1. All Collected Datasets")
        print("  2. Practice Sessions Only (FP1, FP2, FP3)")
        print("  3. Qualifying Sessions Only (Qualifying, Sprint Shootout)")
        print("  4. Race Sessions Only (Grand Prix, Sprint Race)")
        print("  5. Starting Grids Only (Cached OpenF1 Grids)")
        print("\n  0. Cancel / Back\n")

        cat_choice = prompt_choice("Select dataset category", ["1", "2", "3", "4", "5", "0"], default="1")
        if cat_choice == "0":
            return

        categories_map = {
            "1": ["practice", "qualifying", "race", "grids"],
            "2": ["practice"],
            "3": ["qualifying"],
            "4": ["race"],
            "5": ["grids"],
        }
        selected_cats = categories_map.get(cat_choice, ["practice", "qualifying", "race", "grids"])
        cat_names_map = {
            "1": "All Categories",
            "2": "Practice Sessions",
            "3": "Qualifying Sessions",
            "4": "Race Sessions",
            "5": "Starting Grids",
        }

        # Year filter
        year_str = prompt_str("Enter championship year to target [default: 2026, or 'all']: ", default="2026").strip()
        year_filter = None if year_str.lower() in ("all", "*", "") else int(year_str)

        # Historical protection opt-in
        include_historical = False
        print(f"\n{Colors.DIM}Notice: Historical training datasets (prior to 2026) are protected by default.{Colors.RESET}")
        include_historical = prompt_confirm("Include historical training data in deletion?", default=False)

        cat_dict = self.data_manager.discover_collected_datasets(
            categories=selected_cats,
            include_historical=include_historical,
            year=year_filter,
        )
        flat_files = [f for flist in cat_dict.values() for f in flist]
        op_label = f"Delete Collected Datasets ({cat_names_map.get(cat_choice, 'Selected')})"
        plan = self.data_manager.build_deletion_preview(
            flat_files,
            operation_name=op_label,
            include_historical=include_historical,
            operation_type="collected"
        )

        self._render_deletion_preview(plan)
        if plan.get("file_count", 0) == 0:
            prompt_str("Press [Enter] to return", allow_empty=True)
            return

        confirmation_phrase = None
        if include_historical:
            print(f"{Colors.YELLOW}[SECURITY REQUIREMENT]{Colors.RESET} You have opted to include historical training data.")
            typed = prompt_str("Type 'DELETE HISTORICAL DATA' to proceed (or press Enter to cancel): ", allow_empty=True)
            confirmation_phrase = typed.strip()
            confirmed = (confirmation_phrase == "DELETE HISTORICAL DATA")
        else:
            confirmed = prompt_confirm("Are you sure you want to delete these datasets?", default=False)

        result = self.data_manager.execute_deletion_plan(
            plan,
            confirmed=confirmed,
            confirmation_phrase=confirmation_phrase
        )
        self._render_deletion_summary(result)

    def _handle_delete_prediction_outputs(self) -> None:
        """Handles deletion of generated prediction outputs and simulation artifacts."""
        clear_screen()
        print_banner(season=2026, status_text="Prediction Management")
        print(f"{Colors.BOLD}Delete Prediction Outputs{Colors.RESET}\n")
        print("  1. Delete all circuit prediction outputs")
        print("  2. Delete prediction outputs for a specific circuit")
        print("\n  0. Cancel / Back\n")

        scope = prompt_choice("Select scope", ["1", "2", "0"], default="1")
        if scope == "0":
            return

        target_track = None
        if scope == "2":
            # List available track folders in outputs
            out_root = self.root_dir / "outputs"
            available = [d.name for d in out_root.iterdir() if d.is_dir()] if out_root.exists() else []
            if not available:
                print_info("No track output directories found in outputs/.")
                prompt_str("Press [Enter] to return", allow_empty=True)
                return

            print(f"\n{Colors.BOLD}Available Output Circuits:{Colors.RESET}")
            for idx, tr in enumerate(sorted(available), 1):
                name = self.tracks_cfg.get(tr, {}).get("display_name", tr.title())
                print(f"  {idx:>2}. {name} ({tr})")
            print("\n   0. Cancel\n")

            track_c = prompt_choice("Select circuit", [str(i) for i in range(1, len(available) + 1)] + ["0"], default="1")
            if track_c == "0":
                return
            target_track = sorted(available)[int(track_c) - 1]

        out_files = self.data_manager.discover_prediction_outputs(track_key=target_track)
        op_label = f"Delete Prediction Outputs ({target_track.title() if target_track else 'All Circuits'})"
        plan = self.data_manager.build_deletion_preview(out_files, operation_name=op_label, operation_type="outputs")

        self._render_deletion_preview(plan)
        if plan.get("file_count", 0) == 0:
            prompt_str("Press [Enter] to return", allow_empty=True)
            return

        if prompt_confirm("Are you sure you want to delete these prediction outputs?", default=False):
            result = self.data_manager.execute_deletion_plan(plan, confirmed=True)
            self._render_deletion_summary(result)
        else:
            result = self.data_manager.execute_deletion_plan(plan, confirmed=False)
            self._render_deletion_summary(result)

    def _handle_delete_grand_prix_data(self) -> None:
        """Handles targeted deletion of all data associated with a single Grand Prix event."""
        clear_screen()
        print_banner(season=2026, status_text="Grand Prix Cleanup")
        print(f"{Colors.BOLD}Delete Grand Prix Event Data{Colors.RESET}")
        print(f"{Colors.DIM}Isolates and removes cache, raw sessions, and outputs for a single Grand Prix.{Colors.RESET}\n")

        year = prompt_int("Enter championship season year", min_val=2015, max_val=2030, default=2026)

        # List circuits
        circuits = []
        for k, info in sorted(self.tracks_cfg.items(), key=lambda x: x[1].get(f"round_{year}" if year != 2026 else "round_2026", 99)):
            r = info.get(f"round_{year}" if year != 2026 else "round_2026")
            name = info.get("display_name", k.title())
            circuits.append((k, name, r))

        print(f"\n{Colors.BOLD}Configured Circuits for {year}:{Colors.RESET}")
        for idx, (k, name, r) in enumerate(circuits, 1):
            r_str = f"Round {r}" if r is not None else "No round"
            print(f"  {idx:>2}. {name:<32} ({r_str}, key: {k})")
        print("\n   0. Cancel\n")

        c = prompt_choice("Select Grand Prix", [str(i) for i in range(1, len(circuits) + 1)] + ["0"], default="1")
        if c == "0":
            return

        sel_k, sel_name, sel_r = circuits[int(c) - 1]

        include_historical = False
        if year < 2026:
            print(f"\n{Colors.YELLOW}[HISTORICAL NOTICE]{Colors.RESET} Year {year} is prior to current season (2026).")
            include_historical = prompt_confirm(f"Include historical event data for {sel_name} {year}?", default=False)
            if not include_historical:
                print_info("Operation cancelled. Historical data was not opted-in.")
                prompt_str("Press [Enter] to return", allow_empty=True)
                return

        gp_dict = self.data_manager.discover_grand_prix_data(
            track=sel_k,
            year=year,
            round_num=sel_r,
            include_historical=include_historical,
        )
        flat_files = [f for flist in gp_dict.values() for f in flist]
        op_label = f"Delete Grand Prix Data ({sel_name} {year})"
        plan = self.data_manager.build_deletion_preview(
            flat_files,
            operation_name=op_label,
            include_historical=include_historical,
            extra_metadata={"track": sel_k, "year": year, "round": sel_r},
            operation_type="grand_prix"
        )

        self._render_deletion_preview(plan)
        if plan.get("file_count", 0) == 0:
            prompt_str("Press [Enter] to return", allow_empty=True)
            return

        if prompt_confirm(f"Delete all {sel_name} {year} generated data?", default=False):
            result = self.data_manager.execute_deletion_plan(plan, confirmed=True)
            self._render_deletion_summary(result)
        else:
            result = self.data_manager.execute_deletion_plan(plan, confirmed=False)
            self._render_deletion_summary(result)

    def _handle_delete_all_generated_data(self) -> None:
        """Handles comprehensive deletion of all project-generated data with typed phrase confirmation."""
        clear_screen()
        print_banner(season=2026, status_text="DANGER ZONE")
        print(f"{Colors.RED}{Colors.BOLD}WARNING: DELETE ALL PROJECT-GENERATED DATA{Colors.RESET}")
        print(f"{Colors.DIM}This operation identifies all generated caches, collected datasets, and prediction outputs.{Colors.RESET}")
        print(f"{Colors.DIM}Source code, configurations, tests, docs, .git, and audit logs are PERMANENTLY PROTECTED.{Colors.RESET}\n")

        include_historical = prompt_confirm(
            "Include historical training datasets (prior to 2026)?",
            default=False
        )

        all_dict = self.data_manager.discover_all_project_generated_data(include_historical=include_historical)
        flat_files = [f for flist in all_dict.values() for f in flist]
        plan = self.data_manager.build_deletion_preview(
            flat_files,
            operation_name="Delete all project-generated data",
            include_historical=include_historical,
            operation_type="all_generated_data"
        )

        self._render_deletion_preview(plan)
        if plan.get("file_count", 0) == 0:
            prompt_str("Press [Enter] to return", allow_empty=True)
            return

        print(f"\n{Colors.RED}{Colors.BOLD}MANDATORY CONFIRMATION:{Colors.RESET}")
        print("To proceed with this irreversible operation, you must type exactly:")
        print(f"  {Colors.BOLD}DELETE ALL GENERATED DATA{Colors.RESET}\n")

        phrase = prompt_str("Confirmation phrase (or press Enter to cancel): ", allow_empty=True).strip()
        if phrase == "DELETE ALL GENERATED DATA":
            result = self.data_manager.execute_deletion_plan(
                plan,
                confirmed=True,
                confirmation_phrase=phrase
            )
            self._render_deletion_summary(result)
        else:
            print_warning("Confirmation phrase did not match. Operation cancelled.")
            result = self.data_manager.execute_deletion_plan(plan, confirmed=False)
            self._render_deletion_summary(result)

    def _view_deletion_audit_log(self) -> None:
        """Displays persistent deletion audit records from logs/deletion_audit.jsonl."""
        clear_screen()
        print_banner(season=2026, status_text="Audit Trail")
        print(f"{Colors.BOLD}Deletion Audit Trail{Colors.RESET}")
        print(f"{Colors.DIM}Immutable record of data management operations (logs/deletion_audit.jsonl){Colors.RESET}\n")

        records = self.data_manager.audit_logger.get_audit_history(limit=25)
        if not records:
            print_info("No deletion operations logged yet.")
            prompt_str("\nPress [Enter] to return to Data Management Menu", allow_empty=True)
            return

        rows = []
        for r in reversed(records):
            ts = r.get("timestamp", "-")[:19].replace("T", " ")
            op = r.get("operation_type", "-")
            status = r.get("status", "-")
            color_status = Colors.GREEN if status == "COMPLETED" else (Colors.YELLOW if status == "CANCELLED" else Colors.RED)
            del_count = str(r.get("files_deleted_count", len(r.get("files_deleted", []))))
            bytes_val = r.get("storage_reclaimed_bytes", 0)
            reclaimed = DataManager._format_size(bytes_val)
            hist = "Yes" if r.get("historical_data_included") else "No"
            rows.append([ts, op[:32], f"{color_status}{status}{Colors.RESET}", del_count, reclaimed, hist])

        print_table(["Timestamp (UTC)", "Operation", "Status", "Deleted", "Reclaimed", "Historical"], rows)
        prompt_str("\nPress [Enter] to return to Data Management Menu", allow_empty=True)
