"""
ApexGrid — Standalone Motorsport Intelligence CLI Application
=============================================================
Fast, modern, app-like terminal interface powered by Typer and Rich.
Tagline: Predicting the Race Beyond the Grid.
"""

import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
import typer

# Root Directory Anchoring — canonical project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Color Palette Constants (FIA Motorsport Aesthetic)
APEX_RED = "#E10600"
TELEMETRY_CYAN = "#00B4D8"
TIMING_PURPLE = "#9D4EDD"
GREEN_FLAG = "#00D26A"
FLAG_YELLOW = "#FFCC00"
TITANIUM_WHITE = "#F0F0F2"
MUTED_GREY = "#8A8A93"

console = Console()

app = typer.Typer(
    name="apexgrid",
    help="ApexGrid — Predicting the Race Beyond the Grid (Formula 1 Motorsport Intelligence & Inference Engine)",
    no_args_is_help=False,
    rich_markup_mode="rich",
)

history_app = typer.Typer(
    name="history",
    help="Explore verified historical Formula 1 statistics and analytics across 1950–2026",
    no_args_is_help=False,
    rich_markup_mode="rich",
)
app.add_typer(history_app, name="history")


# ==============================================================================
# Helper Functions
# ==============================================================================

def print_error(message: str, exc: Optional[Exception] = None, debug: bool = False) -> None:
    """Displays formatted motorsport error alert."""
    console.print(f"[bold {APEX_RED}]Error:[/] [white]{message}[/]")
    if debug and exc is not None:
        console.print(f"\n[dim]{traceback.format_exc()}[/dim]")


def render_probability_bar(prob_pct: float, width: int = 10) -> str:
    """Renders visual horizontal ASCII probability bar."""
    pct = max(0.0, min(100.0, float(prob_pct or 0.0)))
    filled = int(round((pct / 100.0) * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar} {pct:4.1f}%"


# ==============================================================================
# Root Landing Callback
# ==============================================================================

@app.callback(invoke_without_command=True)
def main_landing(
    ctx: typer.Context,
    menu: bool = typer.Option(False, "--menu", "-m", help="Launch classic interactive terminal menu"),
    classic: bool = typer.Option(False, "--classic", help="Launch classic text-based interactive menu"),
) -> None:
    """Master ApexGrid CLI entrypoint."""
    if menu or classic or ctx.invoked_subcommand is None:
        from src.cli.menu import InteractiveMenu
        InteractiveMenu(root_dir=PROJECT_ROOT).run()
        raise typer.Exit()


# ==============================================================================
# Subcommand: predict
# ==============================================================================

@app.command("predict", help="Execute end-to-end race prediction pipeline with 25,000 Monte Carlo draws")
def cmd_predict(
    track: str = typer.Option(..., "--track", "-t", help="Circuit identifier (e.g. melbourne, sepang, monza, baku)"),
    year: int = typer.Option(2026, "--year", "-y", help="Championship season year"),
    round_num: Optional[int] = typer.Option(None, "--round", "-r", help="Championship round number"),
    stage: str = typer.Option("after_qualifying", "--stage", "-s", help="Prediction timeline stage cutoff"),
    model: str = typer.Option("hybrid", "--model", "-m", help="Prediction model: 'rf', 'xgb', 'ridge', 'hybrid', 'ensemble'/'multi', 'weekend_only'"),
    rf_weight: Optional[float] = typer.Option(None, "--rf-weight", help="Custom Random Forest weight percentage (0-100)"),
    xgb_weight: Optional[float] = typer.Option(None, "--xgb-weight", help="Custom XGBoost weight percentage (0-100)"),
    enhanced: bool = typer.Option(False, "--enhanced", help="Enable 5 ML Enhancements (Pace, Tires, Pit Strategy, Safety Car, Weather)"),
    weekend_only: bool = typer.Option(False, "--weekend-only", help="Predict strictly from active weekend telemetry without prior season history"),
    top: int = typer.Option(10, "--top", help="Number of finishing positions to display"),
    simulations: Optional[int] = typer.Option(None, "--simulations", help="Monte Carlo iterations (e.g. 10000 default, 25000 publication)"),
    debug: bool = typer.Option(False, "--debug", help="Display diagnostic traceback on execution error"),
) -> None:
    """Executes prediction simulation and displays classification table."""
    from src.pipeline.race_prediction_pipeline import RacePredictionPipeline

    model_clean = model.strip().lower()
    if weekend_only:
        model_clean = "weekend_only"
    elif model_clean not in ("rf", "xgb", "ridge", "hybrid", "ensemble", "multi", "weekend_only"):
        print_error(f"Invalid model architecture: '{model}'. Choose from 'rf', 'xgb', 'ridge', 'hybrid', 'ensemble', 'multi', 'weekend_only'.")
        raise typer.Exit(code=2)

    # Validate custom weights
    if (rf_weight is None) != (xgb_weight is None):
        print_error("Both --rf-weight and --xgb-weight must be supplied together for custom weights.")
        raise typer.Exit(code=2)

    if rf_weight is not None and xgb_weight is not None:
        if not (0.0 <= rf_weight <= 100.0 and 0.0 <= xgb_weight <= 100.0):
            print_error("Weights must each be between 0 and 100.")
            raise typer.Exit(code=2)
        if abs((rf_weight + xgb_weight) - 100.0) > 1e-4:
            print_error(f"Weight sum must equal 100% (got {rf_weight} + {xgb_weight} = {rf_weight + xgb_weight}%).")
            raise typer.Exit(code=2)
        if model_clean in ("rf", "xgb", "ridge", "ensemble", "multi"):
            print_error(f"Custom weights cannot be used with '{model_clean}' model (use with 'hybrid').")
            raise typer.Exit(code=2)

    caps = None
    if enhanced:
        caps = {
            "race_pace_intelligence": True,
            "tire_degradation_modeling": True,
            "pit_strategy_simulator": True,
            "race_control_model": True,
            "weather_aware_prediction": True,
        }

    stage_clean = stage.lower().strip()
    if stage_clean == "after_sprint":
        stage_clean = "after_sprint_race"

    pipeline = RacePredictionPipeline(root_dir=PROJECT_ROOT)
    
    with console.status(f"[bold {TELEMETRY_CYAN}]Executing 25,000 Monte Carlo draws for {track.title()} {year} [{stage_clean}]...[/]"):
        try:
            results = pipeline.run(
                track_key=track,
                year=year,
                round_num=round_num,
                stage=stage_clean,
                model=model_clean,
                rf_weight=rf_weight,
                xgb_weight=xgb_weight,
                capabilities=caps,
                n_simulations=simulations,
                weekend_only=weekend_only,
            )
        except Exception as e:
            print_error(f"Prediction pipeline halted: {e}", exc=e, debug=debug)
            raise typer.Exit(code=1)

    df: pd.DataFrame = results["predictions"]
    track_name = results["track_name"].upper()
    top_df = df.iloc[:top]

    # Starting grid provenance
    grid_meta = results.get("grid_meta", {})
    from src.features.grid_features import format_grid_provenance_display
    if results.get("is_provisional_quali"):
        grid_source = "Provisional Qualifying Model"
    elif grid_meta:
        grid_source = format_grid_provenance_display(grid_meta)
    else:
        grid_source = "Actual Observed Qualifying Session"

    mae_val = results.get('selected_mae')
    mae_str = f"{mae_val:.4f}" if isinstance(mae_val, (int, float)) else str(mae_val)

    # Top summary panel
    meta_text = (
        f"Circuit: [bold white]{track_name}[/] ({year}) | Stage: [bold white]{stage_clean.upper()}[/]\n"
        f"Model: [bold cyan]{results.get('selected_model')}[/] (MAE: [bold]{mae_str}[/]) | "
        f"Grid Source: [bold white]{grid_source}[/]"
    )
    console.print(Panel(meta_text, title=f"[bold {APEX_RED}]PREDICTION RESULTS[/]", border_style=APEX_RED))

    # Results Table
    table = Table(title=f"Grand Prix Predicted Classification (Top {top})", border_style="dim", header_style=f"bold {TITANIUM_WHITE}")
    table.add_column("Pos", justify="right", style="bold")
    table.add_column("Driver", style="bold")
    table.add_column("Team", style=MUTED_GREY)
    table.add_column("Grid", justify="right")
    table.add_column("ExpFinish", justify="right", style=TELEMETRY_CYAN)
    table.add_column("Win %", justify="right")
    table.add_column("Podium %", justify="right")
    table.add_column("Points %", justify="right")
    table.add_column("90% Range", justify="center", style=MUTED_GREY)

    for _, r in top_df.iterrows():
        pos_int = int(r.get("PredictedPosition", 0))
        if pos_int == 1:
            pos_style = f"bold {FLAG_YELLOW}"
        elif pos_int == 2:
            pos_style = f"bold {TITANIUM_WHITE}"
        elif pos_int == 3:
            pos_style = "bold #CD7F32"  # Bronze
        elif pos_int <= 10:
            pos_style = f"bold {GREEN_FLAG}"
        else:
            pos_style = MUTED_GREY

        grid_raw = r.get("ActualStartingGrid")
        prov_raw = r.get("provisional_grid_position")
        raw_pit = r.get("is_pit_lane_start", 0)
        from src.cli.menu import format_grid_display
        grid_str = format_grid_display(grid_raw, prov_raw, raw_pit)

        exp_str = f"{r.get('ExpectedFinish', 0.0):.2f}"
        win_str = render_probability_bar(r.get('WinProbability', 0.0), width=6)
        pod_str = f"{r.get('Top3Probability', 0.0):.1f}%"
        pts_str = f"{r.get('PointsProbability', 0.0):.1f}%"
        rng_str = f"P{int(r.get('P5', 1))}-P{int(r.get('P95', 20))}"

        table.add_row(
            f"[{pos_style}]P{pos_int:02d}[/]",
            f"[{pos_style}]{r.get('Abbreviation', '')}[/]",
            str(r.get("TeamName", ""))[:20],
            grid_str,
            exp_str,
            win_str,
            pod_str,
            pts_str,
            rng_str,
        )

    console.print(table)

    # Environmental & Strategy Panel
    if "RecommendedStrategy" in top_df.columns:
        sim_meta = results.get("simulation_meta", {})
        sc_p = sim_meta.get("p_safety_car", "N/A")
        rain_p = sim_meta.get("rain_probability", "N/A")
        strat_text = f"Safety Car Probability: [bold cyan]{sc_p}[/] | Rain Probability: [bold cyan]{rain_p}[/]\n"
        strat_table = Table(border_style="dim", header_style="bold cyan")
        strat_table.add_column("Driver")
        strat_table.add_column("Strategy")
        strat_table.add_column("Stops", justify="right")
        strat_table.add_column("Delta to Alt", justify="right")

        for _, r in top_df.iterrows():
            strat_table.add_row(
                r.get("Abbreviation", ""),
                str(r.get("RecommendedStrategy", "1-Stop")),
                str(int(r.get("OptimalStops", 1))),
                f"{float(r.get('StrategyDeltaSec', 0.0)):.1f}s",
            )
        console.print(Panel(strat_table, title=f"[bold {TELEMETRY_CYAN}]Strategic & Environmental Intelligence[/]", border_style=TELEMETRY_CYAN))

    out_dir = results.get("output_dir", "")
    if out_dir:
        console.print(f"\n[dim]Reports and full dataset written to: {out_dir}[/dim]\n")


# ==============================================================================
# Subcommand: evaluate
# ==============================================================================

@app.command("evaluate", help="Run chronological walk-forward cross-validation on completed season rounds")
def cmd_evaluate(
    track: str = typer.Option(..., "--track", "-t", help="Target circuit identifier"),
    year: int = typer.Option(2026, "--year", "-y", help="Championship season year"),
    round_num: Optional[int] = typer.Option(None, "--round", "-r", help="Championship round number"),
) -> None:
    """Executes model walk-forward evaluation and prints metrics scorecard."""
    from src.pipeline.race_prediction_pipeline import RacePredictionPipeline

    pipeline = RacePredictionPipeline(root_dir=PROJECT_ROOT)
    track_key = track.lower().strip()
    tracks = pipeline.tracks_config.get("tracks", {})
    if track_key not in tracks:
        print_error(f"Track '{track_key}' not found in tracks.json. Available: {list(tracks.keys())}")
        raise typer.Exit(code=1)

    track_info = tracks[track_key]
    target_round = round_num or track_info.get(f"round_{year}", track_info.get("round_2026"))
    if not target_round:
        print_error(f"Round not configured for circuit '{track_key}' in season {year}.")
        raise typer.Exit(code=1)

    train_rounds = list(range(1, target_round))
    if not train_rounds:
        print_error(f"Cannot run walk-forward evaluation for Round {target_round} (requires completed earlier rounds).")
        raise typer.Exit(code=1)

    with console.status(f"[bold {TELEMETRY_CYAN}]Evaluating models across completed {year} rounds {train_rounds}...[/]"):
        try:
            races_df, qualis_df = pipeline.data_loader.load_season_rounds(year, train_rounds)
            sessions_by_round = pipeline.data_loader.load_all_practice_sessions(year, train_rounds)
            training_features = pipeline.feature_engineer.build_training_dataset(
                races_df=races_df,
                qualis_df=qualis_df,
                practice_sessions_by_round=sessions_by_round,
                train_rounds=train_rounds,
            )
            eval_results = pipeline.model_evaluator.evaluate_walk_forward(training_features)
        except Exception as e:
            print_error(f"Evaluation failed: {e}", exc=e)
            raise typer.Exit(code=1)

    table = Table(
        title=f"Chronological Walk-Forward Model Evaluation (Rounds 4–{target_round-1})",
        border_style="dim",
        header_style=f"bold {TITANIUM_WHITE}",
    )
    table.add_column("Model Configuration", style="bold")
    table.add_column("MAE", justify="right", style=TELEMETRY_CYAN)
    table.add_column("MedAE", justify="right")
    table.add_column("RMSE", justify="right")
    table.add_column("Spearman Rank", justify="right", style=GREEN_FLAG)

    for row in eval_results["ensemble_comparisons"]:
        mae_str = f"{row['mae']:.3f}" if row.get("mae") is not None else "BLOCKED"
        medae_str = f"{row['medae']:.3f}" if row.get("medae") is not None else "N/A"
        rmse_str = f"{row['rmse']:.3f}" if row.get("rmse") is not None else "N/A"
        spearman_str = f"{row['spearman']:.3f}" if row.get("spearman") is not None else "N/A"
        table.add_row(
            row["model_name"],
            mae_str,
            medae_str,
            rmse_str,
            spearman_str,
        )

    console.print(table)
    console.print(f"[dim]Empirical DNF Hazard Rate: {eval_results['empirical_dnf_rate']*100:.1f}%[/dim]\n")


# ==============================================================================
# Subcommand: validate
# ==============================================================================

@app.command("validate", help="Audit datasets and feature pipeline for future leakage and data integrity")
def cmd_validate(
    track: str = typer.Option(..., "--track", "-t", help="Target circuit identifier"),
    year: int = typer.Option(2026, "--year", "-y", help="Championship season year"),
    stage: str = typer.Option("after_qualifying", "--stage", "-s", help="Prediction stage cutoff"),
) -> None:
    """Executes dataset integrity and future leakage verification."""
    from src.pipeline.race_prediction_pipeline import RacePredictionPipeline

    pipeline = RacePredictionPipeline(root_dir=PROJECT_ROOT)
    track_key = track.lower().strip()
    tracks = pipeline.tracks_config.get("tracks", {})
    if track_key not in tracks:
        print_error(f"Track '{track_key}' not found in tracks.json.")
        raise typer.Exit(code=1)

    track_info = tracks[track_key]
    target_round = track_info.get(f"round_{year}", track_info.get("round_2026", 1))

    with console.status(f"[bold {TELEMETRY_CYAN}]Auditing anti-leakage constraints for {track.title()} {year} [{stage}]...[/]"):
        try:
            train_rounds = list(range(1, target_round))
            races_df, qualis_df = pipeline.data_loader.load_season_rounds(year, train_rounds)
            sessions_by_round = pipeline.data_loader.load_all_practice_sessions(year, train_rounds)
            weekend_sessions = pipeline.data_loader.load_weekend_sessions(
                track_key, year, stage=stage, round_num=target_round
            )
            quali_df = pipeline.data_loader.load_target_qualifying(track_key, year, round_num=target_round)
            if quali_df is None or quali_df.empty:
                provisional_meta = pipeline.predict_qualifying(track_key=track_key, year=year, stage=stage, round_num=target_round)
                quali_df = provisional_meta["predictions"][["Abbreviation", "TeamName", "PredictedPosition"]].copy()
                quali_df.rename(columns={"PredictedPosition": "Position"}, inplace=True)
                quali_df["QualifyingPosition"] = quali_df["Position"]

            train_feats = pipeline.feature_engineer.build_training_dataset(
                races_df=races_df, qualis_df=qualis_df, practice_sessions_by_round=sessions_by_round, train_rounds=train_rounds, stage=stage
            )
            target_feats = pipeline.feature_engineer.build_target_dataset(
                target_round=target_round,
                track_name=track_info.get("display_name", track_key.title()),
                quali_df=quali_df,
                practice_sessions=weekend_sessions,
                season_races_df=races_df,
                season_qualis_df=qualis_df,
                stage=stage,
                track_key=track_key,
            )
            report = pipeline.data_validator.audit_leakage(train_feats, target_feats, target_round)
        except Exception as e:
            print_error(f"Validation failed: {e}", exc=e)
            raise typer.Exit(code=1)

    status_str = f"[bold {GREEN_FLAG}]PASSED (Zero Leakage)[/]" if report["is_clean"] else f"[bold {APEX_RED}]FAILED[/]"
    content = (
        f"Validation Status: {status_str}\n"
        f"Training Samples : [bold]{report['training_samples']}[/]\n"
        f"Target Lineup    : [bold]{report['target_drivers']} drivers[/]"
    )
    if report["violations"]:
        content += "\n\n[bold red]Violations Detected:[/bold red]\n"
        for v in report["violations"]:
            content += f"  • {v}\n"

    console.print(Panel(content, title=f"[bold]Data Integrity Audit: {track.upper()} {year}[/]", border_style=GREEN_FLAG if report["is_clean"] else APEX_RED))


# ==============================================================================
# Subcommand: collect
# ==============================================================================

@app.command("collect", help="Ingest official practice, qualifying, sprint, and race sessions via FastF1")
def cmd_collect(
    track: Optional[str] = typer.Option(None, "--track", "-t", help="Target circuit identifier (if omitted, collects full season)"),
    year: int = typer.Option(2026, "--year", "-y", help="Championship season year"),
    rounds: Optional[List[int]] = typer.Option(None, "--rounds", help="Explicit round numbers to collect"),
    session_mode: Optional[str] = typer.Option(None, "--session-mode", help="Session mode: full, prediction_only, practice_only"),
    include_practice: bool = typer.Option(False, "--include-practice", help="Include practice sessions (FP1, FP2, FP3)"),
    force: bool = typer.Option(False, "--force", help="Force re-collection of existing session files"),
) -> None:
    """Executes FastF1 session telemetry collection."""
    from src.data.collect_season import collect_season_data, collect_grand_prix

    mode = session_mode or ("full" if include_practice else "prediction_only")

    with console.status(f"[bold {TELEMETRY_CYAN}]Ingesting FastF1 session datasets (Year={year}, Mode={mode})...[/]"):
        try:
            if track:
                collect_grand_prix(track_id=track, year=year, force=force, verbose=True, session_mode=mode)
            else:
                collect_season_data(year=year, rounds=rounds, force=force, verbose=True, session_mode=mode)
            console.print(f"[bold {GREEN_FLAG}]Telemetry collection completed successfully.[/]\n")
        except Exception as e:
            print_error(f"Collection failed: {e}", exc=e)
            raise typer.Exit(code=1)


# ==============================================================================
# Subcommand: clean
# ==============================================================================

@app.command("clean", help="Safely manage and selectively delete cached data or prediction outputs")
def cmd_clean(
    cache: bool = typer.Option(False, "--cache", help="Delete FastF1 cache and temporary download files"),
    outputs: bool = typer.Option(False, "--outputs", help="Delete prediction outputs and generated reports"),
    collected: bool = typer.Option(False, "--collected", help="Delete collected session datasets"),
    track: Optional[str] = typer.Option(None, "--track", "-t", help="Target circuit identifier"),
    year: Optional[int] = typer.Option(None, "--year", "-y", help="Season championship year"),
    round_num: Optional[int] = typer.Option(None, "--round", "-r", help="Championship round number"),
    clean_all: bool = typer.Option(False, "--all", help="Delete all project-generated data"),
    include_historical: bool = typer.Option(False, "--include-historical", help="Opt-in to include historical training data"),
    preview: bool = typer.Option(False, "--preview", help="Dry run: preview files without deleting"),
    force: bool = typer.Option(False, "--force", "-y", help="Bypass confirmation prompt"),
    audit_log: bool = typer.Option(False, "--audit-log", help="Display recent deletion audit log"),
) -> None:
    """Safely manages project-generated storage cleanup with mandatory dry-run verification."""
    from src.data.data_manager import DataManager

    dm = DataManager(root_dir=PROJECT_ROOT)

    if audit_log:
        records = dm.audit_logger.get_audit_history(limit=15)
        table = Table(title="Deletion Audit History", border_style="dim", header_style="bold cyan")
        table.add_column("Timestamp")
        table.add_column("Status")
        table.add_column("Operation")
        table.add_column("Files Reclaimed", justify="right")
        table.add_column("Storage", justify="right")

        for r in reversed(records):
            table.add_row(
                r.get("timestamp", "-")[:19].replace("T", " "),
                r.get("status", "Verified"),
                r.get("operation_type", "-"),
                str(r.get("files_deleted_count", 0)),
                dm._format_size(r.get("storage_reclaimed_bytes", 0)),
            )
        console.print(table)
        return

    if not any([clean_all, cache, outputs, collected, track, round_num]):
        print_error("Must specify a target to clean: --cache, --outputs, --collected, --all, or --track")
        raise typer.Exit(code=2)

    # File discovery
    if clean_all:
        all_dict = dm.discover_all_project_generated_data(include_historical=include_historical)
        target_files = [f for flist in all_dict.values() for f in flist]
        op_name = "Delete all project-generated data"
        op_type = "all_generated_data"
    elif cache:
        target_files = dm.discover_cache_data(year=year, track=track)
        op_name = f"Delete Cached Data ({year if year else 'All Seasons'})"
        op_type = "cache"
    elif outputs:
        target_files = dm.discover_prediction_outputs(track_key=track, year=year)
        op_name = f"Delete Prediction Outputs ({track.title() if track else 'All Circuits'})"
        op_type = "outputs"
    elif collected:
        cat_dict = dm.discover_collected_datasets(include_historical=include_historical, year=year)
        target_files = [f for flist in cat_dict.values() for f in flist]
        op_name = "Delete Collected Datasets"
        op_type = "collected"
    elif track or round_num:
        eff_year = year or 2026
        gp_dict = dm.discover_grand_prix_data(track_key=track or "", year=eff_year, round_num=round_num, include_historical=include_historical)
        target_files = [f for flist in gp_dict.values() for f in flist]
        op_name = f"Delete Grand Prix Data ({track or f'Round {round_num}'} {eff_year})"
        op_type = "grand_prix"
    else:
        target_files = []
        op_name = "Data Deletion"
        op_type = "custom"

    plan = dm.build_deletion_preview(target_files, operation_name=op_name, include_historical=include_historical, operation_type=op_type)

    # Preview Table
    table = Table(title=f"Deletion Preview Plan: {plan['operation']}", border_style=APEX_RED, header_style=f"bold {TITANIUM_WHITE}")
    table.add_column("Category")
    table.add_column("Target Files", justify="right")
    table.add_column("Estimated Space Reclaimed", justify="right", style=TELEMETRY_CYAN)

    for cat, cnt in plan.get("categories", {}).items():
        table.add_row(cat, str(cnt), "-")
    table.add_row("[bold]TOTAL[/]", f"[bold]{plan['file_count']}[/]", f"[bold]{plan['storage_formatted']}[/]")
    console.print(table)
    console.print(f"[dim]Safety: Source code, configs, tests, and audit logs are permanently protected.[/dim]\n")

    if plan["file_count"] == 0:
        console.print("[yellow]No eligible files discovered for deletion.[/yellow]\n")
        return

    if preview:
        console.print("[green]Dry run complete. No files were removed.[/green]\n")
        return

    if not force:
        confirm = typer.confirm(f"Are you sure you want to permanently delete these {plan['file_count']} files?")
        if not confirm:
            console.print("[yellow]Operation cancelled by user. No files were deleted.[/yellow]\n")
            return

    res = dm.execute_deletion_plan(plan, confirmed=True)
    console.print(f"[bold {GREEN_FLAG}]Deletion complete: {res.get('files_deleted_count', 0)} files deleted ({res.get('storage_reclaimed_formatted', '0 B')} reclaimed).[/]\n")


# ==============================================================================
# Subcommand: circuits
# ==============================================================================

@app.command("circuits", help="List all 23 championship circuits in the 2026 FIA Formula 1 Calendar")
def cmd_circuits() -> None:
    """Displays 2026 championship calendar registry."""
    from src.data.data_loader import DataLoader

    loader = DataLoader(root_dir=PROJECT_ROOT)
    tracks_cfg = loader.tracks_config

    table = Table(title="2026 FIA Formula 1 World Championship Calendar", border_style="dim", header_style=f"bold {TITANIUM_WHITE}")
    table.add_column("Rnd", justify="right", style=TELEMETRY_CYAN)
    table.add_column("Circuit Name", style="bold")
    table.add_column("Country")
    table.add_column("Key", style=MUTED_GREY)
    table.add_column("Length (km)", justify="right")
    table.add_column("Type")
    table.add_column("Overtaking", justify="right")
    table.add_column("Speed Bias", justify="right")
    table.add_column("Sprint", justify="center")

    rows = []
    for tk, tinfo in tracks_cfg.items():
        rnd = tinfo.get("round_2026", 99)
        name = tinfo.get("display_name", tk.title())
        country = tinfo.get("country", "")
        length = f"{tinfo.get('track_length_km', 0.0):.3f}"
        ctype = str(tinfo.get("circuit_type", "permanent")).title()
        overtaking = f"{tinfo.get('overtaking_ease', 0.5):.2f}"
        speed = f"{tinfo.get('speed_bias', 0.5):.2f}"
        sprint = f"[{TIMING_PURPLE}]Yes[/]" if tinfo.get("sprint_weekend_2026") else "No"
        rows.append((rnd, name, country, tk, length, ctype, overtaking, speed, sprint))

    rows.sort(key=lambda x: x[0])
    for r in rows:
        table.add_row(f"R{r[0]:02d}", r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8])

    console.print(table)


# ==============================================================================
# Subcommand: health
# ==============================================================================

@app.command("health", help="Audit local environment, FastF1 cache, datasets, and ML package dependencies")
def cmd_health() -> None:
    """Executes system health and dependency readiness checks."""
    from src.data.data_loader import DataLoader

    loader = DataLoader(root_dir=PROJECT_ROOT)
    table = Table(title="ApexGrid System Health & Diagnostic Readiness", border_style="dim", header_style=f"bold {TITANIUM_WHITE}")
    table.add_column("Status", justify="center")
    table.add_column("Subsystem")
    table.add_column("Component")
    table.add_column("Diagnostic Details")

    # Runtime
    table.add_row(f"[{GREEN_FLAG}]OK[/]", "Runtime", "Python Interpreter", f"v{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} ({sys.executable})")

    # Packages
    packages = [
        ("fastf1", "FastF1 Telemetry Engine"),
        ("pandas", "Data Manipulation Core"),
        ("scikit-learn", "Scikit-Learn ML"),
        ("xgboost", "XGBoost Engine"),
        ("rich", "Rich Terminal Engine"),
        ("typer", "Typer CLI Framework"),
    ]
    for pkg, desc in packages:
        try:
            mod_name = "sklearn" if pkg == "scikit-learn" else pkg
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "Available")
            table.add_row(f"[{GREEN_FLAG}]OK[/]", "Dependencies", desc, f"{pkg} v{ver}")
        except ImportError:
            table.add_row(f"[{APEX_RED}]FAIL[/]", "Dependencies", desc, f"Missing: {pkg}")

    # FastF1 Cache
    cache_dir = loader.root_dir / "data" / "cache" / "fastf1"
    if cache_dir.exists():
        size_mb = sum(f.stat().st_size for f in cache_dir.glob("**/*") if f.is_file()) / (1024 * 1024)
        table.add_row(f"[{GREEN_FLAG}]OK[/]", "Storage", "FastF1 Cache", f"Active ({size_mb:.1f} MB in {cache_dir})")
    else:
        table.add_row(f"[{MUTED_GREY}]INFO[/]", "Storage", "FastF1 Cache", "Cache not yet populated")

    # Circuit Registry
    if len(loader.tracks_config) >= 23:
        table.add_row(f"[{GREEN_FLAG}]OK[/]", "Configuration", "2026 Circuit Registry", f"All {len(loader.tracks_config)} Grand Prix circuits loaded")
    else:
        table.add_row(f"[{APEX_RED}]WARN[/]", "Configuration", "2026 Circuit Registry", f"{len(loader.tracks_config)} circuits found")

    # Raw telemetry
    if loader.raw_data_dir.exists():
        file_count = len(list(loader.raw_data_dir.glob("**/*.csv")))
        table.add_row(f"[{GREEN_FLAG}]OK[/]", "Storage", "Raw Telemetry Datasets", f"{file_count} session files found in {loader.raw_data_dir}")
    else:
        table.add_row(f"[{APEX_RED}]WARN[/]", "Storage", "Raw Telemetry Datasets", f"Directory missing: {loader.raw_data_dir}")

    console.print(table)


# ==============================================================================
# Subcommand: menu
# ==============================================================================

@app.command("menu", help="Launch classic interactive terminal menu")
def cmd_menu() -> None:
    """Launches classic text-based interactive menu."""
    from src.cli.menu import InteractiveMenu
    InteractiveMenu(root_dir=PROJECT_ROOT).run()


# ==============================================================================
# Nested Sub-App: history
# ==============================================================================

@history_app.callback(invoke_without_command=True)
def history_landing(ctx: typer.Context) -> None:
    """Historical Formula 1 statistics subsystem (1950–2026)."""
    if ctx.invoked_subcommand is None:
        from src.cli.historical_menu import HistoricalStatsMenu
        HistoricalStatsMenu().run()


@history_app.command("driver", help="Query career and season driver statistics")
def history_driver(
    driver_id: str = typer.Argument(..., help="Driver canonical ID or surname (e.g. alonso, senna, hamilton)"),
    season: Optional[int] = typer.Option(None, "--season", "-y", help="Optional championship season filter"),
) -> None:
    """Fetches driver career stats from provider archive."""
    from src.historical.api_client import HistoricalAPIClient
    from src.historical.analytics.driver_analytics import DriverAnalyticsService

    svc = DriverAnalyticsService(HistoricalAPIClient())
    with console.status(f"[bold {TELEMETRY_CYAN}]Fetching historical driver record for '{driver_id}'...[/]"):
        try:
            stats = svc.get_driver_stats(driver_id=driver_id, season=season)
        except Exception as e:
            print_error(f"Failed to query driver records: {e}", exc=e)
            raise typer.Exit(code=1)

    if not stats:
        console.print(f"[yellow]No historical records found for '{driver_id}'.[/yellow]")
        return

    # Metric Cards Table
    table = Table(title=f"Driver Profile: {stats.driver_name.upper()} ({stats.nationality or 'Unknown'})", border_style="dim")
    table.add_column("Starts", justify="right", style="bold")
    table.add_column("Wins", justify="right", style=f"bold {APEX_RED}")
    table.add_column("Podiums", justify="right", style=f"bold {GREEN_FLAG}")
    table.add_column("Poles (Session P1)", justify="right", style=f"bold {TIMING_PURPLE}")
    table.add_column("Total Points", justify="right")
    table.add_column("DNFs", justify="right", style=MUTED_GREY)

    poles_str = str(stats.qualifying_session_p1s) if stats.qualifying_session_p1s is not None else "N/A"
    table.add_row(
        str(stats.total_starts),
        str(stats.wins),
        str(stats.podiums),
        poles_str,
        f"{stats.total_points:.1f}",
        str(stats.dnf_count),
    )
    console.print(table)

    # Season Breakdown Table
    if stats.seasons_breakdown:
        breakdown_table = Table(title="Seasons Breakdown", border_style="dim", header_style="bold cyan")
        breakdown_table.add_column("Season", justify="right")
        breakdown_table.add_column("Constructor")
        breakdown_table.add_column("Starts", justify="right")
        breakdown_table.add_column("Wins", justify="right")
        breakdown_table.add_column("Podiums", justify="right")
        breakdown_table.add_column("Poles", justify="right")
        breakdown_table.add_column("Points", justify="right")
        breakdown_table.add_column("Avg Finish", justify="right")

        for sb in stats.seasons_breakdown:
            pole_val = str(sb.poles) if sb.poles is not None else ("N/A" if sb.season < 1994 else "Error")
            s_label = f"{sb.season} (Prov)" if sb.is_provisional else str(sb.season)
            breakdown_table.add_row(
                s_label,
                sb.constructor_name[:20],
                str(sb.starts),
                str(sb.wins),
                str(sb.podiums),
                pole_val,
                f"{sb.points:.1f}",
                f"P{sb.avg_finish:.1f}" if sb.avg_finish else "-",
            )
        console.print(breakdown_table)


@history_app.command("h2h", help="Compare two drivers entered as teammates in the same season")
def history_h2h(
    driver1: str = typer.Argument(..., help="First driver ID (e.g. verstappen)"),
    driver2: str = typer.Argument(..., help="Second driver ID (e.g. perez)"),
    season: int = typer.Option(2023, "--season", "-y", help="Championship season to compare"),
) -> None:
    """Compares teammates under identical machinery constraints."""
    from src.historical.api_client import HistoricalAPIClient
    from src.historical.analytics.teammate_analytics import TeammateAnalyticsService

    svc = TeammateAnalyticsService(HistoricalAPIClient())
    with console.status(f"[bold {TELEMETRY_CYAN}]Computing verified teammate head-to-head for {driver1} vs {driver2} ({season})...[/]"):
        try:
            stats = svc.compare_teammates(driver1_id=driver1, driver2_id=driver2, season=season)
        except Exception as e:
            print_error(f"Teammate comparison failed: {e}", exc=e)
            raise typer.Exit(code=1)

    if not stats:
        console.print(f"[yellow]No shared constructor cohabitation found for {driver1} and {driver2} in {season}.[/yellow]")
        return

    table = Table(
        title=f"Teammate Comparison: {stats.driver1_name} vs {stats.driver2_name} ({stats.constructor_name}, {season})",
        border_style="dim",
        header_style=f"bold {TITANIUM_WHITE}",
    )
    table.add_column("Performance Dimension", style="bold")
    table.add_column(stats.driver1_name, justify="right", style=TELEMETRY_CYAN)
    table.add_column(stats.driver2_name, justify="right", style=APEX_RED)
    table.add_column("Differential / Context", style=MUTED_GREY)

    table.add_row("Races as Teammates", str(stats.races_cohabited), str(stats.races_cohabited), f"{stats.both_finished_count} both finished")
    table.add_row("Race Head-to-Head", str(stats.race_h2h[0]), str(stats.race_h2h[1]), "Ahead when both classified")
    table.add_row("Qualifying Head-to-Head", str(stats.qualifying_h2h[0]), str(stats.qualifying_h2h[1]), f"{stats.qualifying_sessions_compared} sessions compared")
    table.add_row("Grand Prix Wins", str(stats.wins_comparison[0]), str(stats.wins_comparison[1]), "")
    table.add_row("Podium Finishes", str(stats.podiums_comparison[0]), str(stats.podiums_comparison[1]), "")
    table.add_row("Championship Points", f"{stats.points_comparison[0]:.1f}", f"{stats.points_comparison[1]:.1f}", "")
    table.add_row("Retirements (DNFs)", str(stats.dnfs_comparison[0]), str(stats.dnfs_comparison[1]), "")

    console.print(table)


@history_app.command("qualifying", help="Query driver qualifying analytics and pole position stats")
def history_qualifying(
    driver_id: str = typer.Argument(..., help="Driver ID (e.g. hamilton, senna, verstappen)"),
    season: Optional[int] = typer.Option(None, "--season", "-y", help="Optional season filter"),
) -> None:
    """Fetches qualifying performance records."""
    from src.historical.api_client import HistoricalAPIClient
    from src.historical.analytics.qualifying_analytics import QualifyingAnalyticsService

    svc = QualifyingAnalyticsService(HistoricalAPIClient())
    with console.status(f"[bold {TELEMETRY_CYAN}]Fetching qualifying analytics for '{driver_id}'...[/]"):
        try:
            summary = svc.get_driver_qualifying_summary(driver_id=driver_id, season=season)
        except Exception as e:
            print_error(f"Qualifying query failed: {e}", exc=e)
            raise typer.Exit(code=1)

    if not summary or summary.total_sessions == 0:
        console.print(f"[yellow]No qualifying records found for '{driver_id}'.[/yellow]")
        return

    table = Table(title=f"Qualifying Intelligence: {summary.driver_name or driver_id.title()}", border_style="dim", header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Context", style=MUTED_GREY)

    table.add_row("Total Qualifying Sessions", str(summary.total_sessions), f"Season: {summary.season or 'Career'}")
    table.add_row("Pole Positions (P1)", str(summary.pole_positions), "")
    table.add_row("Front Row Starts (P1-P2)", str(summary.front_row_starts), "")
    table.add_row("Top 10 Starts", str(summary.top_10_starts), "")
    avg_pos = f"P{summary.avg_qualifying_position:.2f}" if summary.avg_qualifying_position else "-"
    table.add_row("Average Qualifying Pos", avg_pos, "")
    delta = f"{summary.avg_grid_drop_or_gain:+.2f} pos" if summary.avg_grid_drop_or_gain is not None else "-"
    table.add_row("Avg Grid Drop/Gain", delta, "Net penalties / changes")

    console.print(table)


@history_app.command("race", help="Query race craft, positions gained, and retirement records")
def history_race(
    driver_id: str = typer.Argument(..., help="Driver ID (e.g. leclerc, alonso, prost)"),
    season: Optional[int] = typer.Option(None, "--season", "-y", help="Optional season filter"),
) -> None:
    """Fetches race execution and craft summary."""
    from src.historical.api_client import HistoricalAPIClient
    from src.historical.analytics.race_analytics import RacePerformanceAnalyticsService

    svc = RacePerformanceAnalyticsService(HistoricalAPIClient())
    with console.status(f"[bold {TELEMETRY_CYAN}]Fetching race performance data for '{driver_id}'...[/]"):
        try:
            summary = svc.get_driver_race_performance(driver_id=driver_id, season=season)
        except Exception as e:
            print_error(f"Race performance query failed: {e}", exc=e)
            raise typer.Exit(code=1)

    if not summary or summary.total_races == 0:
        console.print(f"[yellow]No race entries found for '{driver_id}'.[/yellow]")
        return

    table = Table(title=f"Race Execution Profile: {summary.driver_name or driver_id.title()}", border_style="dim", header_style="bold green")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Details", style=MUTED_GREY)

    table.add_row("Total Races / Starts", f"{summary.total_races} / {summary.starts}", f"{summary.classified_finishes} classified")
    table.add_row("Wins / Podiums", f"{summary.wins} / {summary.podiums}", f"Win rate: {(summary.wins / max(1, summary.starts)) * 100:.1f}%")
    table.add_row("Points Finishes", str(summary.points_finishes), f"Total Points: {summary.total_points:.1f}")
    table.add_row("Retirements (DNFs)", f"{summary.dnfs} ({summary.dnf_rate_pct:.1f}%)", "")
    pos_gained = str(summary.positions_gained_total) if summary.positions_gained_total is not None else "-"
    avg_gained = f"{summary.avg_positions_gained:+.2f} per start" if summary.avg_positions_gained is not None else "-"
    table.add_row("Net Positions Gained", pos_gained, avg_gained)

    console.print(table)


@history_app.command("circuit", help="Query circuit Grand Prix hosting history, record winners, and pole conversion")
def history_circuit(
    circuit_id: str = typer.Argument(..., help="Circuit identifier (e.g. monza, silverstone, spa, monaco)"),
) -> None:
    """Fetches circuit venue history."""
    from src.historical.api_client import HistoricalAPIClient
    from src.historical.analytics.circuit_analytics import CircuitAnalyticsService

    svc = CircuitAnalyticsService(HistoricalAPIClient())
    with console.status(f"[bold {TELEMETRY_CYAN}]Fetching circuit history for '{circuit_id}'...[/]"):
        try:
            summary = svc.get_circuit_history(circuit_id=circuit_id)
        except Exception as e:
            print_error(f"Circuit query failed: {e}", exc=e)
            raise typer.Exit(code=1)

    if not summary or summary.total_races_hosted == 0:
        console.print(f"[yellow]No Grand Prix records found for circuit '{circuit_id}'.[/yellow]")
        return

    table = Table(title=f"Circuit History: {summary.circuit_name} ({summary.locality}, {summary.country})", border_style="dim", header_style="bold yellow")
    table.add_column("Category", style="bold")
    table.add_column("Detail")
    table.add_column("Record / Count", justify="right")

    table.add_row("Hosting Era", f"{summary.first_gp_year} – {summary.last_gp_year}", f"{summary.total_races_hosted} GPs")

    for d_name, wins in (summary.top_winning_drivers or [])[:3]:
        table.add_row("Most Wins (Driver)", d_name, f"{wins} wins")

    for c_name, wins in (summary.top_winning_constructors or [])[:3]:
        table.add_row("Most Wins (Constructor)", c_name, f"{wins} wins")

    for yr, d_name, c_name in (summary.recent_winners or [])[:5]:
        table.add_row(f"Winner {yr}", d_name, c_name)

    console.print(table)


@history_app.command("standings", help="Query Formula 1 World Championship standings (1950–2026)")
def history_standings(
    season: int = typer.Argument(2024, help="Championship season year"),
) -> None:
    """Fetches official championship standings."""
    from src.historical.api_client import HistoricalAPIClient
    from src.historical.analytics.season_analytics import SeasonAnalyticsService

    svc = SeasonAnalyticsService(HistoricalAPIClient())
    with console.status(f"[bold {TELEMETRY_CYAN}]Fetching championship standings for {season}...[/]"):
        try:
            summary = svc.get_season_summary(year=season)
        except Exception as e:
            print_error(f"Standings query failed: {e}", exc=e)
            raise typer.Exit(code=1)

    if not summary or not getattr(summary, "top_drivers_standings", None):
        console.print(f"[yellow]No driver standings available for season {season}.[/yellow]")
        return

    table = Table(title=f"FIA Formula 1 World Championship Standings — {season}", border_style="dim", header_style="bold cyan")
    table.add_column("Pos", justify="right", style="bold")
    table.add_column("Driver", style="bold")
    table.add_column("Constructor", style=MUTED_GREY)
    table.add_column("Points", justify="right", style=TELEMETRY_CYAN)
    table.add_column("Wins", justify="right", style=APEX_RED)

    for rank, name, team, pts, wins in summary.top_drivers_standings:
        table.add_row(
            str(rank),
            name,
            team,
            f"{pts:.1f}",
            str(wins),
        )

    console.print(table)


@history_app.command("session", help="Query official Grand Prix, qualifying, sprint, or practice results")
def history_session(
    year: int = typer.Option(2024, "--year", "-y", help="Championship season year (1950–2026)"),
    round_num: int = typer.Option(1, "--round", "-r", help="Championship round number"),
    session_type: str = typer.Option("race", "--session", "-s", help="Session type: race, qualifying, sprint, sprint_qualifying, fp1, fp2, fp3"),
) -> None:
    """Queries official FIA session classifications and telemetry lap times."""
    from src.historical.analytics.session_results import SessionResultsService

    svc = SessionResultsService()
    stype_clean = session_type.strip().lower()

    with console.status(f"[bold {TELEMETRY_CYAN}]Fetching session classification for {year} Round {round_num} ({stype_clean})...[/]"):
        try:
            if stype_clean in ("race", "gp"):
                results = svc.get_race_results(year, round_num)
                if not results:
                    console.print(f"[yellow]No race classification recorded for {year} Round {round_num}.[/yellow]")
                    return
                table = Table(title=f"Grand Prix Race Classification — {year} Round {round_num}", border_style="dim", header_style=f"bold {TITANIUM_WHITE}")
                table.add_column("Pos", justify="right", style="bold")
                table.add_column("Driver", style="bold")
                table.add_column("Constructor", style=MUTED_GREY)
                table.add_column("Grid", justify="right")
                table.add_column("Laps", justify="right")
                table.add_column("Time/Gap")
                table.add_column("Status")
                table.add_column("Pts", justify="right", style=TELEMETRY_CYAN)

                for r in results:
                    pos = str(getattr(r, "position", None) or getattr(r, "position_text", "-"))
                    grid = str(getattr(r, "grid_position", getattr(r, "grid", None)) or "-")
                    time_disp = getattr(r, "race_time_or_gap", getattr(r, "time_str", None)) or ("Leader" if pos == "1" else "")
                    pts_val = getattr(r, "points", 0.0) or 0.0
                    table.add_row(
                        pos,
                        getattr(r, "driver_name", "") or getattr(r, "driver_id", "-"),
                        getattr(r, "constructor_name", "") or getattr(r, "constructor_id", "-"),
                        grid,
                        str(getattr(r, "laps_completed", getattr(r, "laps", "-"))),
                        time_disp,
                        getattr(r, "status", "-"),
                        f"{pts_val:.1f}" if isinstance(pts_val, (int, float)) else str(pts_val),
                    )
                console.print(table)

            elif stype_clean in ("qualifying", "quali"):
                results = svc.get_qualifying_results(year, round_num)
                if not results:
                    console.print(f"[yellow]No qualifying classification recorded for {year} Round {round_num}.[/yellow]")
                    return
                table = Table(title=f"Official Qualifying Classification — {year} Round {round_num}", border_style="dim", header_style=f"bold {TITANIUM_WHITE}")
                table.add_column("Pos", justify="right", style="bold")
                table.add_column("Driver", style="bold")
                table.add_column("Constructor", style=MUTED_GREY)
                table.add_column("Q1")
                table.add_column("Q2")
                table.add_column("Q3", style=TELEMETRY_CYAN)
                table.add_column("Laps", justify="right")

                for r in results:
                    table.add_row(
                        str(getattr(r, "position", "-")),
                        getattr(r, "driver_name", "") or getattr(r, "driver_id", "-"),
                        getattr(r, "constructor_name", "") or getattr(r, "constructor_id", "-"),
                        getattr(r, "q1", "-") or "-",
                        getattr(r, "q2", "-") or "-",
                        getattr(r, "q3", "-") or "-",
                        str(getattr(r, "laps", "-")),
                    )
                console.print(table)

            elif stype_clean in ("sprint", "sprint_race"):
                results = svc.get_sprint_results(year, round_num)
                if not results:
                    console.print(f"[yellow]No sprint race classification recorded for {year} Round {round_num}.[/yellow]")
                    return
                table = Table(title=f"Sprint Race Classification — {year} Round {round_num}", border_style="dim", header_style=f"bold {TITANIUM_WHITE}")
                table.add_column("Pos", justify="right", style="bold")
                table.add_column("Driver", style="bold")
                table.add_column("Constructor", style=MUTED_GREY)
                table.add_column("Grid", justify="right")
                table.add_column("Laps", justify="right")
                table.add_column("Time/Gap")
                table.add_column("Status")
                table.add_column("Pts", justify="right", style=TELEMETRY_CYAN)

                for r in results:
                    pos = str(getattr(r, "position", None) or getattr(r, "position_text", "-"))
                    grid = str(getattr(r, "grid_position", getattr(r, "grid", None)) or "-")
                    time_disp = getattr(r, "race_time_or_gap", getattr(r, "time_str", None)) or ("Leader" if pos == "1" else "")
                    pts_val = getattr(r, "points", 0.0) or 0.0
                    table.add_row(
                        pos,
                        getattr(r, "driver_name", "") or getattr(r, "driver_id", "-"),
                        getattr(r, "constructor_name", "") or getattr(r, "constructor_id", "-"),
                        grid,
                        str(getattr(r, "laps_completed", getattr(r, "laps", "-"))),
                        time_disp,
                        getattr(r, "status", "-"),
                        f"{pts_val:.1f}" if isinstance(pts_val, (int, float)) else str(pts_val),
                    )
                console.print(table)

            elif stype_clean in ("sq", "sprint_qualifying", "shootout"):
                results = svc.get_sprint_qualifying_results(year, round_num)
                if not results:
                    console.print(f"[yellow]No sprint qualifying data recorded for {year} Round {round_num}.[/yellow]")
                    return
                table = Table(title=f"Sprint Qualifying (SQ) Classification — {year} Round {round_num}", border_style="dim", header_style=f"bold {TITANIUM_WHITE}")
                table.add_column("Pos", justify="right", style="bold")
                table.add_column("Driver", style="bold")
                table.add_column("Best Lap", style=TELEMETRY_CYAN)
                table.add_column("Gap")
                table.add_column("Laps", justify="right")

                for r in results:
                    table.add_row(str(r["position"]), r["driver"], r["best_lap_time"], r["gap"], str(r["laps_completed"]))
                console.print(table)

            elif stype_clean in ("fp1", "fp2", "fp3", "practice"):
                sess_name = stype_clean if stype_clean.startswith("fp") else "fp1"
                results = svc.get_practice_results(year, round_num, session_name=sess_name)
                if not results:
                    console.print(f"[yellow]No {sess_name.upper()} practice telemetry recorded for {year} Round {round_num}.[/yellow]")
                    return
                table = Table(title=f"{sess_name.upper()} Practice Classification — {year} Round {round_num}", border_style="dim", header_style=f"bold {TITANIUM_WHITE}")
                table.add_column("Pos", justify="right", style="bold")
                table.add_column("Driver", style="bold")
                table.add_column("Best Lap", style=TELEMETRY_CYAN)
                table.add_column("Gap")
                table.add_column("Laps", justify="right")

                for r in results:
                    table.add_row(str(r["position"]), r["driver"], r["best_lap_time"], r["gap"], str(r["laps_completed"]))
                console.print(table)
            else:
                print_error(f"Unknown session type: '{session_type}'. Choose from race, qualifying, sprint, sprint_qualifying, fp1, fp2, fp3.")
                raise typer.Exit(code=2)
        except typer.Exit:
            raise
        except Exception as e:
            print_error(f"Session query failed: {e}", exc=e)
            raise typer.Exit(code=1)


@history_app.command("dominance", help="Query constructor and engine historical dominance rankings")
def history_dominance(
    metric: str = typer.Option("all_time", "--metric", "-m", help="Dominance metric: all_time, win_rate, one_two"),
    top: int = typer.Option(15, "--top", "-t", help="Number of records to display"),
) -> None:
    """Queries all-time constructor supremacy, single-season win rates, and 1-2 finish records."""
    from src.historical.analytics.dominance_analytics import DominanceAnalyticsService

    svc = DominanceAnalyticsService()
    metric_clean = metric.strip().lower()

    with console.status(f"[bold {TELEMETRY_CYAN}]Computing constructor dominance analytics ({metric_clean})...[/]"):
        try:
            if metric_clean in ("all_time", "constructors", "championships"):
                records = svc.get_constructor_dominance()[:top]
                table = Table(title="All-Time Formula 1 Constructor Supremacy Rankings", border_style="dim", header_style=f"bold {TITANIUM_WHITE}")
                table.add_column("Rank", justify="right", style="bold")
                table.add_column("Constructor", style="bold")
                table.add_column("Championships", justify="right", style=APEX_RED)
                table.add_column("Wins", justify="right", style=TELEMETRY_CYAN)
                table.add_column("Poles", justify="right")
                table.add_column("Podiums", justify="right")
                table.add_column("1-2s", justify="right")
                table.add_column("Starts", justify="right", style=MUTED_GREY)
                table.add_column("Win %", justify="right")

                for r in records:
                    table.add_row(
                        str(r["rank"]),
                        r["constructor"],
                        str(r["championships"]),
                        str(r["wins"]),
                        str(r["poles"]),
                        str(r["podiums"]),
                        str(r["one_twos"]),
                        str(r["starts"]),
                        f"{r['win_percentage']:.1f}%",
                    )
                console.print(table)

            elif metric_clean in ("win_rate", "seasons", "dominant"):
                seasons = svc.get_highest_win_rate_seasons(top_n=top)
                table = Table(title=f"Top {len(seasons)} Most Dominant Single Seasons in F1 History", border_style="dim", header_style=f"bold {TITANIUM_WHITE}")
                table.add_column("Rank", justify="right", style="bold")
                table.add_column("Season", justify="center")
                table.add_column("Constructor", style="bold")
                table.add_column("Wins/Races", justify="right")
                table.add_column("Win %", justify="right", style=TELEMETRY_CYAN)
                table.add_column("Champion Driver", style=APEX_RED)
                table.add_column("Historical Context", style=MUTED_GREY)

                for s in seasons:
                    table.add_row(
                        str(s["rank"]),
                        str(s["season"]),
                        s["constructor"],
                        f"{s['wins']}/{s['races']}",
                        f"{s['win_pct']:.1f}%",
                        s["driver_champs"],
                        s["notes"][:45],
                    )
                console.print(table)

            elif metric_clean in ("one_two", "1-2", "one_twos"):
                teams = svc.get_one_two_finish_rankings(top_n=top)
                table = Table(title=f"All-Time Constructor 1-2 Finish Rankings (Top {len(teams)})", border_style="dim", header_style=f"bold {TITANIUM_WHITE}")
                table.add_column("Rank", justify="right", style="bold")
                table.add_column("Constructor", style="bold")
                table.add_column("1-2 Finishes", justify="right", style=APEX_RED)
                table.add_column("Total Wins", justify="right", style=TELEMETRY_CYAN)
                table.add_column("1-2 % of Wins", justify="right")

                for t in teams:
                    table.add_row(
                        str(t["rank"]),
                        t["constructor"],
                        str(t["one_twos"]),
                        str(t["wins"]),
                        f"{t['one_two_pct_of_wins']:.1f}%",
                    )
                console.print(table)
            else:
                print_error(f"Unknown dominance metric: '{metric}'. Choose from all_time, win_rate, one_two.")
                raise typer.Exit(code=2)
        except typer.Exit:
            raise
        except Exception as e:
            print_error(f"Dominance query failed: {e}", exc=e)
            raise typer.Exit(code=1)


@history_app.command("records", help="Query all-time F1 records, streaks, and milestones")
def history_records(
    category: str = typer.Option("wins", "--category", "-c", help="Records category: wins, poles, podiums, streaks, milestones"),
    streak_type: str = typer.Option("wins", "--streak-type", help="Streak type (if category=streaks): wins, podiums, points, starts"),
    top: int = typer.Option(20, "--top", "-t", help="Number of records to display"),
) -> None:
    """Queries all-time driver records, unbroken streaks, and age milestones."""
    from src.historical.analytics.all_time_records import AllTimeRecordsService

    svc = AllTimeRecordsService()
    cat_clean = category.strip().lower()

    with console.status(f"[bold {TELEMETRY_CYAN}]Querying all-time Formula 1 records ({cat_clean})...[/]"):
        try:
            if cat_clean == "wins":
                records = svc.get_all_time_wins(top_n=top)
                table = Table(title=f"All-Time Grand Prix Winners (Top {len(records)})", border_style="dim", header_style=f"bold {TITANIUM_WHITE}")
                table.add_column("Rank", justify="right", style="bold")
                table.add_column("Driver", style="bold")
                table.add_column("Nationality", style=MUTED_GREY)
                table.add_column("Wins", justify="right", style=APEX_RED)
                table.add_column("Starts", justify="right")
                table.add_column("Win %", justify="right", style=TELEMETRY_CYAN)

                for r in records:
                    table.add_row(str(r["rank"]), r["driver"], r["nationality"], str(r["wins"]), str(r["starts"]), f"{r['win_pct']:.1f}%")
                console.print(table)

            elif cat_clean in ("poles", "pole_positions"):
                records = svc.get_all_time_poles(top_n=top)
                table = Table(title=f"All-Time Pole Position Records (Top {len(records)})", border_style="dim", header_style=f"bold {TITANIUM_WHITE}")
                table.add_column("Rank", justify="right", style="bold")
                table.add_column("Driver", style="bold")
                table.add_column("Nationality", style=MUTED_GREY)
                table.add_column("Poles", justify="right", style=APEX_RED)
                table.add_column("Starts", justify="right")
                table.add_column("Pole %", justify="right", style=TELEMETRY_CYAN)

                for r in records:
                    table.add_row(str(r["rank"]), r["driver"], r["nationality"], str(r["poles"]), str(r["starts"]), f"{r['pole_pct']:.1f}%")
                console.print(table)

            elif cat_clean in ("podiums", "podium"):
                records = svc.get_all_time_podiums(top_n=top)
                table = Table(title=f"All-Time Podium Finishes (Top {len(records)})", border_style="dim", header_style=f"bold {TITANIUM_WHITE}")
                table.add_column("Rank", justify="right", style="bold")
                table.add_column("Driver", style="bold")
                table.add_column("Nationality", style=MUTED_GREY)
                table.add_column("Podiums", justify="right", style=APEX_RED)
                table.add_column("Starts", justify="right")
                table.add_column("Podium %", justify="right", style=TELEMETRY_CYAN)

                for r in records:
                    table.add_row(str(r["rank"]), r["driver"], r["nationality"], str(r["podiums"]), str(r["starts"]), f"{r['podium_pct']:.1f}%")
                console.print(table)

            elif cat_clean in ("streaks", "streak"):
                streak_map = {
                    "wins": "consecutive_wins",
                    "podiums": "consecutive_podiums",
                    "points": "consecutive_points",
                    "starts": "consecutive_starts",
                }
                st_key = streak_map.get(streak_type.strip().lower(), "consecutive_wins")
                streaks = svc.get_consecutive_records(st_key)[:top]
                title = st_key.replace("_", " ").title()
                table = Table(title=f"All-Time Streaks: {title}", border_style="dim", header_style=f"bold {TITANIUM_WHITE}")
                table.add_column("Rank", justify="right", style="bold")
                table.add_column("Driver", style="bold")
                table.add_column("Streak Count", justify="right", style=APEX_RED)
                table.add_column("Season", justify="center")
                table.add_column("Grand Prix Span", style=MUTED_GREY)

                for s in streaks:
                    table.add_row(str(s["rank"]), s["driver"], str(s["count"]), s["season"], s["races"])
                console.print(table)

            elif cat_clean in ("milestones", "age"):
                milestones = svc.get_age_milestones()
                # Youngest
                table_y = Table(title="Youngest Grand Prix Winners", border_style="dim", header_style=f"bold {TITANIUM_WHITE}")
                table_y.add_column("Rank", justify="right", style="bold")
                table_y.add_column("Driver", style="bold")
                table_y.add_column("Age", style=TELEMETRY_CYAN)
                table_y.add_column("Event")
                for x in milestones["youngest_winners"]:
                    table_y.add_row(str(x["rank"]), x["driver"], x["age"], x["event"])
                console.print(table_y)

                # Oldest
                table_o = Table(title="Oldest Grand Prix Winners", border_style="dim", header_style=f"bold {TITANIUM_WHITE}")
                table_o.add_column("Rank", justify="right", style="bold")
                table_o.add_column("Driver", style="bold")
                table_o.add_column("Age", style=TELEMETRY_CYAN)
                table_o.add_column("Event")
                for x in milestones["oldest_winners"]:
                    table_o.add_row(str(x["rank"]), x["driver"], x["age"], x["event"])
                console.print(table_o)

                # Most starts no win
                table_u = Table(title="Most Race Starts Without a Grand Prix Win", border_style="dim", header_style=f"bold {TITANIUM_WHITE}")
                table_u.add_column("Rank", justify="right", style="bold")
                table_u.add_column("Driver", style="bold")
                table_u.add_column("Starts", justify="right", style=APEX_RED)
                table_u.add_column("Best Finish")
                for x in milestones["most_starts_no_win"]:
                    table_u.add_row(str(x["rank"]), x["driver"], str(x["starts"]), x["best_finish"])
                console.print(table_u)
            else:
                print_error(f"Unknown records category: '{category}'. Choose from wins, poles, podiums, streaks, milestones.")
                raise typer.Exit(code=2)
        except typer.Exit:
            raise
        except Exception as e:
            print_error(f"Records query failed: {e}", exc=e)
            raise typer.Exit(code=1)


@history_app.command("reliability", help="Query historical mechanical attrition and constructor reliability")
def history_reliability(
    view: str = typer.Option("eras", "--view", "-v", help="Reliability view: eras, attritional_races, constructors"),
    top: int = typer.Option(10, "--top", "-t", help="Number of records to display"),
) -> None:
    """Queries historical mechanical attrition rates, race finish percentages, and constructor reliability."""
    from src.historical.analytics.reliability_analytics import ReliabilityAnalyticsService

    svc = ReliabilityAnalyticsService()
    v_clean = view.strip().lower()

    with console.status(f"[bold {TELEMETRY_CYAN}]Querying reliability analytics ({v_clean})...[/]"):
        try:
            if v_clean in ("eras", "trends", "attrition"):
                eras = svc.get_season_attrition_rates()[:top]
                table = Table(title="Historical Attrition Evolution Across Eras (1950–2026)", border_style="dim", header_style=f"bold {TITANIUM_WHITE}")
                table.add_column("Era", style="bold")
                table.add_column("Races", justify="right")
                table.add_column("Finish Rate %", justify="right", style=GREEN_FLAG)
                table.add_column("Mech. DNF %", justify="right", style=APEX_RED)
                table.add_column("Accident %", justify="right", style=FLAG_YELLOW)
                table.add_column("Avg Finishers", justify="right")

                for e in eras:
                    table.add_row(
                        e["era"],
                        str(e["races"]),
                        f"{e['finish_rate_pct']:.1f}%",
                        f"{e['mech_dnf_pct']:.1f}%",
                        f"{e['accident_dnf_pct']:.1f}%",
                        f"{e['avg_finishers']:.1f}",
                    )
                console.print(table)

            elif v_clean in ("attritional_races", "races", "carnage"):
                races = svc.get_most_attritional_races(top_n=top)
                table = Table(title="Most Attritional Races in F1 History (Fewest Checkered Flag Finishers)", border_style="dim", header_style=f"bold {TITANIUM_WHITE}")
                table.add_column("Grand Prix", style="bold")
                table.add_column("Year", justify="center")
                table.add_column("Starters", justify="right")
                table.add_column("Finishers", justify="right", style=APEX_RED)
                table.add_column("Winner", style=TELEMETRY_CYAN)
                table.add_column("Race Context", style=MUTED_GREY)

                for r in races:
                    table.add_row(r["race"], str(r["year"]), str(r["starters"]), str(r["finishers"]), r["winner"], r["notes"][:45])
                console.print(table)

            elif v_clean in ("constructors", "teams", "reliable"):
                constructors = svc.get_most_reliable_constructors(top_n=top)
                table = Table(title="Constructor Reliability Rankings (Modern V6 Turbo-Hybrid Era)", border_style="dim", header_style=f"bold {TITANIUM_WHITE}")
                table.add_column("Rank", justify="right", style="bold")
                table.add_column("Constructor", style="bold")
                table.add_column("Regulatory Era", style=MUTED_GREY)
                table.add_column("Race Starts", justify="right")
                table.add_column("Finish Rate %", justify="right", style=GREEN_FLAG)
                table.add_column("Engine PU")

                for c in constructors:
                    table.add_row(str(c["rank"]), c["constructor"], c["era"], str(c["starts"]), f"{c['finish_rate_pct']:.1f}%", c["engine"])
                console.print(table)
            else:
                print_error(f"Unknown reliability view: '{view}'. Choose from eras, attritional_races, constructors.")
                raise typer.Exit(code=2)
        except typer.Exit:
            raise
        except Exception as e:
            print_error(f"Reliability query failed: {e}", exc=e)
            raise typer.Exit(code=1)


@history_app.command("eras", help="Query regulatory eras and macro-metrics comparison (1950–2026)")
def history_eras() -> None:
    """Queries technical and regulatory era comparisons across Formula 1 history."""
    from src.historical.analytics.era_analytics import EraAnalyticsService

    svc = EraAnalyticsService()
    with console.status(f"[bold {TELEMETRY_CYAN}]Computing regulatory era comparisons...[/]"):
        try:
            eras = svc.get_era_overview()
        except Exception as e:
            print_error(f"Era query failed: {e}", exc=e)
            raise typer.Exit(code=1)

    table = Table(title="The 8 Canonical Technical Eras of Formula 1 (1950–2026)", border_style="dim", header_style=f"bold {TITANIUM_WHITE}")
    table.add_column("Era", style="bold")
    table.add_column("Period", justify="center")
    table.add_column("Avg Races/Yr", justify="right")
    table.add_column("Avg Grid", justify="right")
    table.add_column("Finish Rate %", justify="right", style=GREEN_FLAG)
    table.add_column("Winners/Yr", justify="right", style=APEX_RED)
    table.add_column("Teams/Yr", justify="right", style=TELEMETRY_CYAN)

    for e in eras:
        table.add_row(
            e["name"],
            e["period"],
            f"{e['avg_races_per_year']:.1f}",
            f"{e['avg_grid_size']:.1f}",
            f"{e['finish_rate_pct']:.1f}%",
            f"{e['unique_winners_per_year']:.1f}",
            f"{e['unique_winning_teams_per_yr']:.1f}",
        )
    console.print(table)

    console.print(f"\n[bold {TITANIUM_WHITE}]Technical Architecture & Defining Innovations:[/bold {TITANIUM_WHITE}]")
    for e in eras:
        console.print(f"  • [bold {TELEMETRY_CYAN}]{e['name']} ({e['period']}):[/bold {TELEMETRY_CYAN}] {e['defining_characteristics']}")


if __name__ == "__main__":
    app()
