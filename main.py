"""
F1 Race Prediction Engine — Master CLI Entrypoint
=================================================
Unified command-line interface for multi-track Formula 1 race predictions.

Commands:
    predict     Run end-to-end prediction pipeline for a circuit
    evaluate    Run chronological walk-forward cross-validation
    validate    Audit datasets for data integrity and future leakage
    collect     Fetch session telemetry via FastF1 (if needed)

Examples:
    python main.py predict --track sepang --year 2026 --stage after_qualifying
    python main.py predict --track baku --year 2026 --stage after_qualifying
    python main.py evaluate --track sepang --year 2026
    python main.py validate --track sepang --year 2026
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.race_prediction_pipeline import RacePredictionPipeline
from src.data.data_loader import DataLoader
from src.data.data_validator import DataValidator
from src.features.build_features import FeatureEngineer
from src.models.evaluate import ModelEvaluator
from src.utils.logging_utils import get_logger

logger = get_logger("main_cli")


def cmd_predict(args):
    """Executes prediction command."""
    pipeline = RacePredictionPipeline(root_dir=ROOT)
    try:
        results = pipeline.run(
            track_key=args.track,
            year=args.year,
            stage=args.stage
        )
    except (ValueError, RuntimeError) as e:
        logger.error(f"Prediction failed: {e}")
        print(f"\n[ERROR] Prediction cannot proceed: {e}\n")
        return

    df = results["predictions"]
    track_name = results["track_name"]

    top_n = getattr(args, "top", 10)

    # Missing session warning banner if provisional qualifying fallback was triggered for post-qualifying stages
    if results.get("is_provisional_quali") and args.stage in ["after_qualifying", "post_race"]:
        print("\n" + "!" * 80)
        print(f"WARNING: Target qualifying data was not found for stage '{args.stage}'.")
        print("Predictions are based on a fallback provisional qualifying model!")
        print("!" * 80)

    # If qualifying was provisionally predicted, display Race Qualifying Top N
    if results.get("is_provisional_quali") and results.get("qualifying_meta"):
        q_meta = results["qualifying_meta"]
        q_df = q_meta["predictions"].iloc[:top_n]
        print("\n" + "=" * 80)
        print(f"RACE QUALIFYING TOP {top_n} PREDICTION: {track_name.upper()} {args.year} [{args.stage.upper()}]")
        print(f"Model: {q_meta.get('selected_model')} | Walk-Forward MAE: {q_meta.get('selected_mae'):.4f} | Pace Sigma: {q_meta.get('calibrated_sigma'):.4f}")
        print("=" * 80)
        print(f"{'Pos':<4} {'Driver':<5} {'Team':<22} {'ExpQuali':<10} {'Pole%':<7} {'Top3%':<9}")
        print("-" * 80)
        for _, r in q_df.iterrows():
            print(
                f"{int(r['PredictedPosition']):<4} "
                f"{r['Abbreviation']:<5} "
                f"{r['TeamName'][:21]:<22} "
                f"{r['ExpectedQualifyingPosition']:<10.2f} "
                f"{r['PoleProbability']:<7.1f} "
                f"{r['Top3Probability']:<9.1f}"
            )
        print("=" * 80)

    # Grand Prix Race Top N Prediction
    top_race = df.iloc[:top_n]
    grid_source = "Provisional Qualifying Model" if results.get("is_provisional_quali") else "Actual Observed Qualifying Session"
    print("\n" + "=" * 80)
    print(f"GRAND PRIX RACE TOP {top_n} PREDICTION: {track_name.upper()} {args.year} [{args.stage.upper()}]")
    print(f"Selected Model   : {results.get('selected_model', 'N/A')}")
    print(f"Selection Method : {results.get('selection_method', 'Chronological Walk-Forward Validation')}")
    print(f"Selected MAE     : {results.get('selected_mae', 'N/A')}")
    print(f"Starting Grid    : {grid_source}")
    print("=" * 80)
    print(f"{'Pos':<4} {'Driver':<5} {'Team':<22} {'Grid':<5} {'ExpFinish':<10} {'Win%':<7} {'Podium%':<9} {'Points%':<9}")
    print("-" * 80)

    for _, r in top_race.iterrows():
        print(
            f"{int(r['PredictedPosition']):<4} "
            f"{r['Abbreviation']:<5} "
            f"{r['TeamName'][:21]:<22} "
            f"{int(r['ActualStartingGrid']):<5} "
            f"{r['ExpectedFinish']:<10.2f} "
            f"{r['WinProbability']:<7.1f} "
            f"{r['Top3Probability']:<9.1f} "
            f"{r['PointsProbability']:<9.1f}"
        )
    print("=" * 80)

    # Post-Race Out-of-Sample Scorecard (if post_race stage with actual race results)
    if results.get("post_race_metrics"):
        m = results["post_race_metrics"]
        print("\n" + "=" * 80)
        print(f"POST-RACE OUT-OF-SAMPLE EVALUATION: {track_name.upper()} {args.year}")
        print("=" * 80)
        print(f"MAE (Expected Finish) : {m['mae_expected']:.4f} (CV Benchmark: 3.5343)")
        print(f"MAE (Predicted Pos)   : {m['mae_predicted']:.4f}")
        print(f"Spearman Rank Corr    : {m['spearman_rank_corr']:.4f}")
        print(f"Top 10 Points Overlap : {m['top10_overlap']} / 10 ({m['top10_overlap']*10:.1f}%)")
        print(f"Top 3 Podium Overlap  : {m['top3_overlap']} / 3 ({m['top3_overlap']/3*100:.1f}%)")
        print(f"Podium Brier Score    : {m['podium_brier_score']:.4f}")
        print("=" * 80)

    print(f"Reports and full predictions written to: {results['output_dir']}\n")


def cmd_evaluate(args):
    """Executes model evaluation command."""
    pipeline = RacePredictionPipeline(root_dir=ROOT)
    track_key = args.track.lower().strip()
    tracks = pipeline.tracks_config.get("tracks", {})
    if track_key not in tracks:
        raise KeyError(f"Track '{track_key}' not found in tracks.json.")

    track_info = tracks[track_key]
    target_round = track_info.get(f"round_{args.year}")
    if target_round is None:
        if args.year == 2026:
            target_round = track_info.get("round_2026")
        else:
            raise ValueError(
                f"Calendar round for track '{track_key}' is not configured for year {args.year} in tracks.json. "
                f"Configured calendar is for 2026."
            )

    train_rounds = [r for r in range(1, target_round)]

    logger.info(f"Evaluating models across completed {args.year} rounds: {train_rounds}")
    races_df, qualis_df = pipeline.data_loader.load_season_rounds(args.year, train_rounds)
    sessions_by_round = pipeline.data_loader.load_all_practice_sessions(args.year, train_rounds)

    training_features = pipeline.feature_engineer.build_training_dataset(
        races_df=races_df,
        qualis_df=qualis_df,
        practice_sessions_by_round=sessions_by_round,
        train_rounds=train_rounds
    )

    eval_results = pipeline.model_evaluator.evaluate_walk_forward(training_features)

    print("\n" + "=" * 80)
    print(f"CHRONOLOGICAL WALK-FORWARD MODEL EVALUATION (Rounds 4-{target_round-1})")
    print("=" * 80)
    print(f"{'Model Configuration':<28} {'MAE':<8} {'MedAE':<8} {'RMSE':<8} {'Spearman':<10}")
    print("-" * 80)

    for row in eval_results["ensemble_comparisons"]:
        print(
            f"{row['model_name']:<28} "
            f"{row['mae']:<8.3f} "
            f"{row['medae']:<8.3f} "
            f"{row['rmse']:<8.3f} "
            f"{row['spearman']:<10.3f}"
        )
    print("=" * 80)
    print(f"Empirical DNF Hazard Rate: {eval_results['empirical_dnf_rate']*100:.1f}%\n")


def cmd_validate(args):
    """Executes validation and leakage audit."""
    pipeline = RacePredictionPipeline(root_dir=ROOT)
    track_key = args.track.lower().strip()
    tracks = pipeline.tracks_config.get("tracks", {})
    if track_key not in tracks:
        raise KeyError(f"Track '{track_key}' not found in tracks.json.")

    track_info = tracks[track_key]
    target_round = track_info.get(f"round_{args.year}")
    if target_round is None:
        if args.year == 2026:
            target_round = track_info.get("round_2026")
        else:
            raise ValueError(
                f"Calendar round for track '{track_key}' is not configured for year {args.year} in tracks.json. "
                f"Configured calendar is for 2026."
            )

    train_rounds = [r for r in range(1, target_round)]

    races_df, qualis_df = pipeline.data_loader.load_season_rounds(args.year, train_rounds)
    sessions_by_round = pipeline.data_loader.load_all_practice_sessions(args.year, train_rounds)
    weekend_sessions = pipeline.data_loader.load_weekend_sessions(track_key, args.year, stage=args.stage)
    quali_df = None
    if args.stage in ["after_qualifying", "post_race"]:
        quali_df = pipeline.data_loader.load_target_qualifying(track_key, args.year)
    if quali_df is None or quali_df.empty:
        logger.info(f"Target qualifying not yet completed/observed for stage '{args.stage}'. Generating provisional qualifying...")
        provisional_quali_meta = pipeline.predict_qualifying(track_key=track_key, year=args.year, stage=args.stage)
        quali_df = provisional_quali_meta["predictions"][["Abbreviation", "TeamName", "PredictedPosition"]].copy()
        quali_df.rename(columns={"PredictedPosition": "Position"}, inplace=True)
        quali_df["QualifyingPosition"] = quali_df["Position"]

    train_feats = pipeline.feature_engineer.build_training_dataset(
        races_df=races_df,
        qualis_df=qualis_df,
        practice_sessions_by_round=sessions_by_round,
        train_rounds=train_rounds
    )
    penalties = track_info.get(
        f"known_penalties_{args.year}",
        track_info.get("known_penalties_2026" if args.year == 2026 else None, {})
    )
    target_feats = pipeline.feature_engineer.build_target_dataset(
        target_round=target_round,
        track_name=track_info.get("display_name", track_key.title()),
        quali_df=quali_df,
        practice_sessions=weekend_sessions,
        season_races_df=races_df,
        season_qualis_df=qualis_df,
        penalties=penalties
    )

    report = pipeline.data_validator.audit_leakage(train_feats, target_feats, target_round)

    display_title = tracks[track_key].get("display_name", track_key.title()).upper()
    print("\n" + "=" * 80)
    print(f"DATA VALIDATION & LEAKAGE AUDIT: {display_title} {args.year}")

    print("=" * 80)
    print(f"Status: {'PASSED (Zero Leakage)' if report['is_clean'] else 'FAILED'}")
    print(f"Training Samples: {report['training_samples']}")
    print(f"Target Lineup:    {report['target_drivers']} drivers")
    if report["violations"]:
        print("\nVIOLATIONS FOUND:")
        for v in report["violations"]:
            print(f"  - {v}")
    else:
        print("All integrity and anti-leakage checks passed successfully.")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="F1 Unified Race Prediction Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=False)

    # Interactive Menu Command
    subparsers.add_parser("menu", help="Launch interactive terminal menu")

    # Predict Command
    predict_parser = subparsers.add_parser("predict", help="Run race prediction pipeline")
    predict_parser.add_argument("--track", required=True, help="Circuit identifier (e.g. sepang, baku, monza)")
    predict_parser.add_argument("--year", type=int, default=2026, help="Season championship year")
    predict_parser.add_argument(
        "--stage",
        default="after_qualifying",
        choices=DataLoader.STAGES,
        help="Prediction timeline stage cutoff"
    )
    predict_parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top finishing positions to display (default: 10)"
    )

    # Evaluate Command
    eval_parser = subparsers.add_parser("evaluate", help="Run chronological model evaluation")
    eval_parser.add_argument("--track", required=True, help="Target circuit identifier")
    eval_parser.add_argument("--year", type=int, default=2026, help="Season championship year")

    # Validate Command
    val_parser = subparsers.add_parser("validate", help="Validate dataset and run leakage checks")
    val_parser.add_argument("--track", required=True, help="Target circuit identifier")
    val_parser.add_argument("--year", type=int, default=2026, help="Season championship year")
    val_parser.add_argument(
        "--stage",
        default="after_qualifying",
        choices=DataLoader.STAGES,
        help="Prediction stage"
    )

    args = parser.parse_args()

    if args.command is None or args.command == "menu":
        from src.cli.menu import InteractiveMenu
        menu = InteractiveMenu(root_dir=ROOT)
        try:
            menu.run()
        except SystemExit as se:
            sys.exit(se.code if se.code is not None else 0)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "validate":
        cmd_validate(args)


if __name__ == "__main__":
    main()
