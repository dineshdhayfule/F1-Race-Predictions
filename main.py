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
    results = pipeline.run(
        track_key=args.track,
        year=args.year,
        stage=args.stage
    )

    df = results["predictions"]
    track_name = results["track_name"]

    print("\n" + "=" * 80)
    print(f"PREDICTION RESULTS: {track_name.upper()} {args.year} [{args.stage.upper()}]")
    print(f"Selected Model   : {results.get('selected_model', 'N/A')}")
    print(f"Selection Method : {results.get('selection_method', 'Chronological Walk-Forward Validation')}")
    print(f"Selected MAE     : {results.get('selected_mae', 'N/A')}")
    print("=" * 80)
    print(f"{'Pos':<4} {'Driver':<5} {'Team':<22} {'Grid':<5} {'ExpFinish':<10} {'Win%':<7} {'Podium%':<9} {'Points%':<9}")
    print("-" * 80)

    for _, r in df.iterrows():
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
    print(f"Full results & markdown report written to: {results['output_dir']}\n")


def cmd_evaluate(args):
    """Executes model evaluation command."""
    pipeline = RacePredictionPipeline(root_dir=ROOT)
    track_key = args.track.lower().strip()
    tracks = pipeline.tracks_config.get("tracks", {})
    if track_key not in tracks:
        raise KeyError(f"Track '{track_key}' not found in tracks.json.")

    target_round = tracks[track_key]["round_2026"]
    train_rounds = [r for r in range(1, target_round)]

    logger.info(f"Evaluating models across completed 2026 rounds: {train_rounds}")
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

    target_round = tracks[track_key]["round_2026"]
    train_rounds = [r for r in range(1, target_round)]

    races_df, qualis_df = pipeline.data_loader.load_season_rounds(args.year, train_rounds)
    sessions_by_round = pipeline.data_loader.load_all_practice_sessions(args.year, train_rounds)
    weekend_sessions = pipeline.data_loader.load_weekend_sessions(track_key, args.year, stage=args.stage)
    quali_df = pipeline.data_loader.load_target_qualifying(track_key, args.year)

    train_feats = pipeline.feature_engineer.build_training_dataset(
        races_df=races_df,
        qualis_df=qualis_df,
        practice_sessions_by_round=sessions_by_round,
        train_rounds=train_rounds
    )
    target_feats = pipeline.feature_engineer.build_target_dataset(
        target_round=target_round,
        track_name=tracks[track_key].get("display_name", track_key.title()),
        quali_df=quali_df,
        practice_sessions=weekend_sessions,
        season_races_df=races_df,
        season_qualis_df=qualis_df,
        penalties=tracks[track_key].get("known_penalties_2026", {})
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
    subparsers = parser.add_subparsers(dest="command", required=True)

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

    if args.command == "predict":
        cmd_predict(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "validate":
        cmd_validate(args)


if __name__ == "__main__":
    main()
