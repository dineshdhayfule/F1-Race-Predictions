"""
Automated data and feature validator with strict zero-leakage enforcement.
"""

from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np


class DataValidator:
    """
    Validates integrity of raw and processed datasets before training or prediction.
    """

    FORBIDDEN_PRE_RACE_COLUMNS = [
        "ActualFinish", "Position", "Points", "Status", "Laps", "Time", "ClassifiedPosition"
    ]

    @staticmethod
    def validate_leakage(target_df: pd.DataFrame, stage: str = "after_qualifying") -> Tuple[bool, List[str]]:
        """
        Ensures target features contains ZERO post-race information.
        """
        errors = []
        if stage != "post_race":
            for col in DataValidator.FORBIDDEN_PRE_RACE_COLUMNS:
                if col in target_df.columns:
                    errors.append(f"LEAKAGE ERROR: Post-race column '{col}' detected in pre-race target dataset.")

        return len(errors) == 0, errors

    @staticmethod
    def audit_leakage(
        training_df: pd.DataFrame,
        target_df: pd.DataFrame,
        target_round: int,
        stage: str = "after_qualifying"
    ) -> Dict[str, Any]:
        """
        Comprehensive audit checking both training and target datasets for future or post-race leakage.
        """
        violations = []

        # 1. Target dataset cannot contain post-race outcome columns
        if stage != "post_race":
            for col in DataValidator.FORBIDDEN_PRE_RACE_COLUMNS:
                if col in target_df.columns:
                    violations.append(f"LEAKAGE: Column '{col}' present in target dataset.")

        # 2. Training dataset cannot contain target round or future rounds
        if "Round" in training_df.columns:
            training_rounds = training_df["Round"].dropna().unique()
            if target_round in training_rounds:
                violations.append(f"LEAKAGE: Target Round {target_round} found in training dataset.")
            future_rounds = [r for r in training_rounds if r > target_round]
            if future_rounds:
                violations.append(f"LEAKAGE: Future rounds {future_rounds} found in training dataset.")

        # 3. Rolling history cannot exceed prior rounds
        if "Round" in training_df.columns and "races_completed" in training_df.columns:
            invalid = training_df[training_df["races_completed"] > training_df["Round"] - 1]
            if not invalid.empty:
                violations.append(
                    f"LEAKAGE: {len(invalid)} rows have races_completed exceeding prior completed rounds."
                )

        return {
            "is_clean": len(violations) == 0,
            "violations": violations,
            "training_samples": len(training_df),
            "target_drivers": len(target_df),
        }


    @staticmethod
    def validate_training_dataset(df: pd.DataFrame, expected_grid_size: int = 22) -> Tuple[bool, List[str]]:
        """
        Validates historical training dataset integrity.
        """
        errors = []
        if df.empty:
            return False, ["Training dataset is completely empty."]

        required = ["Round", "Abbreviation", "ActualFinish", "GridPosition"]
        for col in required:
            if col not in df.columns:
                errors.append(f"Missing essential column '{col}' in training dataset.")

        if "Round" in df.columns and "races_completed" in df.columns:
            invalid_form = df[df["races_completed"] > df["Round"] - 1]
            if not invalid_form.empty:
                errors.append(f"LEAKAGE ERROR: {len(invalid_form)} rows have races_completed exceeding prior completed rounds.")

        # Check grid position bounds
        if "GridPosition" in df.columns:
            grid_s = pd.to_numeric(df["GridPosition"], errors="coerce")
            if (grid_s < 1).any() or (grid_s > expected_grid_size).any():
                errors.append(f"GridPosition out of expected range 1..{expected_grid_size}.")

        return len(errors) == 0, errors

    @staticmethod
    def validate_target_grid(df: pd.DataFrame, expected_grid_size: int = 22) -> Tuple[bool, List[str]]:
        """
        Validates that the target grid has exactly expected_grid_size unique drivers and valid 1-22 starting positions.
        """
        errors = []
        if len(df) != expected_grid_size:
            errors.append(f"Driver count is {len(df)}, expected {expected_grid_size}.")

        if "Abbreviation" in df.columns and df["Abbreviation"].nunique() != expected_grid_size:
            errors.append(f"Duplicate drivers found: {df['Abbreviation'].nunique()} unique out of {len(df)}.")

        if "GridPosition" in df.columns:
            grid_vals = sorted(pd.to_numeric(df["GridPosition"], errors="coerce").astype(int).tolist())
            expected_vals = list(range(1, expected_grid_size + 1))
            if grid_vals != expected_vals:
                errors.append(f"Starting grid is not a unique permutation of 1..{expected_grid_size}.")

        return len(errors) == 0, errors
