"""Validation helpers for ApexGrid-exported session datasets."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd


DRIVER_COLUMNS = {
    "Abbreviation",
    "Driver",
    "FullName",
    "BroadcastName",
    "DriverId",
    "DriverNumber",
}

PRACTICE_SLUGS = {"fp1", "fp2", "fp3"}
RESULT_SLUGS = {
    "qualifying",
    "sprint_qualifying",
    "sprint_race",
    "race",
}


def validate_session_dataframe(
    df: Optional[pd.DataFrame],
    session_slug: str,
) -> Tuple[bool, List[str]]:
    """Validate exported session content and return actionable errors."""
    errors: List[str] = []
    slug = str(session_slug or "").strip().lower()

    if df is None or df.empty:
        return False, ["Dataset is empty."]

    driver_columns = DRIVER_COLUMNS.intersection(df.columns)
    if not driver_columns:
        errors.append("Missing driver identifier column.")
    else:
        has_driver = pd.Series(False, index=df.index)
        for column in driver_columns:
            values = df[column].fillna("").astype(str).str.strip()
            has_driver |= values.ne("")
        if not has_driver.any():
            errors.append("Dataset contains no usable driver identifiers.")

    if slug in PRACTICE_SLUGS:
        if not {"LapTimeSeconds", "LapTime", "LapNumber"}.intersection(df.columns):
            errors.append("Practice dataset requires LapTimeSeconds, LapTime, or LapNumber.")
    elif slug in RESULT_SLUGS:
        result_columns = {
            "Position",
            "ClassifiedPosition",
            "ActualFinish",
            "GridPosition",
            "Status",
        }
        if not result_columns.intersection(df.columns):
            errors.append("Result dataset requires a position, grid, or status column.")
        else:
            usable_result = pd.Series(False, index=df.index)
            for column in result_columns.intersection(df.columns):
                values = df[column].fillna("").astype(str).str.strip()
                usable_result |= values.ne("")
            if not usable_result.any():
                errors.append("Dataset contains no usable result observations.")
    else:
        errors.append(f"Unsupported session type '{slug}'.")

    return not errors, errors


def validate_session_csv(
    file_path: Union[str, Path],
    session_slug: str,
) -> Tuple[bool, List[str]]:
    """
    Safely inspect a session CSV file on disk and validate its schema.
    Returns (True, []) if valid, or (False, [error_messages]) if missing, empty, corrupt, or invalid.
    """
    path = Path(file_path)
    if not path.exists():
        return False, [f"File not found: {path}"]
    if not path.is_file():
        return False, [f"Path is not a regular file: {path}"]
    try:
        if path.stat().st_size == 0:
            return False, [f"File is empty (0 bytes): {path.name}"]
    except OSError as e:
        return False, [f"Unable to read file metadata for {path.name}: {e}"]

    try:
        from apexgrid.data.data_loader import DataLoader
        df = DataLoader.safe_read_csv(path, normalize=False)
    except Exception:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            return False, [f"Failed to parse CSV {path.name}: {e}"]

    if df is None or df.empty:
        return False, [f"Dataset in {path.name} is empty or contains no parseable records."]

    return validate_session_dataframe(df, session_slug)

