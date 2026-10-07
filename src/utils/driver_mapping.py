"""
Driver normalization and identification mapping.
Maintains canonical 3-letter abbreviation, number, and full name mappings.
"""

import pandas as pd


DRIVER_CATALOG = {
    "VER": {"full_name": "Max Verstappen", "number": 3, "default_team": "Red Bull Racing"},
    "HAM": {"full_name": "Lewis Hamilton", "number": 44, "default_team": "Ferrari"},
    "HAD": {"full_name": "Isack Hadjar", "number": 6, "default_team": "Red Bull Racing"},
    "ANT": {"full_name": "Kimi Antonelli", "number": 12, "default_team": "Mercedes"},
    "LEC": {"full_name": "Charles Leclerc", "number": 16, "default_team": "Ferrari"},
    "NOR": {"full_name": "Lando Norris", "number": 1, "default_team": "McLaren"},
    "PIA": {"full_name": "Oscar Piastri", "number": 81, "default_team": "McLaren"},
    "RUS": {"full_name": "George Russell", "number": 63, "default_team": "Mercedes"},
    "GAS": {"full_name": "Pierre Gasly", "number": 10, "default_team": "Alpine"},
    "BOR": {"full_name": "Gabriel Bortoleto", "number": 5, "default_team": "Audi"},
    "LAW": {"full_name": "Liam Lawson", "number": 30, "default_team": "Racing Bulls"},
    "ALO": {"full_name": "Fernando Alonso", "number": 14, "default_team": "Aston Martin"},
    "SAI": {"full_name": "Carlos Sainz", "number": 55, "default_team": "Williams"},
    "STR": {"full_name": "Lance Stroll", "number": 18, "default_team": "Aston Martin"},
    "COL": {"full_name": "Franco Colapinto", "number": 43, "default_team": "Alpine"},
    "LIN": {"full_name": "Arvid Lindblad", "number": 41, "default_team": "Racing Bulls"},
    "HUL": {"full_name": "Nico Hulkenberg", "number": 27, "default_team": "Audi"},
    "BEA": {"full_name": "Oliver Bearman", "number": 87, "default_team": "Haas F1 Team"},
    "OCO": {"full_name": "Esteban Ocon", "number": 31, "default_team": "Haas F1 Team"},
    "ALB": {"full_name": "Alexander Albon", "number": 23, "default_team": "Williams"},
    "BOT": {"full_name": "Valtteri Bottas", "number": 77, "default_team": "Cadillac"},
    "PER": {"full_name": "Sergio Perez", "number": 11, "default_team": "Cadillac"},
    "DOO": {"full_name": "Jack Doohan", "number": 7, "default_team": "Alpine"},
    "TSU": {"full_name": "Yuki Tsunoda", "number": 22, "default_team": "Racing Bulls"},
    "ZHO": {"full_name": "Guanyu Zhou", "number": 24, "default_team": "Kick Sauber"},
    "MAG": {"full_name": "Kevin Magnussen", "number": 20, "default_team": "Haas F1 Team"},
    "SAR": {"full_name": "Logan Sargeant", "number": 2, "default_team": "Williams"},
    "RIC": {"full_name": "Daniel Ricciardo", "number": 3, "default_team": "RB"},
    "VET": {"full_name": "Sebastian Vettel", "number": 5, "default_team": "Aston Martin"},
    "RAI": {"full_name": "Kimi Raikkonen", "number": 7, "default_team": "Alfa Romeo"},
    "GIO": {"full_name": "Antonio Giovinazzi", "number": 99, "default_team": "Alfa Romeo"},
    "MAZ": {"full_name": "Nikita Mazepin", "number": 9, "default_team": "Haas F1 Team"},
    "LAT": {"full_name": "Nicholas Latifi", "number": 6, "default_team": "Williams"},
    "MSC": {"full_name": "Mick Schumacher", "number": 47, "default_team": "Haas F1 Team"},
    "DEV": {"full_name": "Nyck de Vries", "number": 21, "default_team": "AlphaTauri"},
}

# Alias resolution mapping
NAME_TO_CODE = {}
for code, data in DRIVER_CATALOG.items():
    NAME_TO_CODE[code.upper()] = code
    NAME_TO_CODE[data["full_name"].upper()] = code
    parts = data["full_name"].upper().split()
    if len(parts) >= 2:
        NAME_TO_CODE[parts[-1]] = code  # Surname


def normalize_driver(df):
    """
    Ensures a DataFrame has a standard uppercase 'Abbreviation' column.
    Resolves from 'Abbreviation', 'Driver', 'FullName', or 'BroadcastName'.
    """
    out = df.copy()
    col_found = None
    for cand in ["Abbreviation", "Driver", "FullName", "BroadcastName", "driver", "code"]:
        if cand in out.columns:
            col_found = cand
            break

    if col_found is None:
        raise ValueError(f"No driver identifier column found in DataFrame. Available: {list(out.columns)}")

    out["Abbreviation"] = out[col_found].astype(str).str.upper().str.strip()
    # Map if it was a full name or surname
    out["Abbreviation"] = out["Abbreviation"].map(lambda x: NAME_TO_CODE.get(x, x))
    return out


def normalize_driver_code(val: str) -> str:
    """Normalizes a single driver name/abbreviation string to canonical 3-letter code."""
    if pd.isna(val) or val is None:
        return ""
    text = str(val).strip().upper()
    return NAME_TO_CODE.get(text, text)


def normalize_driver_series(series: pd.Series) -> pd.Series:
    """Normalizes a pandas Series of driver identifiers to uppercase canonical 3-letter codes."""
    return series.astype(str).str.strip().str.upper().map(lambda x: NAME_TO_CODE.get(x, x))

