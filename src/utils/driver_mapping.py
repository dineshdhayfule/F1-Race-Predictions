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
    "MAS": {"full_name": "Felipe Massa", "number": 19, "default_team": "Williams"},
    "ROS": {"full_name": "Nico Rosberg", "number": 6, "default_team": "Mercedes"},
    "BUT": {"full_name": "Jenson Button", "number": 22, "default_team": "McLaren"},
    "GRO": {"full_name": "Romain Grosjean", "number": 8, "default_team": "Haas F1 Team"},
    "KUB": {"full_name": "Robert Kubica", "number": 88, "default_team": "Alfa Romeo"},
    "KVY": {"full_name": "Daniil Kvyat", "number": 26, "default_team": "AlphaTauri"},
    "NAS": {"full_name": "Felipe Nasr", "number": 12, "default_team": "Sauber"},
    "PAL": {"full_name": "Jolyon Palmer", "number": 30, "default_team": "Renault"},
    "ERI": {"full_name": "Marcus Ericsson", "number": 9, "default_team": "Sauber"},
    "WEH": {"full_name": "Pascal Wehrlein", "number": 94, "default_team": "Sauber"},
    "GUT": {"full_name": "Esteban Gutierrez", "number": 21, "default_team": "Haas F1 Team"},
    "HAR": {"full_name": "Brendon Hartley", "number": 28, "default_team": "Toro Rosso"},
    "VAN": {"full_name": "Stoffel Vandoorne", "number": 2, "default_team": "McLaren"},
    "SIR": {"full_name": "Sergey Sirotkin", "number": 35, "default_team": "Williams"},
    "HIR": {"full_name": "Ryo Hirakawa", "number": 98, "default_team": "Haas F1 Team"},
}

FIA_COURSE_VEHICLES = {
    "SC1", "SC2", "SC3", "SC4", "SC5", "SC",
    "MC1", "MC2", "MC3", "MC4", "MC5", "MC",
    "FM1", "FM2", "FM3", "FM4", "FM5", "FM",
    "TC1", "TC2", "TC3", "TC4", "TC5", "TC",
}


def is_course_vehicle(item) -> bool:
    """
    Identifies whether an entry represents an FIA course/service vehicle
    (Safety Car, Medical Car, Fast Master / Test Car) rather than an entered F1 driver.
    """
    if isinstance(item, (str, bytes)):
        code = str(item).strip().upper()
        if code in FIA_COURSE_VEHICLES:
            return True
        if (code.startswith("SC") or code.startswith("MC") or code.startswith("FM") or code.startswith("TC")) and len(code) <= 4:
            if any(ch.isdigit() for ch in code):
                return True
        return False
    elif isinstance(item, (pd.Series, dict)):
        abbr = str(item.get("Abbreviation", "")).strip().upper()
        bcast = str(item.get("BroadcastName", "")).strip().upper()
        full = str(item.get("FullName", "")).strip().upper()
        team = item.get("TeamName")
        num = item.get("DriverNumber")

        if is_course_vehicle(abbr):
            return True
        if any(kw in bcast for kw in ["SAFETY CAR", "MEDICAL CAR", "TEST CAR", "COURSE CAR"]):
            return True
        if any(kw in full for kw in ["SAFETY CAR", "MEDICAL CAR", "TEST CAR"]):
            return True
        if full == "NONE NONE" and (pd.isna(team) or str(team).strip().lower() in ["nan", "none", "", "null"]):
            return True
        if pd.isna(team) or str(team).strip().lower() in ["nan", "none", "", "null"]:
            if pd.notna(num):
                try:
                    if int(num) > 100:
                        return True
                except (ValueError, TypeError):
                    pass
            if is_course_vehicle(abbr):
                return True
        return False
    return False


def filter_competing_drivers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters DataFrame to retain only genuine competing Formula 1 drivers,
    filtering out FIA safety cars, medical cars, test cars, and non-driver course vehicles.
    If course vehicles were intercalated into official classification, re-aligns finishing Position
    to preserve contiguous finishing order for all valid competing drivers.
    """
    if df is None or df.empty:
        return df

    mask = ~df.apply(is_course_vehicle, axis=1)
    cleaned = df[mask].copy()

    # If non-driver rows were dropped and Position exists, re-index non-NaN positions
    if len(cleaned) < len(df) and "Position" in cleaned.columns:
        pos_num = pd.to_numeric(cleaned["Position"], errors="coerce")
        has_pos = pos_num.notna()
        if has_pos.any():
            cleaned.loc[has_pos, "Position"] = range(1, int(has_pos.sum()) + 1)

    return cleaned

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
    Resolves from non-empty 'Abbreviation', 'Driver', 'FullName', 'BroadcastName', 'DriverId', 'driver', or 'code'.
    Treats empty strings, whitespace-only, and NaN abbreviations as missing values,
    falling back row-by-row to reliable available identifiers (FullName, DriverId, etc.).
    Preserves modern driver catalog codes and provides stable, collision-free identifiers
    for historical drivers without inventing arbitrary or conflicting 3-letter codes.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    candidates = ["Abbreviation", "Driver", "FullName", "BroadcastName", "DriverId", "driver", "code"]
    available_candidates = [c for c in candidates if c in out.columns]

    if not available_candidates:
        raise ValueError(f"No driver identifier column found in DataFrame. Available: {list(out.columns)}")

    resolved_abbr = pd.Series("", index=out.index, dtype=str)
    for cand in available_candidates:
        cand_vals = out[cand].fillna("").astype(str).str.strip()
        mask_empty = resolved_abbr.eq("") | resolved_abbr.str.upper().isin(["NAN", "NONE"])
        if mask_empty.any():
            valid_cand = cand_vals.ne("") & ~cand_vals.str.upper().isin(["NAN", "NONE"])
            fill_mask = mask_empty & valid_cand
            resolved_abbr.loc[fill_mask] = cand_vals.loc[fill_mask]

    def _map_identifier(val: str) -> str:
        if not val or val.upper() in ["NAN", "NONE"]:
            return ""
        text = str(val).strip().upper()
        # If in modern alias catalog, map to canonical abbreviation (e.g. LEWIS HAMILTON -> HAM)
        if text in NAME_TO_CODE:
            return NAME_TO_CODE[text]
        # Otherwise keep the stable uppercase identifier without inventing conflicting codes
        return text

    out["Abbreviation"] = resolved_abbr.map(_map_identifier)
    out = filter_competing_drivers(out)
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

