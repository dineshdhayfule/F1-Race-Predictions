"""
Team and constructor normalization utilities.
Handles name variations, title sponsors, and rebrands across seasons.
"""

TEAM_ALIASES = {
    "RED BULL RACING": "Red Bull Racing",
    "RED BULL": "Red Bull Racing",
    "RBR": "Red Bull Racing",
    "FERRARI": "Ferrari",
    "SCUDERIA FERRARI": "Ferrari",
    "MCLAREN": "McLaren",
    "MERCEDES": "Mercedes",
    "MERCEDES-AMG": "Mercedes",
    "ASTON MARTIN": "Aston Martin",
    "ASTON MARTIN ARAMCO": "Aston Martin",
    "ALPINE": "Alpine",
    "ALPINE F1 TEAM": "Alpine",
    "HAAS": "Haas F1 Team",
    "HAAS F1 TEAM": "Haas F1 Team",
    "WILLIAMS": "Williams",
    "WILLIAMS RACING": "Williams",
    "RACING BULLS": "Racing Bulls",
    "RB": "Racing Bulls",
    "ALPHATAURI": "Racing Bulls",
    "SCUDERIA ALPHATAURI": "Racing Bulls",
    "TORO ROSSO": "Racing Bulls",
    "AUDI": "Audi",
    "SAUBER": "Audi",
    "KICK SAUBER": "Audi",
    "ALFA ROMEO": "Audi",
    "CADILLAC": "Cadillac",
}


def normalize_team_name(name):
    """
    Map various team string representations to canonical team name.
    """
    if not name or str(name).strip().lower() in {"nan", "none", ""}:
        return "Unknown"
    cleaned = str(name).strip().upper()
    return TEAM_ALIASES.get(cleaned, str(name).strip())
