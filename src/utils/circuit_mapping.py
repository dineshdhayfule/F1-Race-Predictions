"""
Circuit Mapping and Canonical Identifier Resolution Utility
============================================================
Provides centralized, configuration-driven resolution of F1 circuit keys.
Uses config/tracks.json as the authoritative single source of truth.
Ensures feature modules, simulation models, and prediction pipelines always
receive canonical circuit keys ('baku', 'mexico', 'austin', 'canada', etc.)
regardless of whether the input is a display name, event pattern, slug, or alias.
"""

from pathlib import Path
import json
from typing import Dict, List, Optional, Any
from functools import lru_cache

ROOT_DIR = Path(__file__).resolve().parents[2]
TRACKS_CONFIG_PATH = ROOT_DIR / "config" / "tracks.json"

# Common known aliases mapping to canonical keys in config/tracks.json
CANONICAL_ALIASES: Dict[str, str] = {
    # Short abbreviations / acronyms / circuit nicknames
    "cota": "austin",
    "circuit_of_the_americas": "austin",
    "interlagos": "sao_paulo",
    "autodromo_jose_carlos_pace": "sao_paulo",
    "montreal": "canada",
    "circuit_gilles_villeneuve": "canada",
    "villeneuve": "canada",
    "suzuka": "japan",
    "albert_park": "australia",
    "melbourne": "australia",
    "shanghai": "china",
    "red_bull_ring": "austria",
    "spielberg": "austria",
    "catalunya": "barcelona",
    "hungaroring": "hungary",
    "budapest": "hungary",
    "marina_bay": "singapore",
    "yas_marina": "abu_dhabi",
    "losail": "qatar",
    "lusail": "qatar",
    "jeddah": "jeddah",
    "bahrain": "bahrain",
    "sakhir": "bahrain",
    "malaysia": "sepang",
    "malaysian_grand_prix": "sepang",
}


@lru_cache(maxsize=1)
def load_tracks_config() -> Dict[str, Any]:
    """Loads and caches tracks configuration from config/tracks.json."""
    if TRACKS_CONFIG_PATH.exists():
        with open(TRACKS_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("tracks", {})
    return {}


def get_canonical_circuit_keys() -> List[str]:
    """Returns sorted list of all 23 canonical circuit keys configured in tracks.json."""
    return sorted(list(load_tracks_config().keys()))


def resolve_circuit_key(
    identifier: Optional[str],
    fallback: Optional[str] = None
) -> Optional[str]:
    """
    Resolves any track identifier (canonical key, display name, circuit name,
    event name pattern, snake_case slug, or alias) into its canonical key.

    Parameters
    ----------
    identifier : str, optional
        Input track string (e.g. 'baku', 'Azerbaijan Grand Prix', 'azerbaijan_grand_prix', 'COTA').
    fallback : str, optional
        Fallback key to return if circuit cannot be identified (default: None).

    Returns
    -------
    str or None
        Canonical key from tracks.json, or fallback if unresolved.
    """
    if identifier is None:
        return fallback

    raw = str(identifier).strip()
    if not raw:
        return fallback

    clean_key = raw.lower().replace("-", "_").replace(" ", "_").strip()
    tracks_cfg = load_tracks_config()

    # 1. Exact match against canonical keys in tracks.json
    if clean_key in tracks_cfg:
        return clean_key

    # 2. Match against common aliases
    if clean_key in CANONICAL_ALIASES:
        aliased = CANONICAL_ALIASES[clean_key]
        if aliased in tracks_cfg:
            return aliased

    # 3. Match against display names and circuit names in tracks.json
    clean_spaces = raw.lower().replace("_", " ").strip()
    for k, info in tracks_cfg.items():
        disp_name = str(info.get("display_name", "")).lower().strip()
        circ_name = str(info.get("circuit_name", "")).lower().strip()
        country_name = str(info.get("country", "")).lower().strip()

        # Check exact display_name or slugified display_name
        if clean_spaces == disp_name or clean_key == disp_name.replace(" ", "_"):
            return k

        # Check circuit_name
        if clean_spaces == circ_name or clean_key == circ_name.replace(" ", "_"):
            return k

        # Check patterns declared in event_name_patterns
        patterns = info.get("event_name_patterns", [])
        for pat in patterns:
            pat_clean = str(pat).lower().strip()
            if pat_clean and (clean_spaces == pat_clean or clean_key == pat_clean.replace(" ", "_")):
                return k

        # Substring / token matching if display name contained in identifier
        if disp_name and disp_name in clean_spaces:
            return k

    # 4. Partial / fuzzy pattern match across event_name_patterns
    for k, info in tracks_cfg.items():
        patterns = info.get("event_name_patterns", [])
        for pat in patterns:
            pat_clean = str(pat).lower().strip()
            if pat_clean and (pat_clean in clean_spaces or pat_clean in clean_key):
                return k

    return fallback
