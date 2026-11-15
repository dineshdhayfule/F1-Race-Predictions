"""
ApexGrid — Historical F1 Data Models
=====================================
Strongly-typed, immutable dataclasses for Formula 1 historical entities (1950–2026).
Enforces nullable grid positions, safe positions-gained calculation, float points for
fractional awards (e.g. 1954 8.14 pts), raw source status preservation, and in-progress
season identification.

Fetch-only: In-memory representations only; zero disk persistence.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List


def calculate_positions_gained(grid_pos: Optional[int], finish_pos: Optional[int]) -> Optional[int]:
    """
    Safely calculates positions gained or lost during a race.
    Never calculates positions gained when grid data or finish classification is missing,
    null, or non-standard (<= 0).

    Returns:
        Optional[int]: Positive for positions gained, negative for positions lost,
                       None if either position is invalid or unclassified.
    """
    if grid_pos is None or grid_pos <= 0:
        return None
    if finish_pos is None or finish_pos <= 0:
        return None
    return grid_pos - finish_pos


def interpret_grid_position(raw_grid: Any) -> Tuple[Optional[int], str]:
    """
    Interprets raw grid value from source API.
    A Jolpica grid value of 0 must NOT automatically be labelled a pit-lane start.
    Preserves raw value and returns an appropriate unknown/non-standard status unless
    confirmed by another reliable source.

    Returns:
        Tuple[Optional[int], str]: (parsed_grid_int_or_None, status_display_string)
    """
    if raw_grid is None or str(raw_grid).strip() == "":
        return None, "Grid position unrecorded"

    raw_str = str(raw_grid).strip()
    try:
        val = int(raw_str)
        if val > 0:
            return val, f"Grid P{val}"
        elif val == 0:
            # Value of 0 in Jolpica means unclassified on grid or non-standard start (pit lane, failed to start, reserve)
            return None, "Unclassified / Non-standard Grid Position (0)"
        else:
            return None, f"Non-standard Grid Position ({raw_str})"
    except ValueError:
        return None, f"Non-numeric Grid Position ({raw_str})"


@dataclass(frozen=True)
class HistoricalCircuit:
    """
    Provider-supplied circuit metadata.
    Contains ONLY fields returned by the historical provider.
    Unsupported attributes (corners, track direction, lap records) are omitted.
    Missing circuit names remain None (zero synthetic generation).
    """
    circuit_id: str
    name: Optional[str] = None
    locality: Optional[str] = None
    country: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    url: Optional[str] = None


@dataclass(frozen=True)
class HistoricalEvent:
    """Historical Grand Prix event metadata and scheduling status."""
    season: int
    round: int
    event_name: str
    circuit_id: str
    circuit_name: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    url: Optional[str] = None
    is_completed: bool = False
    is_upcoming: bool = False
    is_cancelled: bool = False
    status_label: str = "Scheduled"


@dataclass(frozen=True)
class SeasonCalendar:
    """Complete season schedule and progression."""
    year: int
    total_rounds: int
    completed_rounds: int  # Confirmed completed rounds (0 in calendar-only queries)
    upcoming_rounds: int
    past_scheduled_rounds: int = 0
    is_in_progress: bool = False
    events: List[HistoricalEvent] = field(default_factory=list)
    status_display: str = ""


@dataclass(frozen=True)
class CircuitMetadata:
    """Historical circuit metadata and statistics."""
    circuit_id: str
    circuit_name: str
    locality: Optional[str] = None
    country: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    wiki_url: Optional[str] = None
    # Modern track properties (only populated if circuit matches config/tracks.json)
    track_length_km: Optional[float] = None
    circuit_type: Optional[str] = None
    overtaking_ease: Optional[float] = None
    # Derived historical statistics (computed in RAM from race results)
    total_grands_prix_hosted: Optional[int] = None
    first_gp_year: Optional[int] = None
    last_gp_year: Optional[int] = None
    most_successful_driver: Optional[Tuple[str, int]] = None
    most_successful_constructor: Optional[Tuple[str, int]] = None


@dataclass(frozen=True)
class QualifyingResultEntry:
    """Official qualifying session entry."""
    position: int
    driver_id: str
    driver_name: str
    driver_code: Optional[str]
    constructor_id: str
    constructor_name: str
    q1: Optional[str] = None
    q2: Optional[str] = None
    q3: Optional[str] = None
    best_lap_time: Optional[str] = None
    gap_to_pole: Optional[str] = None
    session_format: str = "Knockout"  # "Knockout", "Qualifying Best (QB)", "Single Lap Shootout (QO)", "Qualifying Aggregate (QA)", etc.
    session_code: Optional[str] = None  # e.g. "QB", "QO", "QA", "Q"


@dataclass(frozen=True)
class StartingGridResultEntry:
    """Official starting grid entry."""
    grid_position: Optional[int]
    raw_grid_value: str
    qualifying_position: Optional[int]
    driver_id: str
    driver_name: str
    constructor_name: str
    grid_status: str
    source_notes: str


@dataclass(frozen=True)
class RaceResultEntry:
    """Official race finishing classification entry."""
    position: Optional[int]
    position_text: str
    driver_id: str
    driver_name: str
    driver_number: Optional[str]
    constructor_id: str
    constructor_name: str
    grid_position: Optional[int]
    raw_grid_value: str
    positions_gained: Optional[int]
    race_time_or_gap: Optional[str]
    laps_completed: int
    points: float
    status: str
    fastest_lap_rank: Optional[int] = None
    fastest_lap_time: Optional[str] = None


@dataclass(frozen=True)
class SprintResultEntry:
    """Official sprint session classification entry."""
    position: Optional[int]
    position_text: str
    driver_id: str
    driver_name: str
    constructor_name: str
    grid_position: Optional[int]
    laps_completed: int
    points: float
    status: str
    race_time_or_gap: Optional[str] = None


@dataclass(frozen=True)
class ChampionshipStandingEntry:
    """Championship standings entry for drivers or constructors."""
    rank: int
    entity_id: str
    entity_name: str
    nationality: Optional[str]
    constructor_name: Optional[str]
    points: float
    wins: int
    standing_type: str = "driver"  # "driver" or "constructor"


@dataclass(frozen=True)
class SeasonSummary:
    """Season calendar and championship completion status."""
    year: int
    total_rounds: int
    completed_rounds: int
    is_in_progress: bool
    status_display: str

    @classmethod
    def create(cls, year: int, total_rounds: int, completed_rounds: int) -> "SeasonSummary":
        is_in_progress = (completed_rounds < total_rounds)
        if total_rounds == 0:
            status = f"Season {year} (Calendar Pending)"
        elif is_in_progress:
            status = f"Season {year} (In-Progress / Provisional: {completed_rounds}/{total_rounds} Rounds Completed)"
        else:
            status = f"Season {year} (Completed Championship: {total_rounds} Rounds)"
        return cls(
            year=year,
            total_rounds=total_rounds,
            completed_rounds=completed_rounds,
            is_in_progress=is_in_progress,
            status_display=status
        )
