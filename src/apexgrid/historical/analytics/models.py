"""
ApexGrid — Historical Analytics Data Models
============================================
Strongly-typed, immutable dataclasses for Formula 1 historical analytics across 1950–2026.
Fetch-only: All calculations and summaries are retained strictly in process memory.
Zero disk persistence.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any


@dataclass(frozen=True)
class DriverSeasonBreakdown:
    """Historical driver performance within a single championship season."""
    season: int
    constructor_name: str
    races: int
    starts: int
    classified_finishes: int
    wins: int
    podiums: int
    poles: Optional[int]  # None if qualifying data unavailable for season (pre-1994) or API error
    points: float
    dnfs: int
    best_finish: Optional[int]
    avg_finish: Optional[float]
    championship_rank: Optional[int] = None
    is_provisional: bool = False
    running_finishes: int = 0
    classified_retirements: int = 0
    unclassified_retirements: int = 0
    non_starts: int = 0


@dataclass(frozen=True)
class DriverCareerStats:
    """Aggregated career statistics for a Formula 1 driver."""
    driver_id: str
    driver_name: str
    nationality: Optional[str]
    seasons_active: List[int]
    total_entries: int  # Grand Prix result records (appearances) in provider archive
    total_starts: int  # Grand Prix starts (entries excluding non-starters/withdrawals)
    classified_finishes: int  # Total officially classified finishes (running finishes + classified retirements)
    wins: int
    podiums: int
    qualifying_session_p1s: Optional[int]  # Complete career qualifying session P1s: None if career has pre-1994 seasons (incomplete coverage) or API error
    known_covered_poles: Optional[int] = None  # Verified qualifying P1 sessions in covered seasons (1994+)
    poles_coverage_note: str = ""
    total_points: float = 0.0
    avg_finish_position: Optional[float] = None
    dnf_count: int = 0  # Total retirements (classified retirements + unclassified retirements)
    running_finishes: int = 0  # Cars running at the chequered flag on lead lap or laps down
    classified_retirements_count: int = 0  # Officially classified by provider despite retiring before flag (provider classification, not independently calculated distance)
    unclassified_retirements_count: int = 0  # Retired before flag without numerical classification
    non_starts_count: int = 0  # Did not start / withdrew before race start
    seasons_without_recorded_entries: List[int] = field(default_factory=list)
    retirement_reasons: Dict[str, int] = field(default_factory=dict)
    seasons_breakdown: List[DriverSeasonBreakdown] = field(default_factory=list)
    coverage_notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class TeammateComparisonStats:
    """Head-to-head comparison between two drivers in the same constructor."""
    driver1_id: str
    driver1_name: str
    driver2_id: str
    driver2_name: str
    constructor_name: str
    season: Optional[int]  # Specific season or None for career aggregate as teammates
    races_cohabited: int
    both_finished_count: int
    race_h2h: Tuple[int, int]  # (d1_ahead, d2_ahead)
    qualifying_sessions_compared: int
    qualifying_h2h: Tuple[int, int]  # (d1_ahead, d2_ahead)
    qualifying_coverage_note: str
    wins_comparison: Tuple[int, int]
    podiums_comparison: Tuple[int, int]
    points_comparison: Tuple[float, float]
    dnfs_comparison: Tuple[int, int]
    notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class QualifyingAnalyticsSummary:
    """Aggregated qualifying analytics for a driver, season, or championship."""
    driver_id: Optional[str]
    driver_name: Optional[str]
    season: Optional[int]
    total_sessions: int
    pole_positions: int
    front_row_starts: int
    top_10_starts: int
    avg_qualifying_position: Optional[float]
    formats_encountered: List[str]
    position_distribution: Dict[str, int]  # e.g. {"P1": 5, "P2-P3": 10, ...}
    qualifying_to_grid_deltas: List[int]
    avg_grid_drop_or_gain: Optional[float]
    coverage_limitations: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RacePerformanceSummary:
    """Aggregated race performance statistics and trends."""
    driver_id: Optional[str]
    driver_name: Optional[str]
    season: Optional[int]
    total_races: int
    starts: int
    classified_finishes: int
    wins: int
    podiums: int
    points_finishes: int
    total_points: float
    dnfs: int
    dnf_rate_pct: float
    positions_gained_total: Optional[int]
    avg_positions_gained: Optional[float]
    valid_grid_starts_count: int
    finishing_distribution: Dict[str, int]
    retirement_breakdown: Dict[str, int] = field(default_factory=dict)
    coverage_notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class CircuitAnalyticsSummary:
    """Historical Grand Prix statistics hosted at a specific circuit."""
    circuit_id: str
    circuit_name: str
    locality: Optional[str]
    country: Optional[str]
    total_races_hosted: int
    first_gp_year: Optional[int]
    last_gp_year: Optional[int]
    top_winning_drivers: List[Tuple[str, int]]
    top_winning_constructors: List[Tuple[str, int]]
    top_podium_drivers: List[Tuple[str, int]]
    recent_winners: List[Tuple[int, str, str]]  # (year, driver_name, constructor_name)
    coverage_notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SeasonAnalyticsSummary:
    """Comprehensive championship season analytical summary."""
    season: int
    total_scheduled_rounds: int
    completed_rounds: int
    is_in_progress: bool
    status_display: str
    participating_drivers_count: int
    participating_constructors_count: int
    top_drivers_standings: List[Tuple[int, str, str, float, int]]  # (rank, name, team, pts, wins)
    top_constructors_standings: List[Tuple[int, str, float, int]]  # (rank, team, pts, wins)
    qualifying_coverage_summary: str
    race_results_coverage_summary: str
    coverage_notes: List[str] = field(default_factory=list)
