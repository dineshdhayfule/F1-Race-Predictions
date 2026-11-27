"""
ApexGrid — Historical F1 Analytics Package (1950–2026)
=======================================================
Fetch-only, memory-only analytics subsystem for historical Formula 1 data.
Zero filesystem persistence.
"""

from apexgrid.historical.models import (
    CircuitMetadata,
    HistoricalCircuit,
    HistoricalEvent,
    SeasonCalendar,
    QualifyingResultEntry,
    StartingGridResultEntry,
    RaceResultEntry,
    SprintResultEntry,
    ChampionshipStandingEntry,
    SeasonSummary,
    calculate_positions_gained,
    interpret_grid_position,
)
from apexgrid.historical.in_memory_cache import InMemoryTTLCache
from apexgrid.historical.rate_limiter import RateLimiter
from apexgrid.historical.api_client import (
    HistoricalAPIClient,
    HistoricalAPIError,
    RateLimitExceededError,
)
from apexgrid.historical.events_service import HistoricalEventsService
from apexgrid.historical.circuits_service import HistoricalCircuitsService
from apexgrid.historical.qualifying_service import HistoricalQualifyingService
from apexgrid.historical.grid_service import HistoricalGridService
from apexgrid.historical.classification_service import HistoricalClassificationService
from apexgrid.historical.analytics import (
    DriverSeasonBreakdown,
    DriverCareerStats,
    TeammateComparisonStats,
    QualifyingAnalyticsSummary,
    RacePerformanceSummary,
    CircuitAnalyticsSummary,
    SeasonAnalyticsSummary,
    DriverAnalyticsService,
    TeammateAnalyticsService,
    QualifyingAnalyticsService,
    RacePerformanceAnalyticsService,
    CircuitAnalyticsService,
    SeasonAnalyticsService,
)

__all__ = [
    "CircuitMetadata",
    "HistoricalCircuit",
    "HistoricalEvent",
    "SeasonCalendar",
    "QualifyingResultEntry",
    "StartingGridResultEntry",
    "RaceResultEntry",
    "SprintResultEntry",
    "ChampionshipStandingEntry",
    "SeasonSummary",
    "calculate_positions_gained",
    "interpret_grid_position",
    "InMemoryTTLCache",
    "RateLimiter",
    "HistoricalAPIClient",
    "HistoricalAPIError",
    "RateLimitExceededError",
    "HistoricalEventsService",
    "HistoricalCircuitsService",
    "HistoricalQualifyingService",
    "HistoricalGridService",
    "HistoricalClassificationService",
    "DriverSeasonBreakdown",
    "DriverCareerStats",
    "TeammateComparisonStats",
    "QualifyingAnalyticsSummary",
    "RacePerformanceSummary",
    "CircuitAnalyticsSummary",
    "SeasonAnalyticsSummary",
    "DriverAnalyticsService",
    "TeammateAnalyticsService",
    "QualifyingAnalyticsService",
    "RacePerformanceAnalyticsService",
    "CircuitAnalyticsService",
    "SeasonAnalyticsService",
]
