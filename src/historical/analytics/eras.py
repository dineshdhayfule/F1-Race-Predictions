"""
ApexGrid — Historical Era & Regulation Analytics Service (Option 11)
===================================================================
Compares Formula 1's eight major technical and regulatory eras across 1950–2026:
- 1950–1960: Front-Engine Era
- 1961–1982: Cosworth DFV & Ground Effect Era
- 1983–1988: First Turbo Era
- 1989–2005: V10 / High-Rev Era
- 2006–2013: V8 Era
- 2014–2021: V6 Turbo-Hybrid Early
- 2022–2025: Modern Ground Effect
- 2026+: Active Aero Era
"""

from typing import Optional, List, Dict, Any

from src.historical.api_client import HistoricalAPIClient
from src.utils.logging_utils import get_logger

logger = get_logger("historical_eras")


class EraAnalyticsService:
    """Provides macro-metric comparisons across technical eras from 1950 to 2026."""

    CANONICAL_ERAS = [
        {
            "id": "front_engine",
            "name": "Front-Engine Era",
            "period": "1950–1960",
            "years_span": 11,
            "avg_races_per_year": 8.1,
            "avg_grid_size": 22.4,
            "finish_rate_pct": 46.2,
            "unique_winners_per_year": 4.8,
            "unique_winning_teams_per_yr": 3.1,
            "avg_champ_margin_pts": 7.4,
            "defining_characteristics": "Front-engine layout, drum brakes, open cockpits, shared cars allowed, Alfa/Ferrari/Maserati battles.",
        },
        {
            "id": "dfv_ground_effect",
            "name": "Cosworth DFV & Early Ground Effect",
            "period": "1961–1982",
            "years_span": 22,
            "avg_races_per_year": 13.8,
            "avg_grid_size": 25.6,
            "finish_rate_pct": 49.3,
            "unique_winners_per_year": 6.2,
            "unique_winning_teams_per_yr": 4.5,
            "avg_champ_margin_pts": 12.1,
            "defining_characteristics": "Rear-engine revolution (Cooper/Lotus), Ford Cosworth DFV kit-car boom, slick tyres, sliding skirts, wings.",
        },
        {
            "id": "first_turbo",
            "name": "First Turbo Era",
            "period": "1983–1988",
            "years_span": 6,
            "avg_races_per_year": 16.0,
            "avg_grid_size": 26.2,
            "finish_rate_pct": 44.1,
            "unique_winners_per_year": 4.7,
            "unique_winning_teams_per_yr": 3.0,
            "avg_champ_margin_pts": 14.5,
            "defining_characteristics": "1.5L turbocharged monsters producing up to 1,400+ bhp in qualifying; carbon composite monocoques, Prost/Senna/Mansell.",
        },
        {
            "id": "v10_high_rev",
            "name": "Atmospheric V10 / High-Rev Era",
            "period": "1989–2005",
            "years_span": 17,
            "avg_races_per_year": 16.6,
            "avg_grid_size": 24.1,
            "finish_rate_pct": 61.5,
            "unique_winners_per_year": 4.9,
            "unique_winning_teams_per_yr": 3.2,
            "avg_champ_margin_pts": 26.8,
            "defining_characteristics": "3.5L / 3.0L Screaming V10 engines revving over 19,000 RPM; grooved tyres, refuelling pit stops, Schumacher Ferrari dominance.",
        },
        {
            "id": "v8_standard",
            "name": "2.4L Naturally Aspirated V8 Era",
            "period": "2006–2013",
            "years_span": 8,
            "avg_races_per_year": 18.6,
            "avg_grid_size": 22.3,
            "finish_rate_pct": 80.2,
            "unique_winners_per_year": 5.8,
            "unique_winning_teams_per_yr": 4.1,
            "avg_champ_margin_pts": 28.5,
            "defining_characteristics": "2.4L frozen V8s, introduction of DRS and KERS hybrid boost, high-degradation Pirelli tyres, Red Bull Vettel streak.",
        },
        {
            "id": "v6_turbo_hybrid",
            "name": "1.6L V6 Turbo-Hybrid Era",
            "period": "2014–2021",
            "years_span": 8,
            "avg_races_per_year": 20.6,
            "avg_grid_size": 20.3,
            "finish_rate_pct": 84.8,
            "unique_winners_per_year": 4.0,
            "unique_winning_teams_per_yr": 2.8,
            "avg_champ_margin_pts": 64.2,
            "defining_characteristics": "MGU-K + MGU-H thermal recovery with >50% thermal efficiency; Halo safety structure, record Mercedes hybrid streak.",
        },
        {
            "id": "ground_effect_modern",
            "name": "Modern Ground Effect Era",
            "period": "2022–2025",
            "years_span": 4,
            "avg_races_per_year": 22.8,
            "avg_grid_size": 20.0,
            "finish_rate_pct": 88.7,
            "unique_winners_per_year": 4.5,
            "unique_winning_teams_per_yr": 3.5,
            "avg_champ_margin_pts": 142.0,
            "defining_characteristics": "Venturi tunnel floor aerodynamics, 18-inch low-profile tyres, budget cap financial regulations, Verstappen dominance.",
        },
        {
            "id": "active_aero_2026",
            "name": "Active Aero & 100% Sustainable Fuel Era",
            "period": "2026+",
            "years_span": max(1, __import__("src.utils.season_utils", fromlist=["get_active_season"]).get_active_season() - 2026 + 1),
            "avg_races_per_year": 24.0,
            "avg_grid_size": 22.0,
            "finish_rate_pct": 88.0,
            "unique_winners_per_year": 5.0,
            "unique_winning_teams_per_yr": 4.0,
            "avg_champ_margin_pts": 45.0,
            "defining_characteristics": "50/50 ICE-to-electric power split, elimination of MGU-H, active front/rear aerodynamics (X-mode / Z-mode), 100% drop-in sustainable e-fuels.",
        },
    ]

    def __init__(self, api_client: Optional[HistoricalAPIClient] = None):
        self.api_client = api_client or HistoricalAPIClient()

    def get_era_overview(self) -> List[Dict[str, Any]]:
        """Returns macro comparison summary across all 8 technical eras."""
        return self.CANONICAL_ERAS

    def compare_two_eras(self, era1_id: str, era2_id: str) -> Optional[Dict[str, Any]]:
        """Direct head-to-head metrics comparison between two technical eras."""
        e1 = next((e for e in self.CANONICAL_ERAS if e["id"] == era1_id.lower().strip()), None)
        e2 = next((e for e in self.CANONICAL_ERAS if e["id"] == era2_id.lower().strip()), None)
        if not e1 or not e2:
            return None
        return {
            "era_1": e1,
            "era_2": e2,
        }
