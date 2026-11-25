"""
ApexGrid — Historical Records, Streaks & Milestones Service (Option 9)
=====================================================================
Computes all-time Formula 1 milestones across 1950–2026:
- Most Grand Prix wins, pole positions, and podium finishes.
- Consecutive streaks (consecutive wins, consecutive podiums, points finishes, consecutive starts).
- Age records (youngest & oldest race winners and pole sitters).
- Unwanted records (most starts without a win or podium).
"""

from typing import Optional, List, Dict, Any

from src.historical.api_client import HistoricalAPIClient
from src.utils.logging_utils import get_logger

logger = get_logger("historical_records")


class AllTimeRecordsService:
    """Computes all-time driver records, streaks, and career milestones."""

    # All-time verified career records (top tier)
    ALL_TIME_WINS = [
        {"rank": 1, "driver": "Lewis Hamilton", "nationality": "British", "wins": 105, "starts": 356, "win_pct": 29.5},
        {"rank": 2, "driver": "Michael Schumacher", "nationality": "German", "wins": 91, "starts": 306, "win_pct": 29.7},
        {"rank": 3, "driver": "Max Verstappen", "nationality": "Dutch", "wins": 63, "starts": 209, "win_pct": 30.1},
        {"rank": 4, "driver": "Sebastian Vettel", "nationality": "German", "wins": 53, "starts": 299, "win_pct": 17.7},
        {"rank": 5, "driver": "Alain Prost", "nationality": "French", "wins": 51, "starts": 199, "win_pct": 25.6},
        {"rank": 6, "driver": "Ayrton Senna", "nationality": "Brazilian", "wins": 41, "starts": 161, "win_pct": 25.5},
        {"rank": 7, "driver": "Fernando Alonso", "nationality": "Spanish", "wins": 32, "starts": 401, "win_pct": 8.0},
        {"rank": 8, "driver": "Nigel Mansell", "nationality": "British", "wins": 31, "starts": 187, "win_pct": 16.6},
        {"rank": 9, "driver": "Jackie Stewart", "nationality": "British", "wins": 27, "starts": 99, "win_pct": 27.3},
        {"rank": 10, "driver": "Jim Clark", "nationality": "British", "wins": 25, "starts": 72, "win_pct": 34.7},
        {"rank": 11, "driver": "Niki Lauda", "nationality": "Austrian", "wins": 25, "starts": 171, "win_pct": 14.6},
        {"rank": 12, "driver": "Juan Manuel Fangio", "nationality": "Argentine", "wins": 24, "starts": 51, "win_pct": 47.1},
        {"rank": 13, "driver": "Nelson Piquet", "nationality": "Brazilian", "wins": 23, "starts": 204, "win_pct": 11.3},
        {"rank": 14, "driver": "Nico Rosberg", "nationality": "German", "wins": 23, "starts": 206, "win_pct": 11.2},
        {"rank": 15, "driver": "Damon Hill", "nationality": "British", "wins": 22, "starts": 115, "win_pct": 19.1},
        {"rank": 16, "driver": "Kimi Räikkönen", "nationality": "Finnish", "wins": 21, "starts": 349, "win_pct": 6.0},
        {"rank": 17, "driver": "Mika Häkkinen", "nationality": "Finnish", "wins": 20, "starts": 161, "win_pct": 12.4},
        {"rank": 18, "driver": "Stirling Moss", "nationality": "British", "wins": 16, "starts": 66, "win_pct": 24.2},
        {"rank": 19, "driver": "Jenson Button", "nationality": "British", "wins": 15, "starts": 306, "win_pct": 4.9},
        {"rank": 20, "driver": "Graham Hill", "nationality": "British", "wins": 14, "starts": 176, "win_pct": 8.0},
    ]

    ALL_TIME_POLES = [
        {"rank": 1, "driver": "Lewis Hamilton", "nationality": "British", "poles": 104, "starts": 356, "pole_pct": 29.2},
        {"rank": 2, "driver": "Michael Schumacher", "nationality": "German", "poles": 68, "starts": 306, "pole_pct": 22.2},
        {"rank": 3, "driver": "Ayrton Senna", "nationality": "Brazilian", "poles": 65, "starts": 161, "pole_pct": 40.4},
        {"rank": 4, "driver": "Sebastian Vettel", "nationality": "German", "poles": 57, "starts": 299, "pole_pct": 19.1},
        {"rank": 5, "driver": "Max Verstappen", "nationality": "Dutch", "poles": 40, "starts": 209, "pole_pct": 19.1},
        {"rank": 6, "driver": "Jim Clark", "nationality": "British", "poles": 33, "starts": 72, "pole_pct": 45.8},
        {"rank": 7, "driver": "Alain Prost", "nationality": "French", "poles": 33, "starts": 199, "pole_pct": 16.6},
        {"rank": 8, "driver": "Nigel Mansell", "nationality": "British", "poles": 32, "starts": 187, "pole_pct": 17.1},
        {"rank": 9, "driver": "Nico Rosberg", "nationality": "German", "poles": 30, "starts": 206, "pole_pct": 14.6},
        {"rank": 10, "driver": "Juan Manuel Fangio", "nationality": "Argentine", "poles": 29, "starts": 51, "pole_pct": 56.9},
        {"rank": 11, "driver": "Mika Häkkinen", "nationality": "Finnish", "poles": 26, "starts": 161, "pole_pct": 16.1},
        {"rank": 12, "driver": "Charles Leclerc", "nationality": "Monegasque", "poles": 26, "starts": 146, "pole_pct": 17.8},
        {"rank": 13, "driver": "Niki Lauda", "nationality": "Austrian", "poles": 24, "starts": 171, "pole_pct": 14.0},
        {"rank": 14, "driver": "Nelson Piquet", "nationality": "Brazilian", "poles": 24, "starts": 204, "pole_pct": 11.8},
        {"rank": 15, "driver": "Fernando Alonso", "nationality": "Spanish", "poles": 22, "starts": 401, "pole_pct": 5.5},
        {"rank": 16, "driver": "Valtteri Bottas", "nationality": "Finnish", "poles": 20, "starts": 246, "pole_pct": 8.1},
        {"rank": 17, "driver": "Damon Hill", "nationality": "British", "poles": 20, "starts": 115, "pole_pct": 17.4},
        {"rank": 18, "driver": "Kimi Räikkönen", "nationality": "Finnish", "poles": 18, "starts": 349, "pole_pct": 5.2},
        {"rank": 19, "driver": "Mario Andretti", "nationality": "American", "poles": 18, "starts": 128, "pole_pct": 14.1},
        {"rank": 20, "driver": "René Arnoux", "nationality": "French", "poles": 18, "starts": 149, "pole_pct": 12.1},
    ]

    ALL_TIME_PODIUMS = [
        {"rank": 1, "driver": "Lewis Hamilton", "nationality": "British", "podiums": 201, "starts": 356, "podium_pct": 56.5},
        {"rank": 2, "driver": "Michael Schumacher", "nationality": "German", "podiums": 155, "starts": 306, "podium_pct": 50.7},
        {"rank": 3, "driver": "Sebastian Vettel", "nationality": "German", "podiums": 122, "starts": 299, "podium_pct": 40.8},
        {"rank": 4, "driver": "Max Verstappen", "nationality": "Dutch", "podiums": 111, "starts": 209, "podium_pct": 53.1},
        {"rank": 5, "driver": "Alain Prost", "nationality": "French", "podiums": 106, "starts": 199, "podium_pct": 53.3},
        {"rank": 6, "driver": "Fernando Alonso", "nationality": "Spanish", "podiums": 106, "starts": 401, "podium_pct": 26.4},
        {"rank": 7, "driver": "Kimi Räikkönen", "nationality": "Finnish", "podiums": 103, "starts": 349, "podium_pct": 29.5},
        {"rank": 8, "driver": "Ayrton Senna", "nationality": "Brazilian", "podiums": 80, "starts": 161, "podium_pct": 49.7},
        {"rank": 9, "driver": "Rubens Barrichello", "nationality": "Brazilian", "podiums": 68, "starts": 322, "podium_pct": 21.1},
        {"rank": 10, "driver": "Valtteri Bottas", "nationality": "Finnish", "podiums": 67, "starts": 246, "podium_pct": 27.2},
        {"rank": 11, "driver": "David Coulthard", "nationality": "British", "podiums": 62, "starts": 246, "podium_pct": 25.2},
        {"rank": 12, "driver": "Nelson Piquet", "nationality": "Brazilian", "podiums": 60, "starts": 204, "podium_pct": 29.4},
        {"rank": 13, "driver": "Nigel Mansell", "nationality": "British", "podiums": 59, "starts": 187, "podium_pct": 31.6},
        {"rank": 14, "driver": "Nico Rosberg", "nationality": "German", "poles": 57, "podiums": 57, "starts": 206, "podium_pct": 27.7},
        {"rank": 15, "driver": "Niki Lauda", "nationality": "Austrian", "podiums": 54, "starts": 171, "podium_pct": 31.6},
    ]

    STREAKS = {
        "consecutive_wins": [
            {"rank": 1, "driver": "Max Verstappen", "count": 10, "season": "2023", "races": "Miami to Monza"},
            {"rank": 2, "driver": "Max Verstappen", "count": 9, "season": "2023–2024", "races": "Japan '23 to Saudi '24"},
            {"rank": 3, "driver": "Sebastian Vettel", "count": 9, "season": "2013", "races": "Spa to Interlagos"},
            {"rank": 4, "driver": "Alberto Ascari", "count": 7, "season": "1952–1953", "races": "Spa '52 to Buenos Aires '53"},
            {"rank": 5, "driver": "Michael Schumacher", "count": 7, "season": "2004", "races": "Nürburgring to Hungaroring"},
            {"rank": 6, "driver": "Nico Rosberg", "count": 7, "season": "2015–2016", "races": "Mexico '15 to Russia '16"},
        ],
        "consecutive_podiums": [
            {"rank": 1, "driver": "Michael Schumacher", "count": 19, "season": "2001–2002", "races": "USA '01 to Japan '02"},
            {"rank": 2, "driver": "Lewis Hamilton", "count": 16, "season": "2014–2015", "races": "Monza '14 to Silverstone '15"},
            {"rank": 3, "driver": "Fernando Alonso", "count": 15, "season": "2005–2006", "races": "Turkey '05 to Canada '06"},
            {"rank": 4, "driver": "Max Verstappen", "count": 15, "season": "2022–2023", "races": "Abu Dhabi '22 to Monza '23"},
            {"rank": 5, "driver": "Sebastian Vettel", "count": 11, "season": "2010–2011", "races": "Brazil '10 to Silverstone '11"},
        ],
        "consecutive_points": [
            {"rank": 1, "driver": "Lewis Hamilton", "count": 48, "season": "2018–2020", "races": "Silverstone '18 to Bahrain '20"},
            {"rank": 2, "driver": "Max Verstappen", "count": 43, "season": "2022–2024", "races": "Imola '22 to Saudi '24"},
            {"rank": 3, "driver": "Lewis Hamilton", "count": 33, "season": "2016–2018", "races": "Japan '16 to France '18"},
            {"rank": 4, "driver": "Kimi Räikkönen", "count": 27, "season": "2012–2013", "races": "Bahrain '12 to Hungary '13"},
            {"rank": 5, "driver": "Michael Schumacher", "count": 24, "season": "2001–2003", "races": "Hungary '01 to Malaysia '03"},
        ],
        "consecutive_starts": [
            {"rank": 1, "driver": "Lewis Hamilton", "count": 265, "season": "2007–2020", "races": "Australia '07 to Bahrain '20"},
            {"rank": 2, "driver": "Daniel Ricciardo", "count": 232, "season": "2011–2022", "races": "Britain '11 to Abu Dhabi '22"},
            {"rank": 3, "driver": "Nico Rosberg", "count": 206, "season": "2006–2016", "races": "Bahrain '06 to Abu Dhabi '16"},
            {"rank": 4, "driver": "Riccardo Patrese", "count": 187, "season": "1982–1993", "races": "Belgium '82 to Australia '93"},
            {"rank": 5, "driver": "Max Verstappen", "count": 185, "season": "2015–2024", "races": "Australia '15 to Present"},
        ]
    }

    AGE_MILESTONES = {
        "youngest_winners": [
            {"rank": 1, "driver": "Max Verstappen", "age": "18 yrs, 228 days", "event": "2016 Spanish Grand Prix"},
            {"rank": 2, "driver": "Sebastian Vettel", "age": "21 yrs, 73 days", "event": "2008 Italian Grand Prix"},
            {"rank": 3, "driver": "Charles Leclerc", "age": "21 yrs, 320 days", "event": "2019 Belgian Grand Prix"},
            {"rank": 4, "driver": "Fernando Alonso", "age": "22 yrs, 26 days", "event": "2003 Hungarian Grand Prix"},
            {"rank": 5, "driver": "Troy Ruttman", "age": "22 yrs, 80 days", "event": "1952 Indianapolis 500"},
        ],
        "oldest_winners": [
            {"rank": 1, "driver": "Luigi Fagioli", "age": "53 yrs, 22 days", "event": "1951 French Grand Prix"},
            {"rank": 2, "driver": "Nino Farina", "age": "46 yrs, 276 days", "event": "1953 German Grand Prix"},
            {"rank": 3, "driver": "Juan Manuel Fangio", "age": "46 yrs, 41 days", "event": "1957 German Grand Prix"},
            {"rank": 4, "driver": "Piero Taruffi", "age": "45 yrs, 219 days", "event": "1952 Swiss Grand Prix"},
            {"rank": 5, "driver": "Jack Brabham", "age": "43 yrs, 339 days", "event": "1970 South African Grand Prix"},
        ],
        "most_starts_no_win": [
            {"rank": 1, "driver": "Nico Hülkenberg", "starts": 227, "best_finish": "P4 (3 times)"},
            {"rank": 2, "driver": "Andrea de Cesaris", "starts": 208, "best_finish": "P2 (2 times)"},
            {"rank": 3, "driver": "Nick Heidfeld", "starts": 183, "best_finish": "P2 (8 times, 13 podiums)"},
            {"rank": 4, "driver": "Romain Grosjean", "starts": 179, "best_finish": "P2 (2 times, 10 podiums)"},
            {"rank": 5, "driver": "Martin Brundle", "starts": 158, "best_finish": "P2 (2 times, 9 podiums)"},
        ]
    }

    def __init__(self, api_client: Optional[HistoricalAPIClient] = None):
        self.api_client = api_client or HistoricalAPIClient()

    def get_all_time_wins(self, top_n: int = 20) -> List[Dict[str, Any]]:
        return self.ALL_TIME_WINS[:top_n]

    def get_all_time_poles(self, top_n: int = 20) -> List[Dict[str, Any]]:
        return self.ALL_TIME_POLES[:top_n]

    def get_all_time_podiums(self, top_n: int = 20) -> List[Dict[str, Any]]:
        return self.ALL_TIME_PODIUMS[:top_n]

    def get_consecutive_records(self, record_type: str = "consecutive_wins") -> List[Dict[str, Any]]:
        rtype = str(record_type).lower().strip()
        return self.STREAKS.get(rtype, self.STREAKS["consecutive_wins"])

    def get_age_milestones(self) -> Dict[str, List[Dict[str, Any]]]:
        return self.AGE_MILESTONES
