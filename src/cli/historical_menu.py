"""
ApexGrid — Historical Stats & Analytics Interactive CLI Menu
=============================================================
Interactive terminal presentation layer for Formula 1 historical analytics across 1950–2026.
Fetch-only: All data is queried on-demand and calculated in memory; zero disk persistence.
Provides guided workflows for:
1. Driver Statistics & Career Breakdowns
2. Teammate Head-to-Head Comparisons
3. Qualifying Performance & Pole Statistics
4. Race Performance & Positions Gained
5. Circuit History & Historical Winners
6. Season Summaries & Championship Progression
0. Return to Main Menu
"""

import sys
from typing import Optional, List, Dict, Any

from src.cli.terminal_ui import (
    clear_screen,
    print_banner,
    print_header,
    print_success,
    print_warning,
    print_error,
    print_info,
    prompt_choice,
    prompt_str,
    prompt_int,
    print_table,
    render_probability_bar,
    get_console,
    Colors,
    HAS_RICH,
    _supports_color,
)
from src.historical.api_client import HistoricalAPIClient
from src.historical.analytics import (
    DriverAnalyticsService,
    TeammateAnalyticsService,
    QualifyingAnalyticsService,
    RacePerformanceAnalyticsService,
    CircuitAnalyticsService,
    SeasonAnalyticsService,
)
from src.utils.logging_utils import get_logger

logger = get_logger("historical_cli_menu")


class HistoricalStatsMenu:
    """Master controller for historical analytics terminal interface."""

    def __init__(self, api_client: Optional[HistoricalAPIClient] = None):
        self.api_client = api_client or HistoricalAPIClient()
        self.driver_service = DriverAnalyticsService(api_client=self.api_client)
        self.teammate_service = TeammateAnalyticsService(api_client=self.api_client)
        self.qualifying_service = QualifyingAnalyticsService(api_client=self.api_client)
        self.race_service = RacePerformanceAnalyticsService(api_client=self.api_client)
        self.circuit_service = CircuitAnalyticsService(api_client=self.api_client)
        self.season_service = SeasonAnalyticsService(api_client=self.api_client)

    def run(self) -> None:
        """Main interactive event loop for historical stats."""
        while True:
            try:
                clear_screen()
                print_banner(season=2026, status_text="Historical Analytics")
                console = get_console()
                if HAS_RICH and console and _supports_color():
                    console.print("[bold #F5F5F7]HISTORICAL MOTORSPORT INTELLIGENCE (1950–2026)[/]\n")
                    console.print("  [bold #00B4D8]1.[/] Driver Career & Season Statistics")
                    console.print("  [bold #00B4D8]2.[/] Teammate Head-to-Head Comparison")
                    console.print("  [bold #00B4D8]3.[/] Qualifying Analytics & Pole Statistics")
                    console.print("  [bold #00B4D8]4.[/] Race Performance & Positions Gained")
                    console.print("  [bold #00B4D8]5.[/] Circuit History & Track Records")
                    console.print("  [bold #00B4D8]6.[/] Championship Season Summary & Standings")
                    console.print("\n  [bold red]0.[/] Return to Main Menu\n")
                else:
                    print(f"{Colors.BOLD}HISTORICAL MOTORSPORT INTELLIGENCE (1950–2026){Colors.RESET}\n")
                    print("  1. Driver Career & Season Statistics")
                    print("  2. Teammate Head-to-Head Comparison")
                    print("  3. Qualifying Analytics & Pole Statistics")
                    print("  4. Race Performance & Positions Gained")
                    print("  5. Circuit History & Track Records")
                    print("  6. Championship Season Summary & Standings")
                    print("\n  0. Return to Main Menu\n")

                choice = prompt_choice("Select an analytics category", ["1", "2", "3", "4", "5", "6", "0"], default="1")

                if choice == "0":
                    break
                elif choice == "1":
                    self.view_driver_stats()
                elif choice == "2":
                    self.view_teammate_h2h()
                elif choice == "3":
                    self.view_qualifying_stats()
                elif choice == "4":
                    self.view_race_performance()
                elif choice == "5":
                    self.view_circuit_stats()
                elif choice == "6":
                    self.view_season_stats()

            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                print_error(f"Error in Historical Analytics: {e}")
                prompt_str("\nPress [Enter] to continue", allow_empty=True)

    # =========================================================================
    # 1. Driver Statistics
    # =========================================================================
    def view_driver_stats(self) -> None:
        """Guided workflow to view career and per-season driver statistics."""
        print_header("Driver Statistics", "Historical Career Trajectory & Finishes")
        driver_input = prompt_str("Enter Driver ID or Surname (e.g. 'alonso', 'senna', 'hamilton')")
        if not driver_input:
            return

        season_input = prompt_str("Enter Season year (optional, press Enter for full career)", allow_empty=True)
        season_val = int(season_input) if season_input.isdigit() else None

        print_info(f"Querying historical provider data for driver '{driver_input}'...")
        stats = self.driver_service.get_driver_stats(driver_input, season=season_val)
        if not stats:
            print_warning(f"No historical records found for driver '{driver_input}' (check spelling or provider availability).")
            prompt_str("\nPress [Enter] to return", allow_empty=True)
            return

        profile_sub = f"Nationality: {stats.nationality or 'Unknown'} | Active Seasons: {min(stats.seasons_active)}–{max(stats.seasons_active)}"
        if stats.seasons_without_recorded_entries:
            profile_sub += f" (Seasons without recorded championship entries in provider archive: {', '.join(str(y) for y in stats.seasons_without_recorded_entries)})"
        print_header(f"Driver Profile: {stats.driver_name.upper()}", profile_sub)

        # Determine qualifying session P1 display string
        if stats.qualifying_session_p1s is not None:
            poles_display = str(stats.qualifying_session_p1s)
        elif stats.known_covered_poles is not None:
            poles_display = f"Unavailable (Incomplete pre-1994 archive; {stats.known_covered_poles} known in 1994+ era)"
        else:
            if any(s >= 1994 for s in stats.seasons_active):
                poles_display = "Unavailable (API/Provider Error)"
            else:
                poles_display = "Unavailable (Pre-1994 Archive Incomplete)"

        # Determine average finish position display string
        if stats.avg_finish_position is not None:
            avg_fin_display = f"P{stats.avg_finish_position:.2f} (across {stats.classified_finishes} classified finishes)"
        else:
            avg_fin_display = "N/A (No classified finishes)"

        overview_headers = ["Metric", "Career Value"]
        overview_rows = [
            ["Recorded Grand Prix Appearances", str(stats.total_entries)],
            ["Grand Prix Starts", str(stats.total_starts)],
            ["Non-Starts (DNS/Withdrawn)", str(stats.non_starts_count)],
            ["Classified Finishes", str(stats.classified_finishes)],
            ["  - Running Finishes (Chequered Flag)", str(stats.running_finishes)],
        ]
        if stats.classified_retirements_count > 0:
            overview_rows.append(["  - Classified Despite Retiring (Provider Classified)", str(stats.classified_retirements_count)])
        overview_rows.extend([
            ["Unclassified Retirements (DNF)", str(stats.unclassified_retirements_count)],
            ["Total Retirements (All DNFs)", str(stats.dnf_count)],
            ["Grand Prix Victories", str(stats.wins)],
            ["Podium Finishes", str(stats.podiums)],
            ["Qualifying Session P1s", poles_display],
            ["Championship Points", f"{stats.total_points:.2f}"],
            ["Average Finish Position", avg_fin_display],
        ])
        print_table(overview_headers, overview_rows, align_right=[False, True])

        if stats.seasons_breakdown:
            print(f"\n{Colors.BOLD}Season-by-Season Breakdown:{Colors.RESET}")
            s_headers = ["Season", "Constructor", "Starts", "Wins", "Podiums", "Poles", "Points", "Best", "Avg Fin"]
            s_rows = []
            for sb in stats.seasons_breakdown:
                if sb.poles is not None:
                    p_str = str(sb.poles)
                elif sb.season < 1994:
                    p_str = "N/A"
                else:
                    p_str = "Error"
                season_label = f"{sb.season} (Prov)" if sb.is_provisional else str(sb.season)
                s_rows.append([
                    season_label,
                    sb.constructor_name[:18],
                    str(sb.starts),
                    str(sb.wins),
                    str(sb.podiums),
                    p_str,
                    f"{sb.points:.1f}",
                    f"P{sb.best_finish}" if sb.best_finish else "-",
                    f"P{sb.avg_finish:.1f}" if sb.avg_finish else "-",
                ])
            print_table(s_headers, s_rows, align_right=[False, False, True, True, True, True, True, True, True])

        if stats.coverage_notes:
            print(f"\n{Colors.DIM}Provider Coverage & Historical Limitations:{Colors.RESET}")
            for note in stats.coverage_notes:
                print(f"  * {note}")

        prompt_str("\nPress [Enter] to return to Historical Menu", allow_empty=True)

    # =========================================================================
    # 2. Teammate Head-to-Head
    # =========================================================================
    def view_teammate_h2h(self) -> None:
        """Guided workflow to compare two drivers who competed for the same constructor."""
        print_header("Teammate Head-to-Head Comparison", "Same Team & Season Performance")
        season_year = prompt_int("Enter Championship Season (1950–2026)", min_val=1950, max_val=2026, default=2023)
        driver1 = prompt_str("Enter First Driver ID (e.g. 'verstappen', 'hamilton', 'senna')")
        driver2 = prompt_str("Enter Second Driver ID (e.g. 'perez', 'rosberg', 'prost')")
        if not driver1 or not driver2:
            return

        print_info(f"Computing head-to-head comparison for {driver1} vs {driver2} in {season_year}...")
        h2h = self.teammate_service.compare_teammates(driver1, driver2, season_year)
        if not h2h:
            print_warning(f"Could not establish teammate comparison. Ensure both drivers competed for the same constructor in {season_year}.")
            prompt_str("\nPress [Enter] to return", allow_empty=True)
            return

        print_header(f"HEAD-TO-HEAD: {h2h.driver1_name.upper()} vs {h2h.driver2_name.upper()}", f"Constructor: {h2h.constructor_name} | Season: {h2h.season} ({h2h.races_cohabited} Races as Teammates)")

        headers = ["Comparison Dimension", h2h.driver1_name, h2h.driver2_name]
        rows = [
            ["Races Finished Ahead", str(h2h.race_h2h[0]), str(h2h.race_h2h[1])],
            ["Qualifying Ahead", str(h2h.qualifying_h2h[0]), str(h2h.qualifying_h2h[1])],
            ["Grand Prix Wins", str(h2h.wins_comparison[0]), str(h2h.wins_comparison[1])],
            ["Podium Finishes", str(h2h.podiums_comparison[0]), str(h2h.podiums_comparison[1])],
            ["Championship Points", f"{h2h.points_comparison[0]:.2f}", f"{h2h.points_comparison[1]:.2f}"],
            ["Retirements (DNF)", str(h2h.dnfs_comparison[0]), str(h2h.dnfs_comparison[1])],
        ]
        print_table(headers, rows, align_right=[False, True, True])

        print(f"\n{Colors.DIM}Comparison Rules & Notes:{Colors.RESET}")
        print(f"  * {h2h.qualifying_coverage_note}")
        for n in h2h.notes:
            print(f"  * {n}")

        prompt_str("\nPress [Enter] to return to Historical Menu", allow_empty=True)

    # =========================================================================
    # 3. Qualifying Statistics
    # =========================================================================
    def view_qualifying_stats(self) -> None:
        """Guided workflow to view qualifying analytics and pole positions."""
        print_header("Qualifying Analytics", "Session Formats, Pole Positions & Grid Drops")
        driver_input = prompt_str("Enter Driver ID (e.g. 'verstappen', 'leclerc', 'senna')")
        if not driver_input:
            return

        season_input = prompt_str("Enter Season year (optional, press Enter for full career)", allow_empty=True)
        season_val = int(season_input) if season_input.isdigit() else None

        print_info(f"Querying qualifying analytics for '{driver_input}'...")
        q_stats = self.qualifying_service.get_driver_qualifying_summary(driver_input, season=season_val)
        if not q_stats:
            print_warning(f"No qualifying records found for driver '{driver_input}'.")
            prompt_str("\nPress [Enter] to return", allow_empty=True)
            return

        subtitle = f"Season: {q_stats.season}" if q_stats.season else "Career Across Available Sessions"
        print_header(f"Qualifying Record: {q_stats.driver_name.upper()}", subtitle)

        headers = ["Metric", "Value"]
        rows = [
            ["Qualifying Sessions Recorded", str(q_stats.total_sessions)],
            ["Pole Positions (P1)", str(q_stats.pole_positions)],
            ["Front Row Starts (P1-P2)", str(q_stats.front_row_starts)],
            ["Top 10 Starts", str(q_stats.top_10_starts)],
            ["Average Qualifying Position", f"P{q_stats.avg_qualifying_position:.2f}" if q_stats.avg_qualifying_position else "N/A"],
            ["Formats Encountered", ", ".join(q_stats.formats_encountered)],
        ]
        if q_stats.avg_grid_drop_or_gain is not None:
            rows.append(["Average Grid Penalty Drop", f"{q_stats.avg_grid_drop_or_gain:+.2f} positions"])

        print_table(headers, rows, align_right=[False, True])

        if q_stats.coverage_limitations:
            print(f"\n{Colors.DIM}Provider Coverage Limitations:{Colors.RESET}")
            for n in q_stats.coverage_limitations:
                print(f"  * {n}")

        prompt_str("\nPress [Enter] to return to Historical Menu", allow_empty=True)

    # =========================================================================
    # 4. Race Performance Analytics
    # =========================================================================
    def view_race_performance(self) -> None:
        """Guided workflow to view race performance, positions gained, and finishing distributions."""
        print_header("Race Performance Analytics", "Positions Gained, Finishing Distributions & Reliability")
        driver_input = prompt_str("Enter Driver ID (e.g. 'alonso', 'verstappen', 'prost')")
        if not driver_input:
            return

        season_input = prompt_str("Enter Season year (optional, press Enter for full career)", allow_empty=True)
        season_val = int(season_input) if season_input.isdigit() else None

        print_info(f"Computing race performance analytics for '{driver_input}'...")
        r_stats = self.race_service.get_driver_race_performance(driver_input, season=season_val)
        if not r_stats:
            print_warning(f"No race classification records found for '{driver_input}'.")
            prompt_str("\nPress [Enter] to return", allow_empty=True)
            return

        print_header(f"Race Execution: {r_stats.driver_name.upper()}", f"Starts: {r_stats.starts} | Classified: {r_stats.classified_finishes}")

        headers = ["Dimension", "Statistic"]
        rows = [
            ["Grand Prix Starts", str(r_stats.starts)],
            ["Classified Finishes", str(r_stats.classified_finishes)],
            ["Points Scoring Finishes", str(r_stats.points_finishes)],
            ["Total Points Scored", f"{r_stats.total_points:.2f}"],
            ["DNF Rate", f"{r_stats.dnf_rate_pct:.1f}% ({r_stats.dnfs} retirements)"],
            ["Net Positions Gained (Career)", f"{r_stats.positions_gained_total:+d}" if r_stats.positions_gained_total is not None else "N/A"],
            ["Average Positions Gained / Race", f"{r_stats.avg_positions_gained:+.2f}" if r_stats.avg_positions_gained is not None else "N/A"],
            ["Valid Grid Starts Analyzed", str(r_stats.valid_grid_starts_count)],
        ]
        print_table(headers, rows, align_right=[False, True])

        if r_stats.finishing_distribution:
            print(f"\n{Colors.BOLD}Finishing Position Distribution:{Colors.RESET}")
            d_headers = ["Position Bracket", "Count"]
            d_rows = [[k, str(v)] for k, v in r_stats.finishing_distribution.items()]
            print_table(d_headers, d_rows, align_right=[False, True])

        if r_stats.retirement_breakdown:
            print(f"\n{Colors.BOLD}Top Retirement Causes:{Colors.RESET}")
            ret_headers = ["Reason / Status", "Occurrences"]
            ret_rows = [[k, str(v)] for k, v in list(r_stats.retirement_breakdown.items())[:5]]
            print_table(ret_headers, ret_rows, align_right=[False, True])

        for n in r_stats.coverage_notes:
            print(f"\n{Colors.DIM}  * {n}{Colors.RESET}")

        prompt_str("\nPress [Enter] to return to Historical Menu", allow_empty=True)

    # =========================================================================
    # 5. Circuit Statistics
    # =========================================================================
    def view_circuit_stats(self) -> None:
        """Guided workflow to view track history and historical winners."""
        print_header("Circuit History & Statistics", "Venue Records & Historical Winners")
        circuit_input = prompt_str("Enter Circuit ID (e.g. 'monza', 'silverstone', 'spa', 'monaco')")
        if not circuit_input:
            return

        print_info(f"Querying historical statistics for venue '{circuit_input}'...")
        c_stats = self.circuit_service.get_circuit_history(circuit_input)
        if not c_stats:
            print_warning(f"No historical Grand Prix records found for circuit '{circuit_input}'.")
            prompt_str("\nPress [Enter] to return", allow_empty=True)
            return

        location_str = f"{c_stats.locality or ''}, {c_stats.country or ''}".strip(", ")
        print_header(f"Circuit: {c_stats.circuit_name.upper()}", f"Location: {location_str} | Hosted: {c_stats.first_gp_year}–{c_stats.last_gp_year} ({c_stats.total_races_hosted} Grands Prix)")

        if c_stats.top_winning_drivers:
            print(f"\n{Colors.BOLD}Top Winning Drivers:{Colors.RESET}")
            w_headers = ["Driver", "Wins"]
            w_rows = [[d, str(w)] for d, w in c_stats.top_winning_drivers[:5]]
            print_table(w_headers, w_rows, align_right=[False, True])

        if c_stats.top_winning_constructors:
            print(f"\n{Colors.BOLD}Top Winning Constructors:{Colors.RESET}")
            cw_headers = ["Constructor", "Wins"]
            cw_rows = [[c, str(w)] for c, w in c_stats.top_winning_constructors[:5]]
            print_table(cw_headers, cw_rows, align_right=[False, True])

        if c_stats.recent_winners:
            print(f"\n{Colors.BOLD}Recent Grand Prix Winners:{Colors.RESET}")
            rec_headers = ["Year", "Winning Driver", "Constructor"]
            rec_rows = [[str(y), d, c] for y, d, c in c_stats.recent_winners]
            print_table(rec_headers, rec_rows, align_right=[True, False, False])

        for n in c_stats.coverage_notes:
            print(f"\n{Colors.DIM}  * {n}{Colors.RESET}")

        prompt_str("\nPress [Enter] to return to Historical Menu", allow_empty=True)

    # =========================================================================
    # 6. Season Statistics
    # =========================================================================
    def view_season_stats(self) -> None:
        """Guided workflow to view season progression and championship standings."""
        print_header("Championship Season Analytics", "Standings, Participation & Coverage")
        year = prompt_int("Enter Season Year (1950–2026)", min_val=1950, max_val=2026, default=2023)

        print_info(f"Fetching championship analytics for season {year}...")
        s_stats = self.season_service.get_season_summary(year)
        if not s_stats:
            print_warning(f"No championship calendar or result records found for season {year}.")
            prompt_str("\nPress [Enter] to return", allow_empty=True)
            return

        print_header(f"Championship Season {year}", s_stats.status_display)

        headers = ["Championship Dimension", "Summary"]
        rows = [
            ["Scheduled Rounds", str(s_stats.total_scheduled_rounds)],
            ["Completed Rounds", str(s_stats.completed_rounds)],
            ["Participating Drivers", str(s_stats.participating_drivers_count)],
            ["Participating Constructors", str(s_stats.participating_constructors_count)],
            ["Qualifying Coverage", s_stats.qualifying_coverage_summary],
            ["Race Results Coverage", s_stats.race_results_coverage_summary],
        ]
        print_table(headers, rows, align_right=[False, True])

        if s_stats.top_drivers_standings:
            print(f"\n{Colors.BOLD}Top Drivers Championship Standings:{Colors.RESET}")
            d_headers = ["Pos", "Driver", "Constructor", "Points", "Wins"]
            d_rows = [[str(pos), name, team[:18], f"{pts:.1f}", str(wins)] for pos, name, team, pts, wins in s_stats.top_drivers_standings]
            print_table(d_headers, d_rows, align_right=[True, False, False, True, True])

        if s_stats.top_constructors_standings:
            print(f"\n{Colors.BOLD}Top Constructors Championship Standings:{Colors.RESET}")
            c_headers = ["Pos", "Constructor", "Points", "Wins"]
            c_rows = [[str(pos), team, f"{pts:.1f}", str(wins)] for pos, team, pts, wins in s_stats.top_constructors_standings]
            print_table(c_headers, c_rows, align_right=[True, False, True, True])

        if s_stats.coverage_notes:
            print(f"\n{Colors.DIM}Season Notes & Coverage Limitations:{Colors.RESET}")
            for n in s_stats.coverage_notes:
                print(f"  * {n}")

        prompt_str("\nPress [Enter] to return to Historical Menu", allow_empty=True)
