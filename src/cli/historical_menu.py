"""
ApexGrid — Historical Stats & Analytics Interactive CLI Menu
=============================================================
Interactive terminal presentation layer for Formula 1 historical analytics across 1950–2026.
Fetch-only: All data is queried on-demand and calculated in memory; zero disk persistence.
Provides guided workflows for:
1. Driver Statistics & Career Breakdowns
2. Teammate Head-to-Head Comparisons
3. Qualifying Performance & Pole Statistics
4. Race Execution & Overtakes/Positions Gained
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
    get_terminal_width,
    format_menu_divider,
    get_console,
    HAS_RICH,
    _supports_color,
    Colors,
)
from src.historical.api_client import HistoricalAPIClient
from src.historical.analytics import (
    SessionResultsService,
    DriverAnalyticsService,
    TeammateAnalyticsService,
    QualifyingAnalyticsService,
    RacePerformanceAnalyticsService,
    CircuitAnalyticsService,
    SeasonAnalyticsService,
    DominanceAnalyticsService,
    AllTimeRecordsService,
    ReliabilityAnalyticsService,
    EraAnalyticsService,
)
from src.utils.logging_utils import get_logger

logger = get_logger("historical_cli_menu")


class HistoricalStatsMenu:
    """Master controller for historical analytics terminal interface."""

    def __init__(self, api_client: Optional[HistoricalAPIClient] = None):
        self.api_client = api_client or HistoricalAPIClient()
        self.session_service = SessionResultsService(api_client=self.api_client)
        self.driver_service = DriverAnalyticsService(api_client=self.api_client)
        self.teammate_service = TeammateAnalyticsService(api_client=self.api_client)
        self.qualifying_service = QualifyingAnalyticsService(api_client=self.api_client)
        self.race_service = RacePerformanceAnalyticsService(api_client=self.api_client)
        self.circuit_service = CircuitAnalyticsService(api_client=self.api_client)
        self.season_service = SeasonAnalyticsService(api_client=self.api_client)
        self.dominance_service = DominanceAnalyticsService(api_client=self.api_client)
        self.records_service = AllTimeRecordsService(api_client=self.api_client)
        self.reliability_service = ReliabilityAnalyticsService(api_client=self.api_client)
        self.era_service = EraAnalyticsService(api_client=self.api_client)

    def run(self) -> None:
        """Main interactive event loop for historical stats."""
        while True:
            try:
                clear_screen()
                print_banner(season=2026, status_text="Historical Analytics")
                w = get_terminal_width()
                title_str = "HISTORICAL MOTORSPORT INTELLIGENCE (1950–2026)"
                if w < len(title_str) + 4:
                    title_str = "HISTORICAL MOTORSPORT INTELLIGENCE"

                console = get_console()
                if HAS_RICH and console and _supports_color():
                    console.print(f"[bold #F5F5F7]{title_str}[/]\n")
                    console.print(f"  {format_menu_divider('Weekend & Session Classifications', 'bold #00B4D8')}")
                    console.print("  [bold white]1.[/] Checkered Flag & Session Results (FP1–FP3, SQ, Sprint, Quali, Race)")
                    console.print(f"\n  {format_menu_divider('Driver, Team & Circuit Analytics', 'bold #B138DD')}")
                    console.print("  [bold white]2.[/] Driver Career & Season Statistics")
                    console.print("  [bold white]3.[/] Teammate Head-to-Head Comparison")
                    console.print("  [bold white]4.[/] Qualifying Analytics & Pole Statistics")
                    console.print("  [bold white]5.[/] Race Performance & Positions Gained")
                    console.print("  [bold white]6.[/] Circuit History & Track Records")
                    console.print("  [bold white]7.[/] Championship Season Summary & Standings")
                    console.print(f"\n  {format_menu_divider('All-Time Records & Era History', 'bold #FFB800')}")
                    console.print("  [bold white]8.[/] Constructor & Engine Dominance Analytics")
                    console.print("  [bold white]9.[/] All-Time Records, Streaks & Milestones")
                    console.print(" [bold white]10.[/] Historical Reliability & Retirement Stats")
                    console.print(" [bold white]11.[/] F1 Eras & Regulation Comparison (1950–2026)")
                    console.print("\n  [bold red]0.[/] Return to Main Menu\n")
                else:
                    print(f"{Colors.BOLD}{title_str}{Colors.RESET}\n")
                    print(f"  {Colors.CYAN}{format_menu_divider('Weekend & Session Classifications')}{Colors.RESET}")
                    print("  1. Checkered Flag & Session Results (FP1–FP3, SQ, Sprint, Quali, Race)")
                    print(f"\n  {Colors.MAGENTA}{format_menu_divider('Driver, Team & Circuit Analytics')}{Colors.RESET}")
                    print("  2. Driver Career & Season Statistics")
                    print("  3. Teammate Head-to-Head Comparison")
                    print("  4. Qualifying Analytics & Pole Statistics")
                    print("  5. Race Performance & Positions Gained")
                    print("  6. Circuit History & Track Records")
                    print("  7. Championship Season Summary & Standings")
                    print(f"\n  {Colors.YELLOW}{format_menu_divider('All-Time Records & Era History')}{Colors.RESET}")
                    print("  8. Constructor & Engine Dominance Analytics")
                    print("  9. All-Time Records, Streaks & Milestones")
                    print(" 10. Historical Reliability & Retirement Stats")
                    print(" 11. F1 Eras & Regulation Comparison (1950–2026)")
                    print(f"\n  {Colors.RED}0.{Colors.RESET} Return to Main Menu\n")

                choice = prompt_choice(
                    "Select an analytics category",
                    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "0"],
                    default="1"
                )

                if choice == "0":
                    break
                elif choice == "1":
                    self.view_session_results()
                elif choice == "2":
                    self.view_driver_stats()
                elif choice == "3":
                    self.view_teammate_h2h()
                elif choice == "4":
                    self.view_qualifying_stats()
                elif choice == "5":
                    self.view_race_performance()
                elif choice == "6":
                    self.view_circuit_stats()
                elif choice == "7":
                    self.view_season_stats()
                elif choice == "8":
                    self.view_constructor_dominance()
                elif choice == "9":
                    self.view_all_time_records()
                elif choice == "10":
                    self.view_reliability_stats()
                elif choice == "11":
                    self.view_era_comparisons()

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
        print_table(overview_headers, overview_rows)

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
            print_table(s_headers, s_rows)

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
        print_table(headers, rows)

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

        print_table(headers, rows)

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
        print_table(headers, rows)

        if r_stats.finishing_distribution:
            print(f"\n{Colors.BOLD}Finishing Position Distribution:{Colors.RESET}")
            d_headers = ["Position Bracket", "Count"]
            d_rows = [[k, str(v)] for k, v in r_stats.finishing_distribution.items()]
            print_table(d_headers, d_rows)

        if r_stats.retirement_breakdown:
            print(f"\n{Colors.BOLD}Top Retirement Causes:{Colors.RESET}")
            ret_headers = ["Reason / Status", "Occurrences"]
            ret_rows = [[k, str(v)] for k, v in list(r_stats.retirement_breakdown.items())[:5]]
            print_table(ret_headers, ret_rows)

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
            print_table(w_headers, w_rows)

        if c_stats.top_winning_constructors:
            print(f"\n{Colors.BOLD}Top Winning Constructors:{Colors.RESET}")
            cw_headers = ["Constructor", "Wins"]
            cw_rows = [[c, str(w)] for c, w in c_stats.top_winning_constructors[:5]]
            print_table(cw_headers, cw_rows)

        if c_stats.recent_winners:
            print(f"\n{Colors.BOLD}Recent Grand Prix Winners:{Colors.RESET}")
            rec_headers = ["Year", "Winning Driver", "Constructor"]
            rec_rows = [[str(y), d, c] for y, d, c in c_stats.recent_winners]
            print_table(rec_headers, rec_rows)

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
        print_table(headers, rows)

        if s_stats.top_drivers_standings:
            print(f"\n{Colors.BOLD}Top Drivers Championship Standings:{Colors.RESET}")
            d_headers = ["Pos", "Driver", "Constructor", "Points", "Wins"]
            d_rows = [[str(pos), name, team[:18], f"{pts:.1f}", str(wins)] for pos, name, team, pts, wins in s_stats.top_drivers_standings]
            print_table(d_headers, d_rows)

        if s_stats.top_constructors_standings:
            print(f"\n{Colors.BOLD}Top Constructors Championship Standings:{Colors.RESET}")
            c_headers = ["Pos", "Constructor", "Points", "Wins"]
            c_rows = [[str(pos), team, f"{pts:.1f}", str(wins)] for pos, team, pts, wins in s_stats.top_constructors_standings]
            print_table(c_headers, c_rows)

        if s_stats.coverage_notes:
            print(f"\n{Colors.DIM}Season Notes & Coverage Limitations:{Colors.RESET}")
            for n in s_stats.coverage_notes:
                print(f"  * {n}")

        prompt_str("\nPress [Enter] to return to Historical Menu", allow_empty=True)

    # =========================================================================
    # 1. Checkered Flag & Session Results (New Option 1)
    # =========================================================================
    def view_session_results(self) -> None:
        """Guided workflow to view official session results (FP1-3, SQ, Sprint, Quali, Race)."""
        print_header("Checkered Flag & Session Results", "Official Finishing Classifications (1950–2026)")
        year = prompt_int("Enter Season Year (1950–2026)", min_val=1950, max_val=2026, default=2024)
        round_num = prompt_int("Enter Round Number", min_val=1, max_val=30, default=1)

        # Detect format dynamically
        is_sprint = (year >= 2021) and self.session_service.is_sprint_round(year, round_num)

        if year < 2018:
            print(f"\nAvailable Session Types ({year} Historical Era):")
            print("  1. Race (Grand Prix Checkered Flag Classification)")
            print("  2. Qualifying (Official Grid / Starting Classification)")
            sess_choice = prompt_choice("Select session", ["1", "2"], default="1")
            action = "race" if sess_choice == "1" else "qualifying"
        elif is_sprint:
            print("\nAvailable Session Types (Sprint Weekend Format):")
            print("  1. Race (Grand Prix Checkered Flag Classification)")
            print("  2. Qualifying (Grand Prix Qualifying Classification)")
            print("  3. Sprint Race (Sprint Classification)")
            print("  4. Sprint Qualifying (SQ Telemetry / Shootout)")
            print("  5. Practice 1 (FP1 Telemetry Lap Times)")
            sess_choice = prompt_choice("Select session", ["1", "2", "3", "4", "5"], default="1")
            action_map = {
                "1": ("race", None),
                "2": ("qualifying", None),
                "3": ("sprint_race", None),
                "4": ("sprint_qualifying", None),
                "5": ("practice", "fp1"),
            }
            action, stype = action_map[sess_choice]
        else:
            print("\nAvailable Session Types (Conventional Weekend Format):")
            print("  1. Race (Grand Prix Checkered Flag Classification)")
            print("  2. Qualifying (Knockout / Official Grid Classification)")
            print("  3. Practice 1 (FP1 Telemetry Lap Times)")
            print("  4. Practice 2 (FP2 Telemetry Lap Times)")
            print("  5. Practice 3 (FP3 Telemetry Lap Times)")
            sess_choice = prompt_choice("Select session", ["1", "2", "3", "4", "5"], default="1")
            action_map = {
                "1": ("race", None),
                "2": ("qualifying", None),
                "3": ("practice", "fp1"),
                "4": ("practice", "fp2"),
                "5": ("practice", "fp3"),
            }
            action, stype = action_map[sess_choice]

        if action == "race":
            print_info(f"Retrieving official Race classification for {year} Round {round_num}...")
            race_results = self.session_service.get_race_results(year, round_num)
            if not race_results:
                print_warning(f"No race classification recorded for {year} Round {round_num}.")
            else:
                headers = ["Pos", "Driver", "Constructor", "Grid", "Laps", "Time/Gap", "Status", "Pts"]
                rows = []
                for r in race_results:
                    pos = str(getattr(r, "position", None) or getattr(r, "position_text", "-"))
                    grid = str(getattr(r, "grid_position", getattr(r, "grid", None)) or "-")
                    time_disp = getattr(r, "race_time_or_gap", getattr(r, "time_str", None)) or ("Leader" if pos == "1" else "")
                    d_name = getattr(r, "driver_name", getattr(r, "driver_id", "Unknown"))
                    c_name = getattr(r, "constructor_name", "Unknown")[:18]
                    laps = str(getattr(r, "laps_completed", getattr(r, "laps", 0)))
                    status = str(getattr(r, "status", "Finished"))[:14]
                    pts = f"{getattr(r, 'points', 0.0):g}"
                    rows.append([pos, d_name, c_name, grid, laps, time_disp, status, pts])
                print_header(f"{year} Round {round_num} — Grand Prix Race Results", f"{len(race_results)} Classified Entries")
                print_table(headers, rows)

        elif action == "qualifying":
            print_info(f"Retrieving official Qualifying classification for {year} Round {round_num}...")
            quali_results = self.session_service.get_qualifying_results(year, round_num)
            if not quali_results:
                if year < 1994:
                    print_warning(f"Qualifying session classifications are unavailable in historical provider archive for pre-1994 seasons (Race starting grids exist in Option 1).")
                else:
                    print_warning(f"No qualifying results recorded for {year} Round {round_num}.")
            else:
                has_q3 = any(getattr(q, "q3", None) for q in quali_results)
                has_q2 = any(getattr(q, "q2", None) for q in quali_results)
                if has_q3:
                    headers = ["Pos", "Driver", "Constructor", "Q1", "Q2", "Q3"]
                    rows = [[str(getattr(q, "position", "-")), getattr(q, "driver_name", "Unknown"), getattr(q, "constructor_name", "Unknown")[:16], getattr(q, "q1", None) or "-", getattr(q, "q2", None) or "-", getattr(q, "q3", None) or "-"] for q in quali_results]
                elif has_q2:
                    headers = ["Pos", "Driver", "Constructor", "Q1", "Q2"]
                    rows = [[str(getattr(q, "position", "-")), getattr(q, "driver_name", "Unknown"), getattr(q, "constructor_name", "Unknown")[:16], getattr(q, "q1", None) or "-", getattr(q, "q2", None) or "-"] for q in quali_results]
                else:
                    headers = ["Pos", "Driver", "Constructor", "Time"]
                    rows = [[str(getattr(q, "position", "-")), getattr(q, "driver_name", "Unknown"), getattr(q, "constructor_name", "Unknown")[:16], getattr(q, "best_lap_time", None) or getattr(q, "q1", None) or "-"] for q in quali_results]
                print_header(f"{year} Round {round_num} — Official Qualifying Results", f"{len(quali_results)} Classified Entries")
                print_table(headers, rows)

        elif action == "sprint_race":
            print_info(f"Retrieving official Sprint classification for {year} Round {round_num}...")
            sprint_results = self.session_service.get_sprint_results(year, round_num)
            if not sprint_results:
                print_warning(f"No sprint race data found for {year} Round {round_num}.")
            else:
                headers = ["Pos", "Driver", "Constructor", "Grid", "Laps", "Time/Gap", "Status", "Pts"]
                rows = []
                for s in sprint_results:
                    pos = str(getattr(s, "position", None) or getattr(s, "position_text", "-"))
                    grid = str(getattr(s, "grid_position", getattr(s, "grid", None)) or "-")
                    time_disp = getattr(s, "race_time_or_gap", getattr(s, "time_str", None)) or ("Leader" if pos == "1" else "")
                    d_name = getattr(s, "driver_name", getattr(s, "driver_id", "Unknown"))
                    c_name = getattr(s, "constructor_name", "Unknown")[:18]
                    laps = str(getattr(s, "laps_completed", getattr(s, "laps", 0)))
                    status = str(getattr(s, "status", "Finished"))[:14]
                    pts = f"{getattr(s, 'points', 0.0):g}"
                    rows.append([pos, d_name, c_name, grid, laps, time_disp, status, pts])
                print_header(f"{year} Round {round_num} — Sprint Race Results", f"{len(sprint_results)} Classified Entries")
                print_table(headers, rows)

        elif action == "sprint_qualifying":
            print_info(f"Retrieving Sprint Qualifying (SQ) classification for {year} Round {round_num}...")
            sq_results = self.session_service.get_sprint_qualifying_results(year, round_num)
            if not sq_results:
                print_warning(f"No sprint qualifying (SQ) data recorded for {year} Round {round_num}.")
            else:
                has_q3 = any(getattr(q, "q3", None) for q in sq_results)
                has_q2 = any(getattr(q, "q2", None) for q in sq_results)
                if has_q3:
                    headers = ["Pos", "Driver", "Constructor", "SQ1", "SQ2", "SQ3"]
                    rows = [[str(getattr(q, "position", "-")), getattr(q, "driver_name", "Unknown"), getattr(q, "constructor_name", "Unknown")[:16], getattr(q, "q1", None) or "-", getattr(q, "q2", None) or "-", getattr(q, "q3", None) or "-"] for q in sq_results]
                elif has_q2:
                    headers = ["Pos", "Driver", "Constructor", "SQ1", "SQ2"]
                    rows = [[str(getattr(q, "position", "-")), getattr(q, "driver_name", "Unknown"), getattr(q, "constructor_name", "Unknown")[:16], getattr(q, "q1", None) or "-", getattr(q, "q2", None) or "-"] for q in sq_results]
                else:
                    headers = ["Pos", "Driver", "Constructor", "Time"]
                    rows = [[str(getattr(q, "position", "-")), getattr(q, "driver_name", "Unknown"), getattr(q, "constructor_name", "Unknown")[:16], getattr(q, "best_lap_time", None) or getattr(q, "q1", None) or "-"] for q in sq_results]
                print_header(f"{year} Round {round_num} — Sprint Qualifying (SQ) Results", f"{len(sq_results)} Drivers Classified")
                print_table(headers, rows)

        elif action == "practice":
            print_info(f"Retrieving {stype.upper()} practice timing for {year} Round {round_num}...")
            practice_rows = self.session_service.get_practice_classification(year, round_num, session_type=stype)
            if not practice_rows:
                print_warning(f"No {stype.upper()} telemetry available for {year} Round {round_num} (or session not held).")
            else:
                headers = ["Pos", "Driver", "Best Lap", "Gap", "Laps"]
                table_rows = [[str(r["position"]), r["driver"], r["best_lap_time"], r["gap"], str(r["laps_completed"])] for r in practice_rows]
                print_header(f"{year} Round {round_num} — {stype.upper()} Classification", f"{len(practice_rows)} Drivers Recorded")
                print_table(headers, table_rows)

        prompt_str("\nPress [Enter] to return to Historical Menu", allow_empty=True)

        prompt_str("\nPress [Enter] to return to Historical Menu", allow_empty=True)

    # =========================================================================
    # 8. Constructor & Engine Dominance Analytics (New Option 8)
    # =========================================================================
    def view_constructor_dominance(self) -> None:
        """Guided workflow for constructor and engine dominance analytics."""
        print_header("Constructor & Engine Dominance", "All-Time Constructor Supremacy & Win Rates (1950–2026)")
        print("  1. All-Time Constructor Leaderboard (Championships, Wins, Poles, 1-2s)")
        print("  2. Highest Win-Rate Seasons in F1 History (Dominant Cars & Seasons)")
        print("  3. All-Time 1-2 Finish Rankings")
        print(f"\n  {Colors.RED}0.{Colors.RESET} Back\n")
        sub_choice = prompt_choice("Select dominance metric", ["1", "2", "3", "0"], default="1")
        if sub_choice == "0":
            return

        if sub_choice == "1":
            print_info("Compiling all-time constructor dominance rankings...")
            records = self.dominance_service.get_constructor_dominance()
            headers = ["Rank", "Constructor", "Championships", "Wins", "Poles", "Podiums", "1-2s", "Starts", "Win %"]
            rows = [
                [
                    str(r["rank"]),
                    r["constructor"],
                    str(r["championships"]),
                    str(r["wins"]),
                    str(r["poles"]),
                    str(r["podiums"]),
                    str(r["one_twos"]),
                    str(r["starts"]),
                    f"{r['win_percentage']:.1f}%",
                ]
                for r in records
            ]
            print_header("All-Time Constructor Rankings", "Sorted by World Championships & Grand Prix Wins")
            print_table(headers, rows)

        elif sub_choice == "2":
            print_info("Fetching highest win-rate single seasons in F1 history...")
            seasons = self.dominance_service.get_highest_win_rate_seasons(top_n=15)
            headers = ["Rank", "Season", "Constructor", "Wins/Races", "Win %", "Champion Driver", "Historical Context"]
            rows = [
                [
                    str(s["rank"]),
                    str(s["season"]),
                    s["constructor"],
                    f"{s['wins']}/{s['races']}",
                    f"{s['win_pct']:.1f}%",
                    s["driver_champs"],
                    s["notes"][:45],
                ]
                for s in seasons
            ]
            print_header("Top 15 Most Dominant Single Seasons in F1 History", "Minimum 8 Races Held")
            print_table(headers, rows)

        elif sub_choice == "3":
            print_info("Fetching all-time 1-2 finish leaderboards...")
            teams = self.dominance_service.get_one_two_finish_rankings(top_n=12)
            headers = ["Rank", "Constructor", "1-2 Finishes", "Total Wins", "1-2 % of Wins"]
            rows = [
                [
                    str(t["rank"]),
                    t["constructor"],
                    str(t["one_twos"]),
                    str(t["wins"]),
                    f"{t['one_two_pct_of_wins']:.1f}%",
                ]
                for t in teams
            ]
            print_header("All-Time Constructor 1-2 Finishes", "Both Drivers Finishing P1 and P2")
            print_table(headers, rows)

        prompt_str("\nPress [Enter] to return to Historical Menu", allow_empty=True)

    # =========================================================================
    # 9. All-Time Records, Streaks & Milestones (New Option 9)
    # =========================================================================
    def view_all_time_records(self) -> None:
        """Guided workflow for all-time driver records, streaks, and milestones."""
        print_header("All-Time Records & Milestones", "Poles, Wins, Podiums, Streaks & Age Records (1950–2026)")
        print("  1. All-Time Grand Prix Wins (Top 20)")
        print("  2. All-Time Pole Positions (Top 20)")
        print("  3. All-Time Podiums (Top 15)")
        print("  4. Consecutive Streaks (Wins, Podiums, Points, Starts)")
        print("  5. Age Milestones & Unwanted Records (Youngest/Oldest, Most Starts No Win)")
        print(f"\n  {Colors.RED}0.{Colors.RESET} Back\n")
        sub_choice = prompt_choice("Select records view", ["1", "2", "3", "4", "5", "0"], default="1")
        if sub_choice == "0":
            return

        if sub_choice == "1":
            records = self.records_service.get_all_time_wins(top_n=20)
            headers = ["Rank", "Driver", "Nationality", "Wins", "Starts", "Win %"]
            rows = [[str(r["rank"]), r["driver"], r["nationality"], str(r["wins"]), str(r["starts"]), f"{r['win_pct']:.1f}%"] for r in records]
            print_header("All-Time Formula 1 Race Winners", "Top 20 Drivers in History")
            print_table(headers, rows)

        elif sub_choice == "2":
            records = self.records_service.get_all_time_poles(top_n=20)
            headers = ["Rank", "Driver", "Nationality", "Poles", "Starts", "Pole %"]
            rows = [[str(r["rank"]), r["driver"], r["nationality"], str(r["poles"]), str(r["starts"]), f"{r['pole_pct']:.1f}%"] for r in records]
            print_header("All-Time Formula 1 Pole Positions", "Top 20 Drivers in History")
            print_table(headers, rows)

        elif sub_choice == "3":
            records = self.records_service.get_all_time_podiums(top_n=15)
            headers = ["Rank", "Driver", "Nationality", "Podiums", "Starts", "Podium %"]
            rows = [[str(r["rank"]), r["driver"], r["nationality"], str(r["podiums"]), str(r["starts"]), f"{r['podium_pct']:.1f}%"] for r in records]
            print_header("All-Time Formula 1 Podium Finishes", "Top 15 Drivers in History")
            print_table(headers, rows)

        elif sub_choice == "4":
            print("\nSelect Streak Type:")
            print("  1. Consecutive Wins")
            print("  2. Consecutive Podiums")
            print("  3. Consecutive Points Finishes")
            print("  4. Consecutive Race Starts")
            print(f"\n  {Colors.RED}0.{Colors.RESET} Back\n")
            streak_sel = prompt_choice("Select streak", ["1", "2", "3", "4", "0"], default="1")
            if streak_sel == "0":
                return
            streak_key = {"1": "consecutive_wins", "2": "consecutive_podiums", "3": "consecutive_points", "4": "consecutive_starts"}[streak_sel]
            streaks = self.records_service.get_consecutive_records(streak_key)
            headers = ["Rank", "Driver", "Streak Count", "Season", "Grand Prix Span"]
            rows = [[str(s["rank"]), s["driver"], str(s["count"]), s["season"], s["races"]] for s in streaks]
            title = streak_key.replace("_", " ").title()
            print_header(f"All-Time Streaks: {title}", "Longest Unbroken Runs in F1 History")
            print_table(headers, rows)

        elif sub_choice == "5":
            milestones = self.records_service.get_age_milestones()
            print(f"\n{Colors.BOLD}Youngest Grand Prix Winners:{Colors.RESET}")
            y_headers = ["Rank", "Driver", "Age", "Event"]
            y_rows = [[str(x["rank"]), x["driver"], x["age"], x["event"]] for x in milestones["youngest_winners"]]
            print_table(y_headers, y_rows)

            print(f"\n{Colors.BOLD}Oldest Grand Prix Winners:{Colors.RESET}")
            o_headers = ["Rank", "Driver", "Age", "Event"]
            o_rows = [[str(x["rank"]), x["driver"], x["age"], x["event"]] for x in milestones["oldest_winners"]]
            print_table(o_headers, o_rows)

            print(f"\n{Colors.BOLD}Most Race Starts Without a Grand Prix Win:{Colors.RESET}")
            u_headers = ["Rank", "Driver", "Starts", "Best Finish"]
            u_rows = [[str(x["rank"]), x["driver"], str(x["starts"]), x["best_finish"]] for x in milestones["most_starts_no_win"]]
            print_table(u_headers, u_rows)

        prompt_str("\nPress [Enter] to return to Historical Menu", allow_empty=True)

    # =========================================================================
    # 10. Historical Reliability & Retirement Stats (New Option 10)
    # =========================================================================
    def view_reliability_stats(self) -> None:
        """Guided workflow for reliability, DNF causes, and attrition trends."""
        print_header("Historical Reliability & Retirement Stats", "Mechanical Attrition & Finish Rates Across Decades")
        print("  1. Era-by-Era Attrition Trends (1950s to 2026)")
        print("  2. Most Attritional Races in F1 History (Fewest Finishers)")
        print("  3. Modern Era Constructor Reliability Rankings")
        print(f"\n  {Colors.RED}0.{Colors.RESET} Back\n")
        sub_choice = prompt_choice("Select reliability view", ["1", "2", "3", "0"], default="1")
        if sub_choice == "0":
            return

        if sub_choice == "1":
            eras = self.reliability_service.get_season_attrition_rates()
            headers = ["Era", "Races", "Finish Rate %", "Mech. DNF %", "Accident %", "Avg Finishers"]
            rows = [
                [
                    e["era"],
                    str(e["races"]),
                    f"{e['finish_rate_pct']:.1f}%",
                    f"{e['mech_dnf_pct']:.1f}%",
                    f"{e['accident_dnf_pct']:.1f}%",
                    f"{e['avg_finishers']:.1f}",
                ]
                for e in eras
            ]
            print_header("Historical Attrition Evolution (1950–2026)", "Mechanical Failure Decline Across Decades")
            print_table(headers, rows)

        elif sub_choice == "2":
            races = self.reliability_service.get_most_attritional_races(top_n=8)
            headers = ["Grand Prix", "Year", "Starters", "Finishers", "Winner", "Race Context"]
            rows = [[r["race"], str(r["year"]), str(r["starters"]), str(r["finishers"]), r["winner"], r["notes"][:45]] for r in races]
            print_header("Most Attritional Races in History", "Events with the Fewest Checkered Flag Finishers")
            print_table(headers, rows)

        elif sub_choice == "3":
            constructors = self.reliability_service.get_most_reliable_constructors(top_n=5)
            headers = ["Rank", "Constructor", "Regulatory Era", "Race Starts", "Finish Rate %", "Engine PU"]
            rows = [
                [
                    str(c["rank"]),
                    c["constructor"],
                    c["era"],
                    str(c["starts"]),
                    f"{c['finish_rate_pct']:.1f}%",
                    c["engine"],
                ]
                for c in constructors
            ]
            print_header("Constructor Reliability Rankings (V6 Turbo-Hybrid)", "Lowest Mechanical Attrition Rates")
            print_table(headers, rows)

        prompt_str("\nPress [Enter] to return to Historical Menu", allow_empty=True)

    # =========================================================================
    # 11. F1 Eras & Regulation Comparison (New Option 11)
    # =========================================================================
    def view_era_comparisons(self) -> None:
        """Guided workflow for regulatory and technical era comparisons."""
        print_header("F1 Eras & Regulation Comparisons (1950–2026)", "Technical Architecture & Macro-Metrics Across 8 Eras")
        eras = self.era_service.get_era_overview()

        headers = ["Era", "Period", "Avg Races/Yr", "Avg Grid", "Finish Rate %", "Winners/Yr", "Teams/Yr"]
        rows = [
            [
                e["name"],
                e["period"],
                f"{e['avg_races_per_year']:.1f}",
                f"{e['avg_grid_size']:.1f}",
                f"{e['finish_rate_pct']:.1f}%",
                f"{e['unique_winners_per_year']:.1f}",
                f"{e['unique_winning_teams_per_yr']:.1f}",
            ]
            for e in eras
        ]
        print_header("The 8 Canonical Technical Eras of Formula 1", "Macro Metrics Comparison")
        print_table(headers, rows)

        print(f"\n{Colors.BOLD}Technical Architecture & Defining Innovations:{Colors.RESET}")
        for e in eras:
            print(f"  * {Colors.BOLD}{e['name']} ({e['period']}){Colors.RESET}: {e['defining_characteristics']}")

        prompt_str("\nPress [Enter] to return to Historical Menu", allow_empty=True)
