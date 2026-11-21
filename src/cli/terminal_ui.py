"""
Terminal UI Utilities and Design System for ApexGrid Motorsport Intelligence Platform.
Predicting the Race Beyond the Grid.

Provides rich terminal presentation with graceful fallback to standard ANSI/ASCII
for non-TTY environments, piped output, and terminals without Rich or color support.
"""

import os
import sys
import shutil
from typing import List, Optional, Any, Sequence, Dict, Tuple

try:
    import rich
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.theme import Theme
    from rich.text import Text
    from rich.align import Align
    from rich.columns import Columns
    from rich.box import ROUNDED, SIMPLE, HORIZONTALS, SIMPLE_HEAVY, DOUBLE_EDGE
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.rule import Rule
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# Check color support
def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    if sys.platform == "win32":
        return (
            os.environ.get("ANSICON") is not None
            or "WT_SESSION" in os.environ
            or os.environ.get("TERM_PROGRAM") is not None
            or os.environ.get("COLORTERM") is not None
            or (hasattr(sys, "getwindowsversion") and sys.getwindowsversion().major >= 10)
        )
    return True


SUPPORTS_COLOR = _supports_color()


class Colors:
    """Standard ANSI color escape sequences for backward compatibility."""
    RESET = "\033[0m" if SUPPORTS_COLOR else ""
    BOLD = "\033[1m" if SUPPORTS_COLOR else ""
    DIM = "\033[2m" if SUPPORTS_COLOR else ""
    RED = "\033[91m" if SUPPORTS_COLOR else ""
    GREEN = "\033[92m" if SUPPORTS_COLOR else ""
    YELLOW = "\033[93m" if SUPPORTS_COLOR else ""
    BLUE = "\033[94m" if SUPPORTS_COLOR else ""
    MAGENTA = "\033[95m" if SUPPORTS_COLOR else ""
    CYAN = "\033[96m" if SUPPORTS_COLOR else ""
    WHITE = "\033[97m" if SUPPORTS_COLOR else ""


# Motorsport Palette Theme
APEX_THEME = None
if HAS_RICH:
    APEX_THEME = Theme({
        "info": "bold #00B4D8",
        "warning": "bold #FFB800",
        "error": "bold #E10600",
        "success": "bold #00D084",
        "brand": "bold #E10600",
        "purple": "bold #B138DD",
        "cyan": "#00B4D8",
        "muted": "#8E8E93",
        "highlight": "bold #F5F5F7",
        "gold": "bold #FFD700",
        "silver": "bold #C0C0C0",
        "bronze": "bold #CD7F32",
    })

_CONSOLE_INSTANCE: Optional[Any] = None


def get_console() -> Optional[Any]:
    """Returns the Rich Console configured for ApexGrid or None if Rich is absent."""
    global _CONSOLE_INSTANCE
    if not HAS_RICH:
        return None
    no_color = bool(os.environ.get("NO_COLOR"))
    # Create or refresh console instance when NO_COLOR state changes
    if _CONSOLE_INSTANCE is None or getattr(_CONSOLE_INSTANCE, "_apex_no_color", None) != no_color:
        _CONSOLE_INSTANCE = Console(
            theme=APEX_THEME,
            highlight=False,
            no_color=no_color,
            color_system=None if (no_color or not _supports_color()) else "auto",
        )
        _CONSOLE_INSTANCE._apex_no_color = no_color
    return _CONSOLE_INSTANCE


def clear_screen() -> None:
    """Clears the console screen safely."""
    os.system("cls" if os.name == "nt" else "clear")


def print_banner(season: int = 2026, status_text: str = "Motorsport Intelligence Ready") -> None:
    """Displays the main ApexGrid motorsport intelligence header banner."""
    console = get_console()
    if HAS_RICH and console and _supports_color():
        brand_text = Text()
        brand_text.append("A P E X G R I D\n", style="bold #E10600")
        brand_text.append("Predicting the Race Beyond the Grid", style="bold #00B4D8")
        panel = Panel(
            Align.center(brand_text),
            box=ROUNDED,
            border_style="#E10600",
            padding=(0, 2),
            subtitle=f"[bold #F5F5F7]Season: {season}[/]  [#8E8E93]|[/]  [bold #00D084]Status: {status_text}[/]",
            subtitle_align="center",
        )
        console.print(panel)
        console.print()
    else:
        width = 68
        border = "=" * width
        print(Colors.CYAN + border + Colors.RESET)
        print(Colors.BOLD + Colors.WHITE + "             ApexGrid".center(width) + Colors.RESET)
        print(Colors.CYAN + "       Predicting the Race Beyond the Grid".center(width) + Colors.RESET)
        print(Colors.CYAN + border + Colors.RESET)
        print(f" Season: {Colors.BOLD}{season}{Colors.RESET} | Status: {Colors.GREEN}{status_text}{Colors.RESET}")
        print(border + "\n")


def print_header(title: str, subtitle: Optional[str] = None) -> None:
    """Prints a distinct section header."""
    console = get_console()
    if HAS_RICH and console and _supports_color():
        console.print()
        console.rule(f"[bold #00B4D8]{title.upper()}[/]", style="#8E8E93")
        if subtitle:
            console.print(Align.center(f"[dim]{subtitle}[/]"))
        console.print()
    else:
        width = 68
        print("\n" + Colors.CYAN + "-" * width + Colors.RESET)
        print(Colors.BOLD + Colors.WHITE + f"  {title.upper()}" + Colors.RESET)
        if subtitle:
            print(Colors.DIM + f"  {subtitle}" + Colors.RESET)
        print(Colors.CYAN + "-" * width + Colors.RESET)


def print_success(message: str) -> None:
    """Prints an operation success status line."""
    console = get_console()
    if HAS_RICH and console and _supports_color():
        console.print(f" [bold #00D084]✓ READY[/]  {message}")
    else:
        print(f" {Colors.GREEN}[OK]{Colors.RESET} {message}")


def print_warning(message: str) -> None:
    """Prints an operation warning status line."""
    console = get_console()
    if HAS_RICH and console and _supports_color():
        console.print(f" [bold #FFB800]▲ WARN[/]   {message}")
    else:
        print(f" {Colors.YELLOW}[WARN]{Colors.RESET} {message}")


def print_error(message: str) -> None:
    """Prints an operation error status line."""
    console = get_console()
    if HAS_RICH and console and _supports_color():
        console.print(f" [bold #E10600]✖ ERROR[/]  {message}")
    else:
        print(f" {Colors.RED}[ERROR]{Colors.RESET} {message}")


def print_info(message: str) -> None:
    """Prints an informational status line."""
    console = get_console()
    if HAS_RICH and console and _supports_color():
        console.print(f" [bold #00B4D8]ℹ INFO[/]   {message}")
    else:
        print(f" {Colors.BLUE}[INFO]{Colors.RESET} {message}")


def prompt_choice(prompt_text: str, valid_choices: Sequence[str], default: Optional[str] = None) -> str:
    """
    Prompts the user for a single choice among valid_choices.
    Validates input loop until a valid choice is entered or EOF/Ctrl+C is received.
    """
    valid_set = {str(c).strip().lower() for c in valid_choices}
    default_str = f" [{default}]" if default else ""

    while True:
        try:
            raw = input(f"{Colors.BOLD}{prompt_text}{default_str}:{Colors.RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n")
            return "0"

        if not raw and default:
            return default

        val = raw.lower()
        if val in valid_set:
            for orig in valid_choices:
                if str(orig).strip().lower() == val:
                    return str(orig)
            return val

        print_warning(f"Invalid option '{raw}'. Please enter one of: {', '.join(valid_choices)}")


def prompt_int(
    prompt_text: str,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
    default: Optional[int] = None,
    allowed_values: Optional[List[int]] = None,
) -> Optional[int]:
    """Prompts the user for an integer within an optional range or explicit allowed values."""
    range_hint = ""
    if allowed_values is not None:
        sorted_vals = sorted(allowed_values)
        if len(sorted_vals) <= 1:
            range_hint = ""
        elif sorted_vals == list(range(sorted_vals[0], sorted_vals[-1] + 1)):
            range_hint = f" ({sorted_vals[0]}-{sorted_vals[-1]})"
        else:
            range_hint = f" ({', '.join(str(v) for v in sorted_vals)})"
    elif min_val is not None and max_val is not None:
        if min_val == max_val:
            range_hint = ""
        else:
            range_hint = f" ({min_val}-{max_val})"
    elif min_val is not None:
        range_hint = f" (>= {min_val})"
    default_str = f" [{default}]" if default is not None else ""

    while True:
        try:
            raw = input(f"{Colors.BOLD}{prompt_text}{range_hint}{default_str}:{Colors.RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n")
            return None

        if not raw and default is not None:
            return default

        try:
            val = int(raw)
            if allowed_values is not None and val not in allowed_values:
                print_warning(f"Value must be one of: {sorted(allowed_values)}.")
                continue
            if min_val is not None and val < min_val:
                print_warning(f"Value must be at least {min_val}.")
                continue
            if max_val is not None and val > max_val:
                print_warning(f"Value must be at most {max_val}.")
                continue
            return val
        except ValueError:
            print_warning(f"Please enter a valid integer.")


def prompt_str(prompt_text: str, default: Optional[str] = None, allow_empty: bool = False) -> str:
    """Prompts for a string input with optional default."""
    default_str = f" [{default}]" if default else ""
    while True:
        try:
            raw = input(f"{Colors.BOLD}{prompt_text}{default_str}:{Colors.RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n")
            return ""

        if not raw and default:
            return default
        if not raw and not allow_empty:
            print_warning("Input cannot be empty.")
            continue
        return raw


def prompt_confirm(prompt_text: str, default: bool = True) -> bool:
    """Asks for confirmation (y/n)."""
    hint = " [Y/n]" if default else " [y/N]"
    while True:
        try:
            raw = input(f"{Colors.BOLD}{prompt_text}{hint}:{Colors.RESET} ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n")
            return False

        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print_warning("Please respond with 'y' or 'n'.")


def print_table(
    headers: List[str],
    rows: List[List[Any]],
    align_right: Optional[List[bool]] = None,
    title: Optional[str] = None,
    box_style: Optional[Any] = None,
) -> None:
    """Prints a styled tabular output with Rich formatting or ASCII border fallback."""
    if not rows:
        print("  (No records found)\n")
        return

    console = get_console()
    if HAS_RICH and console and _supports_color():
        box_type = box_style or ROUNDED
        table = Table(box=box_type, header_style="bold #00B4D8", border_style="#8E8E93", title=title)
        for i, h in enumerate(headers):
            justify = "right" if (align_right and i < len(align_right) and align_right[i]) else "left"
            table.add_column(h, justify=justify)
        for row in rows:
            str_row = [str(cell) if cell is not None else "" for cell in row]
            table.add_row(*str_row)
        console.print(table)
        console.print()
    else:
        num_cols = len(headers)
        str_rows = [[str(cell) if cell is not None else "" for cell in row] for row in rows]
        col_widths = [len(h) for h in headers]
        for row in str_rows:
            for i in range(num_cols):
                if i < len(row):
                    col_widths[i] = max(col_widths[i], len(row[i]))

        header_line = " | ".join(
            h.rjust(col_widths[i]) if (align_right and i < len(align_right) and align_right[i]) else h.ljust(col_widths[i])
            for i, h in enumerate(headers)
        )
        separator = "-+-".join("-" * col_widths[i] for i in range(num_cols))

        print(Colors.BOLD + header_line + Colors.RESET)
        print(separator)
        for row in str_rows:
            line = " | ".join(
                (row[i] if i < len(row) else "").rjust(col_widths[i])
                if (align_right and i < len(align_right) and align_right[i])
                else (row[i] if i < len(row) else "").ljust(col_widths[i])
                for i in range(num_cols)
            )
            print(line)
        print()


def paginate_text(text: str, page_size: int = 30) -> None:
    """Displays text paginated with [Enter] for next page or 'q' to quit."""
    lines = text.splitlines()
    total = len(lines)
    if total <= page_size:
        print(text)
        return

    idx = 0
    while idx < total:
        chunk = lines[idx : idx + page_size]
        print("\n".join(chunk))
        idx += page_size
        if idx < total:
            try:
                resp = input(f"\n{Colors.DIM}-- Press [Enter] for more ({idx}/{total}), or 'q' to stop --{Colors.RESET} ").strip().lower()
                if resp == "q":
                    break
            except (KeyboardInterrupt, EOFError):
                break
    print()


def render_probability_bar(pct: float, width: int = 10) -> str:
    """Renders a visual probability bar string e.g. [██████░░░░] 60.0%."""
    if pct < 0.0:
        pct = 0.0
    if pct > 100.0:
        pct = 100.0
    filled = int(round((pct / 100.0) * width))
    empty = width - filled
    bar = "█" * filled + "░" * empty

    if HAS_RICH and _supports_color():
        if pct >= 50.0:
            color = "#00D084"
        elif pct >= 20.0:
            color = "#00B4D8"
        elif pct >= 5.0:
            color = "#FFB800"
        else:
            color = "#8E8E93"
        return f"[{color}]{bar}[/] {pct:5.1f}%"
    return f"{bar} {pct:5.1f}%"


def print_podium(p1: Dict[str, Any], p2: Optional[Dict[str, Any]] = None, p3: Optional[Dict[str, Any]] = None) -> None:
    """Displays a stylized podium showcase for P1, P2, and P3."""
    console = get_console()
    if HAS_RICH and console and _supports_color():
        cards = []

        # P2 Card
        if p2:
            t2 = Text()
            t2.append("🥈 P2 PODIUM\n", style="bold #C0C0C0")
            t2.append(f"{p2.get('Driver', p2.get('Abbreviation', 'P2'))}\n", style="bold white")
            t2.append(f"{p2.get('Team', p2.get('TeamName', ''))[:18]}\n", style="#8E8E93")
            win_pct = p2.get("WinProbability", 0.0)
            pod_pct = p2.get("Top3Probability", 0.0)
            t2.append(f"Win: {win_pct:.1f}% | Pod: {pod_pct:.1f}%", style="cyan")
            cards.append(Panel(Align.center(t2), box=ROUNDED, border_style="#C0C0C0", width=24))

        # P1 Card (Center, highlighted)
        if p1:
            t1 = Text()
            t1.append("🏆 P1 WINNER\n", style="bold #FFD700")
            t1.append(f"{p1.get('Driver', p1.get('Abbreviation', 'P1'))}\n", style="bold white")
            t1.append(f"{p1.get('Team', p1.get('TeamName', ''))[:18]}\n", style="#8E8E93")
            win_pct = p1.get("WinProbability", 0.0)
            pod_pct = p1.get("Top3Probability", 0.0)
            t1.append(f"Win: {win_pct:.1f}% | Pod: {pod_pct:.1f}%", style="bold #00D084")
            cards.append(Panel(Align.center(t1), box=ROUNDED, border_style="#FFD700", width=26))

        # P3 Card
        if p3:
            t3 = Text()
            t3.append("🥉 P3 PODIUM\n", style="bold #CD7F32")
            t3.append(f"{p3.get('Driver', p3.get('Abbreviation', 'P3'))}\n", style="bold white")
            t3.append(f"{p3.get('Team', p3.get('TeamName', ''))[:18]}\n", style="#8E8E93")
            win_pct = p3.get("WinProbability", 0.0)
            pod_pct = p3.get("Top3Probability", 0.0)
            t3.append(f"Win: {win_pct:.1f}% | Pod: {pod_pct:.1f}%", style="cyan")
            cards.append(Panel(Align.center(t3), box=ROUNDED, border_style="#CD7F32", width=24))

        if cards:
            console.print(Align.center(Columns(cards)))
            console.print()
    else:
        # Fallback simple podium text
        print(f"\n  [PODIUM] P1: {p1.get('Abbreviation', 'P1')} | P2: {p2.get('Abbreviation', '-') if p2 else '-'} | P3: {p3.get('Abbreviation', '-') if p3 else '-'}\n")
