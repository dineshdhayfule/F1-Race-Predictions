"""
Terminal UI Utilities and Design System for ApexGrid Motorsport Intelligence Platform.
Predicting the Race Beyond the Grid.

Provides rich terminal presentation with graceful fallback to standard ANSI/ASCII
for non-TTY environments, piped output, and terminals without Rich or color support.
"""

import os
import sys
import shutil
import re
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


ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def strip_ansi(text: Any) -> str:
    """Strips ANSI escape sequences from text for accurate visible length calculations."""
    if text is None:
        return ""
    return ANSI_ESCAPE_RE.sub("", str(text))


def get_terminal_width(default: int = 80, min_w: int = 40, max_w: int = 120) -> int:
    """
    Returns detected terminal width clamped within [min_w, max_w].
    Queries terminal dimensions dynamically on each call to respond to window resizing.
    """
    try:
        size = shutil.get_terminal_size(fallback=(default, 24))
        width = size.columns
    except Exception:
        width = default
    if not isinstance(width, int) or width <= 0:
        width = default
    return max(min_w, min(max_w, width))


def get_terminal_height(default: int = 24, min_h: int = 10) -> int:
    """
    Returns detected terminal height clamped to at least min_h.
    Queries terminal dimensions dynamically on each call.
    """
    try:
        size = shutil.get_terminal_size(fallback=(80, default))
        height = size.lines
    except Exception:
        height = default
    if not isinstance(height, int) or height <= 0:
        height = default
    return max(min_h, height)


def format_menu_divider(title: str, color_style: Optional[str] = None) -> str:
    """Formats a dynamic-width menu section divider (e.g. ── Core Pipeline ──────)."""
    term_width = get_terminal_width()
    prefix = f"── {title} "
    vis_len = len(prefix)
    dash_count = max(4, term_width - vis_len - 4)
    line = f"{prefix}{'─' * dash_count}"
    if color_style and HAS_RICH and _supports_color():
        return f"[{color_style}]{line}[/]"
    return line


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


def print_banner(season: Optional[int] = None, status_text: str = "Motorsport Intelligence Ready") -> None:
    """Displays the main ApexGrid motorsport intelligence header banner adapting to terminal width."""
    if season is None:
        from src.utils.season_utils import get_active_season
        season = get_active_season()
    term_width = get_terminal_width()
    console = get_console()
    if HAS_RICH and console and _supports_color():
        brand_text = Text()
        brand_text.append("A P E X G R I D\n", style="bold #E10600")
        tagline = "Predicting the Race Beyond the Grid"
        if term_width < 45:
            tagline = "Race Intelligence"
        brand_text.append(tagline, style="bold #00B4D8")

        if term_width < 60:
            subtitle = f"[bold #F5F5F7]S{season}[/] | [bold #00D084]{status_text[:max(10, term_width - 16)]}[/]"
        else:
            subtitle = f"[bold #F5F5F7]Season: {season}[/]  [#8E8E93]|[/]  [bold #00D084]Status: {status_text}[/]"

        panel = Panel(
            Align.center(brand_text),
            box=ROUNDED,
            border_style="#E10600",
            padding=(0, 1 if term_width < 50 else 2),
            subtitle=subtitle,
            subtitle_align="center",
            width=term_width,
        )
        console.print(panel)
        console.print()
    else:
        width = term_width
        border = "=" * width
        title_str = "ApexGrid"
        tagline_str = "Predicting the Race Beyond the Grid"
        if width < len(tagline_str) + 4:
            tagline_str = "Race Intelligence"

        print(Colors.CYAN + border + Colors.RESET)
        print(Colors.BOLD + Colors.WHITE + title_str.center(width) + Colors.RESET)
        print(Colors.CYAN + tagline_str.center(width) + Colors.RESET)
        print(Colors.CYAN + border + Colors.RESET)

        status_line = f" Season: {season} | Status: {status_text}"
        if len(status_line) > width - 2:
            print(f" Season: {Colors.BOLD}{season}{Colors.RESET}")
            print(f" Status: {Colors.GREEN}{status_text}{Colors.RESET}")
        else:
            print(f" Season: {Colors.BOLD}{season}{Colors.RESET} | Status: {Colors.GREEN}{status_text}{Colors.RESET}")
        print(border + "\n")


def print_header(title: str, subtitle: Optional[str] = None) -> None:
    """Prints a distinct section header adapting to terminal width."""
    term_width = get_terminal_width()
    console = get_console()
    if HAS_RICH and console and _supports_color():
        console.print()
        header_title = title.upper()
        if len(header_title) > term_width - 6:
            header_title = header_title[:max(10, term_width - 9)] + "..."
        console.rule(f"[bold #00B4D8]{header_title}[/]", style="#8E8E93")
        if subtitle:
            sub = subtitle
            if len(sub) > term_width - 4:
                sub = sub[:max(10, term_width - 7)] + "..."
            console.print(Align.center(f"[dim]{sub}[/]"))
        console.print()
    else:
        width = term_width
        print("\n" + Colors.CYAN + "─" * width + Colors.RESET)
        disp_title = title.upper()
        if len(disp_title) > width - 4:
            disp_title = disp_title[:max(10, width - 7)] + "..."
        print(Colors.BOLD + Colors.WHITE + f"  {disp_title}" + Colors.RESET)
        if subtitle:
            disp_sub = subtitle
            if len(disp_sub) > width - 4:
                disp_sub = disp_sub[:max(10, width - 7)] + "..."
            print(Colors.DIM + f"  {disp_sub}" + Colors.RESET)
        print(Colors.CYAN + "─" * width + Colors.RESET)


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
    allow_zero_back: bool = False,
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
            if allow_zero_back and val == 0:
                return 0
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


def _pad_ansi(text: str, width: int, align_right: bool = False) -> str:
    """Pads text to width taking into account invisible ANSI escape sequences."""
    vis_len = len(strip_ansi(text))
    pad = max(0, width - vis_len)
    if align_right:
        return (" " * pad) + text
    return text + (" " * pad)


def _truncate_table_columns(
    headers: List[str],
    str_rows: List[List[str]],
    col_widths: List[int],
    term_width: int,
    align_right: Optional[List[bool]] = None,
) -> Tuple[List[List[str]], List[int]]:
    """
    Intelligently truncates wider text columns to fit within term_width.
    Preserves numeric and short code columns intact.
    """
    num_cols = len(headers)
    divider_width = 3 * (num_cols - 1)
    avail_width = max(num_cols * 3, term_width - divider_width)

    new_widths = list(col_widths)
    excess = sum(new_widths) - avail_width

    if excess > 0:
        shrinkable = [
            i for i in range(num_cols)
            if not (align_right and i < len(align_right) and align_right[i] and new_widths[i] <= 8)
        ]
        while excess > 0 and shrinkable:
            max_col = max(shrinkable, key=lambda c: new_widths[c])
            if new_widths[max_col] <= 6:
                break
            step = min(excess, max(1, (new_widths[max_col] - 6) // 2))
            new_widths[max_col] -= step
            excess -= step

    new_rows = []
    for row in str_rows:
        new_row = []
        for i in range(num_cols):
            cell = row[i] if i < len(row) else ""
            target_w = new_widths[i]
            vis_len = len(strip_ansi(cell))
            if vis_len > target_w:
                plain = strip_ansi(cell)
                if target_w > 3:
                    cell = plain[: target_w - 3] + "..."
                else:
                    cell = plain[:target_w]
            new_row.append(cell)
        new_rows.append(new_row)

    return new_rows, new_widths


def _print_ascii_table(
    headers: List[str],
    str_rows: List[List[str]],
    col_widths: List[int],
    align_right: Optional[List[bool]] = None,
) -> None:
    num_cols = len(headers)
    header_line = " | ".join(
        _pad_ansi(headers[i], col_widths[i], bool(align_right and i < len(align_right) and align_right[i]))
        for i in range(num_cols)
    )
    separator = "-+-".join("-" * col_widths[i] for i in range(num_cols))

    print(Colors.BOLD + header_line + Colors.RESET)
    print(separator)
    for row in str_rows:
        line = " | ".join(
            _pad_ansi(row[i] if i < len(row) else "", col_widths[i], bool(align_right and i < len(align_right) and align_right[i]))
            for i in range(num_cols)
        )
        print(line)
    print()


def _print_stacked_table(
    headers: List[str],
    str_rows: List[List[str]],
    title: Optional[str] = None,
    term_width: int = 40,
) -> None:
    """Displays table rows as stacked records when width is too narrow for tabular view."""
    if title:
        print_header(title)

    num_cols = len(headers)
    sep_width = max(20, min(term_width, 60))

    for idx, row in enumerate(str_rows, 1):
        suffix_len = max(0, sep_width - 16 - len(str(idx)) - len(str(len(str_rows))))
        print(Colors.CYAN + f"── Record {idx}/{len(str_rows)} " + ("─" * suffix_len) + Colors.RESET)
        for i in range(num_cols):
            if i < len(row) and row[i]:
                h = headers[i]
                v = row[i]
                print(f"  {Colors.BOLD}{h}{Colors.RESET}: {v}")
        print()


def print_table(
    headers: List[str],
    rows: List[List[Any]],
    align_right: Optional[List[bool]] = None,
    title: Optional[str] = None,
    box_style: Optional[Any] = None,
) -> None:
    """Prints a styled tabular output adapting dynamically to terminal width."""
    if not rows:
        print("  (No records found)\n")
        return

    term_width = get_terminal_width()
    num_cols = len(headers)
    str_rows = [[str(cell) if cell is not None else "" for cell in row] for row in rows]

    col_widths = [len(strip_ansi(h)) for h in headers]
    for row in str_rows:
        for i in range(num_cols):
            if i < len(row):
                col_widths[i] = max(col_widths[i], len(strip_ansi(row[i])))

    total_natural_width = sum(col_widths) + 3 * (num_cols - 1)
    min_needed = num_cols * 4 + 3 * (num_cols - 1)
    use_stacked = (term_width < min_needed and num_cols > 3) or (term_width < 45 and num_cols >= 5)

    if use_stacked:
        _print_stacked_table(headers, str_rows, title=title, term_width=term_width)
        return

    if total_natural_width > term_width:
        str_rows, col_widths = _truncate_table_columns(
            headers, str_rows, col_widths, term_width, align_right
        )

    console = get_console()
    if HAS_RICH and console and _supports_color():
        box_type = box_style or ROUNDED
        table = Table(box=box_type, header_style="bold #00B4D8", border_style="#8E8E93", title=title)
        for i, h in enumerate(headers):
            justify = "right" if (align_right and i < len(align_right) and align_right[i]) else "left"
            is_numeric = col_widths[i] <= 8 and bool(align_right and i < len(align_right) and align_right[i])
            table.add_column(h, justify=justify, no_wrap=is_numeric, overflow="ellipsis")
        for row in str_rows:
            table.add_row(*row)
        console.print(table)
        console.print()
    else:
        _print_ascii_table(headers, str_rows, col_widths, align_right)


def paginate_text(text: str, page_size: Optional[int] = None) -> None:
    """Displays text paginated with [Enter] for next page or 'q' to quit."""
    if page_size is None or page_size <= 0:
        page_size = max(5, get_terminal_height() - 4)

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

    if width == 10:
        tw = get_terminal_width()
        if tw < 50:
            width = 4
        elif tw < 75:
            width = 6

    filled = int(round((pct / 100.0) * width))
    empty = max(0, width - filled)
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


def print_podium(
    p1: Dict[str, Any],
    p2: Optional[Dict[str, Any]] = None,
    p3: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
) -> None:
    """Displays a stylized podium showcase for P1, P2, and P3."""
    term_width = get_terminal_width()
    console = get_console()
    if HAS_RICH and console and _supports_color():
        if term_width >= 78:
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
                if title:
                    console.print(Align.center(Text(title, style="bold #FFD700")))
                console.print(Align.center(Columns(cards)))
                console.print()
        else:
            podium_text = Text()
            if p1:
                p1_name = p1.get('Driver', p1.get('Abbreviation', 'P1'))
                p1_team = p1.get('Team', p1.get('TeamName', ''))[:18]
                p1_win = p1.get("WinProbability", 0.0)
                p1_pod = p1.get("Top3Probability", 0.0)
                podium_text.append(f"🏆 P1: {p1_name} ({p1_team})\n", style="bold #FFD700")
                podium_text.append(f"   Win: {p1_win:.1f}% | Pod: {p1_pod:.1f}%\n", style="bold #00D084")
            if p2:
                p2_name = p2.get('Driver', p2.get('Abbreviation', 'P2'))
                p2_team = p2.get('Team', p2.get('TeamName', ''))[:18]
                p2_win = p2.get("WinProbability", 0.0)
                p2_pod = p2.get("Top3Probability", 0.0)
                podium_text.append(f"🥈 P2: {p2_name} ({p2_team})\n", style="bold #C0C0C0")
                podium_text.append(f"   Win: {p2_win:.1f}% | Pod: {p2_pod:.1f}%\n", style="cyan")
            if p3:
                p3_name = p3.get('Driver', p3.get('Abbreviation', 'P3'))
                p3_team = p3.get('Team', p3.get('TeamName', ''))[:18]
                p3_win = p3.get("WinProbability", 0.0)
                p3_pod = p3.get("Top3Probability", 0.0)
                podium_text.append(f"🥉 P3: {p3_name} ({p3_team})\n", style="bold #CD7F32")
                podium_text.append(f"   Win: {p3_win:.1f}% | Pod: {p3_pod:.1f}%", style="cyan")

            card_width = min(term_width, 55)
            panel = Panel(
                podium_text,
                title=title or "[bold #FFD700]🏆 PODIUM SHOWCASE[/]",
                box=ROUNDED,
                border_style="#FFD700",
                width=card_width,
            )
            console.print(Align.center(panel))
            console.print()
    else:
        # Fallback simple podium text
        p1_code = p1.get('Abbreviation', p1.get('Driver', 'P1'))
        p2_code = p2.get('Abbreviation', p2.get('Driver', '-')) if p2 else '-'
        p3_code = p3.get('Abbreviation', p3.get('Driver', '-')) if p3 else '-'
        hdr = title if title else "[PODIUM]"
        if term_width >= 60:
            print(f"\n  {hdr} P1: {p1_code} | P2: {p2_code} | P3: {p3_code}\n")
        else:
            print(f"\n  {hdr}\n    P1: {p1_code}\n    P2: {p2_code}\n    P3: {p3_code}\n")
