"""
Terminal UI Utilities for Interactive CLI Menu.
Pure Python standard library implementation compatible with Windows PowerShell, CMD, and Unix terminals.
"""

import os
import sys
import shutil
from typing import List, Optional, Any, Sequence, Dict


# Check color support
def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    if sys.platform == "win32":
        return os.environ.get("ANSICON") is not None or "WT_SESSION" in os.environ or os.environ.get("TERM_PROGRAM") is not None
    return True


SUPPORTS_COLOR = _supports_color()


class Colors:
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


def clear_screen() -> None:
    """Clears the console screen safely."""
    # Use standard console clear
    os.system("cls" if os.name == "nt" else "clear")


def print_banner(season: int = 2026, status_text: str = "Unified Engine Ready") -> None:
    """Displays the main ASCII header banner."""
    width = 68
    border = "=" * width
    print(Colors.CYAN + border + Colors.RESET)
    print(Colors.BOLD + Colors.WHITE + "             FORMULA 1 RACE PREDICTION ENGINE".center(width) + Colors.RESET)
    print(Colors.CYAN + "                  Unified Machine Learning System".center(width) + Colors.RESET)
    print(Colors.CYAN + border + Colors.RESET)
    print(f" Season: {Colors.BOLD}{season}{Colors.RESET} | Engine Status: {Colors.GREEN}{status_text}{Colors.RESET}")
    print(border + "\n")


def print_header(title: str, subtitle: Optional[str] = None) -> None:
    """Prints a distinct section header."""
    width = 68
    print("\n" + Colors.CYAN + "-" * width + Colors.RESET)
    print(Colors.BOLD + Colors.WHITE + f"  {title.upper()}" + Colors.RESET)
    if subtitle:
        print(Colors.DIM + f"  {subtitle}" + Colors.RESET)
    print(Colors.CYAN + "-" * width + Colors.RESET)


def print_success(message: str) -> None:
    print(f" {Colors.GREEN}[OK]{Colors.RESET} {message}")


def print_warning(message: str) -> None:
    print(f" {Colors.YELLOW}[WARN]{Colors.RESET} {message}")


def print_error(message: str) -> None:
    print(f" {Colors.RED}[ERROR]{Colors.RESET} {message}")


def print_info(message: str) -> None:
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
            # Return the matching casing from original valid_choices
            for orig in valid_choices:
                if str(orig).strip().lower() == val:
                    return str(orig)
            return val

        print_warning(f"Invalid option '{raw}'. Please enter one of: {', '.join(valid_choices)}")


def prompt_int(
    prompt_text: str,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
    default: Optional[int] = None
) -> Optional[int]:
    """Prompts the user for an integer within an optional range."""
    range_hint = ""
    if min_val is not None and max_val is not None:
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


def print_table(headers: List[str], rows: List[List[Any]], align_right: Optional[List[bool]] = None) -> None:
    """Prints a nicely aligned tabular output with ASCII borders."""
    if not rows:
        print("  (No records found)\n")
        return

    num_cols = len(headers)
    str_rows = [[str(cell) for cell in row] for row in rows]

    col_widths = [len(h) for h in headers]
    for row in str_rows:
        for i in range(num_cols):
            if i < len(row):
                col_widths[i] = max(col_widths[i], len(row[i]))

    # Print header
    header_line = " | ".join(
        h.rjust(col_widths[i]) if (align_right and i < len(align_right) and align_right[i]) else h.ljust(col_widths[i])
        for i, h in enumerate(headers)
    )
    separator = "-+-".join("-" * col_widths[i] for i in range(num_cols))

    print(Colors.BOLD + header_line + Colors.RESET)
    print(separator)

    # Print rows
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
