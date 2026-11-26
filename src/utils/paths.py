"""
Centralized Dual-Mode Path Resolver
===================================
Supports both:
1. Developer Mode (`pip install -e .` inside a git clone): uses repo config/, data/, outputs/, logs/.
2. Standalone Pip Mode (installed wheel outside repo): reads bundled src/config/*.json and writes
   mutable runtime directories (outputs, data, cache, logs) under ~/.apexgrid/.
"""

from pathlib import Path
from typing import Optional

# Anchor to package/src root or repo root
_SRC_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SRC_DIR.parent


def is_dev_mode() -> bool:
    """True if running inside a git repo clone with local pyproject.toml and config/."""
    return (_REPO_ROOT / "pyproject.toml").exists() and (_REPO_ROOT / "config").is_dir()


def get_runtime_root() -> Path:
    """
    Returns the base mutable workspace root:
    - Dev mode: repository root
    - Standalone pip mode: ~/.apexgrid (created if missing)
    """
    if is_dev_mode():
        return _REPO_ROOT
    root = Path.home() / ".apexgrid"
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_config_path(filename: str) -> Path:
    """
    Resolve configuration files in priority:
    1. Local repo config/ (if dev mode)
    2. Bundled package config/ (src/config/)
    3. User home ~/.apexgrid/config/
    """
    if is_dev_mode():
        local_cfg = _REPO_ROOT / "config" / filename
        if local_cfg.exists():
            return local_cfg

    bundled_cfg = _SRC_DIR / "config" / filename
    if bundled_cfg.exists():
        return bundled_cfg

    user_cfg = Path.home() / ".apexgrid" / "config" / filename
    if user_cfg.exists():
        return user_cfg

    # Fallback to local config even if missing, to give clear FileNotFoundError
    return _REPO_ROOT / "config" / filename


def resolve_config_file(filename: str, root_dir: Optional[Path] = None) -> Path:
    """
    Resolve a config file, honoring an explicit root_dir/config/filename if it exists
    (e.g., in isolated unit tests), otherwise falling back to get_config_path().
    """
    if root_dir is not None:
        candidate = Path(root_dir) / "config" / filename
        if candidate.exists():
            return candidate
    return get_config_path(filename)


def get_workspace_dir(category: str) -> Path:
    """
    Resolve mutable working directories ('outputs', 'data/raw', 'data/cache', 'logs').
    In dev mode: returns repo_root / category.
    In standalone mode: returns ~/.apexgrid / category.
    Automatically creates the directory if it does not exist.
    """
    path = get_runtime_root() / category
    path.mkdir(parents=True, exist_ok=True)
    return path
