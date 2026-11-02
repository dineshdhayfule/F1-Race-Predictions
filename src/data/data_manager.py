"""
Safe Data Management and Deletion Engine
========================================
Provides selective, verified deletion of project-generated caches, datasets,
and prediction outputs with strict safety validations:
1. Preview before deletion (counts, exact paths, storage size).
2. Mandatory explicit confirmation (typed phrase for destructive operations).
3. Project-root boundary enforcement (rejects traversal, symlink escapes).
4. Centralized file protection (never deletes source code, configs, tests, docs, git metadata, audit logs).
5. Default protection of historical training datasets.
6. Persistent, append-only deletion audit logging.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
import json
import os
import shutil

from src.data.deletion_audit import DeletionAuditLogger
from src.utils.logging_utils import get_logger

logger = get_logger("data_manager")


class DataManager:
    """Safely manages and selectively deletes project-generated data artifacts."""

    PROTECTED_EXTENSIONS: Set[str] = {
        ".py", ".pyc", ".pyo", ".ini", ".toml", ".yaml", ".yml",
        ".env", ".gitignore", ".gitattributes", ".sh", ".bat", ".ps1"
    }

    PROTECTED_DIR_NAMES: Set[str] = {
        ".git", ".venv", "venv", "env", "src", "tests", "config",
        "docs", ".pytest_cache", ".idea", ".vscode", "logs"
    }

    def __init__(self, root_dir: Optional[Path] = None, audit_logger: Optional[DeletionAuditLogger] = None):
        self.root_dir = (root_dir or Path(__file__).resolve().parents[2]).resolve()
        self.audit_logger = audit_logger or DeletionAuditLogger(root_dir=self.root_dir)

        # Configured tracks
        self.tracks_config_path = self.root_dir / "config" / "tracks.json"
        self.tracks_config: Dict[str, Any] = {}
        if self.tracks_config_path.exists():
            try:
                with open(self.tracks_config_path, "r", encoding="utf-8") as f:
                    self.tracks_config = json.load(f).get("tracks", {})
            except Exception as e:
                logger.warning(f"Failed to load tracks config for data manager: {e}")

    # =========================================================================
    # PATH VALIDATION & SAFETY GUARDS
    # =========================================================================

    def validate_path(self, target: Path) -> Path:
        """
        Validates that a path is strictly inside the project root and safe to touch.
        Raises ValueError if outside root, traversal attempt, or resolving to project root.
        """
        resolved = target.resolve()

        if resolved == self.root_dir:
            raise ValueError(f"Safety violation: Target cannot be project root directory ({self.root_dir}).")

        try:
            resolved.relative_to(self.root_dir)
        except ValueError:
            raise ValueError(f"Safety violation: Target '{target}' resolves outside project root ({self.root_dir}).")

        # Disallow symlinks pointing outside root
        if target.is_symlink():
            link_target = target.readlink()
            link_resolved = (target.parent / link_target).resolve()
            try:
                link_resolved.relative_to(self.root_dir)
            except ValueError:
                raise ValueError(f"Safety violation: Symlink '{target}' points outside project root.")

        return resolved

    def is_protected(self, target: Path, include_historical: bool = False) -> Tuple[bool, str]:
        """
        Determines whether a file or directory is protected from deletion.
        Returns (is_protected: bool, reason: str).
        """
        try:
            resolved = self.validate_path(target)
        except ValueError as e:
            return True, str(e)

        rel = resolved.relative_to(self.root_dir)
        parts = rel.parts

        # Top-level directory protection
        if len(parts) >= 1 and parts[0] in self.PROTECTED_DIR_NAMES:
            return True, f"In protected system directory: '{parts[0]}'"

        # Extension protection
        if resolved.is_file() and resolved.suffix.lower() in self.PROTECTED_EXTENSIONS:
            return True, f"Protected file type extension: '{resolved.suffix}'"

        # Explicit documentation protection
        if resolved.name.lower().startswith("readme") or (len(parts) == 1 and resolved.suffix.lower() == ".md"):
            return True, f"Documentation file: '{resolved.name}'"

        # Protect executive reports in outputs root
        if len(parts) == 2 and parts[0] == "outputs" and resolved.name.startswith("EXECUTIVE_REPORT"):
            return True, f"Protected executive architecture documentation: '{resolved.name}'"

        # Protect official FIA starting grid manifests
        if len(parts) >= 2 and parts[0] == "data" and parts[1] == "grids":
            return True, "Protected official FIA starting grid manifest"

        # Audit log protection
        if "deletion_audit.jsonl" in resolved.name or "logs" in parts:
            return True, "Protected deletion audit trail"

        # Historical training datasets protection by default
        if not include_historical:
            # Past seasons (e.g. data/raw/season/<year> where year < 2026)
            if len(parts) >= 4 and parts[0] == "data" and parts[1] == "raw" and parts[2] == "season":
                try:
                    yr = int(parts[3])
                    if yr < 2026:
                        return True, f"Protected historical training dataset ({yr})"
                except ValueError:
                    pass

            # Historical track sessions (data/raw/tracks/<track>/<year>)
            if len(parts) >= 5 and parts[0] == "data" and parts[1] == "raw" and parts[2] == "tracks":
                try:
                    yr = int(parts[4])
                    if yr < 2026:
                        return True, f"Protected historical circuit dataset ({parts[3]} {yr})"
                except ValueError:
                    pass

        return False, "Eligible"

    # =========================================================================
    # ARTIFACT DISCOVERY
    # =========================================================================

    def discover_cache_data(
        self,
        year: Optional[int] = None,
        track: Optional[str] = None
    ) -> List[Path]:
        """
        Discovers temporary and reproducible cache files:
        - data/cache/**
        - fastf1_cache/**, .cache/**, cache/**
        - scratch/**, temporary files
        Optionally filters by year and/or circuit identifier.
        """
        candidates: List[Path] = []
        cache_roots = [
            self.root_dir / "data" / "cache",
            self.root_dir / "fastf1_cache",
            self.root_dir / ".cache",
            self.root_dir / "cache",
            self.root_dir / "scratch",
        ]

        for c_root in cache_roots:
            if c_root.exists():
                for root, _, files in os.walk(c_root):
                    for f in files:
                        p = Path(root) / f
                        if year is not None and str(year) not in p.as_posix():
                            continue
                        if track is not None and track.lower() not in p.as_posix().lower():
                            continue
                        prot, _ = self.is_protected(p)
                        if not prot:
                            candidates.append(p)

        # Temporary files in root or data
        for root_dir in [self.root_dir, self.root_dir / "data"]:
            if root_dir.exists():
                for f in root_dir.glob("*.tmp"):
                    if year is not None and str(year) not in f.name:
                        continue
                    prot, _ = self.is_protected(f)
                    if not prot:
                        candidates.append(f)
                for f in root_dir.glob("*.temp"):
                    if year is not None and str(year) not in f.name:
                        continue
                    prot, _ = self.is_protected(f)
                    if not prot:
                        candidates.append(f)

        return sorted(list(set(candidates)))

    def discover_prediction_outputs(
        self,
        track_key: Optional[str] = None,
        year: Optional[int] = None
    ) -> List[Path]:
        """
        Discovers generated prediction outputs in outputs/:
        - Qualifying & race reports (.md)
        - Monte Carlo simulation tables & CSVs (.csv)
        - Model comparison tables
        """
        candidates: List[Path] = []
        out_root = self.root_dir / "outputs"
        if not out_root.exists():
            return []

        search_dirs = [out_root / track_key.lower().strip()] if track_key else [
            d for d in out_root.iterdir() if d.is_dir()
        ]

        for d in search_dirs:
            if not d.exists():
                continue
            for root, _, files in os.walk(d):
                for f in files:
                    p = Path(root) / f
                    if year is not None and f"{year}" not in f:
                        continue
                    prot, _ = self.is_protected(p)
                    if not prot:
                        candidates.append(p)

        return sorted(list(set(candidates)))

    def discover_collected_datasets(
        self,
        categories: Optional[List[str]] = None,
        include_historical: bool = False,
        year: Optional[int] = None,
    ) -> Dict[str, List[Path]]:
        """
        Discovers collected datasets in data/raw/:
        - Practice: *fp1*, *fp2*, *fp3*
        - Qualifying: *qualifying*, *sprint_qualifying*, *shootout*
        - Race: *race*, *sprint_race*
        - Grids: cached OpenF1 grids (data/raw/grids)
        - Processed: data/processed
        """
        valid_cats = {"practice", "qualifying", "race", "grids", "processed"}
        selected_cats = set(categories) if categories else valid_cats

        classified: Dict[str, List[Path]] = {cat: [] for cat in selected_cats}
        raw_root = self.root_dir / "data" / "raw"
        proc_root = self.root_dir / "data" / "processed"

        # Search data/raw
        if raw_root.exists():
            for root, _, files in os.walk(raw_root):
                for f in files:
                    p = Path(root) / f
                    if year is not None and f"{year}" not in p.as_posix():
                        continue

                    prot, _ = self.is_protected(p, include_historical=include_historical)
                    if prot:
                        continue

                    f_lower = f.lower()
                    if "grids" in p.parts:
                        if "grids" in selected_cats:
                            classified["grids"].append(p)
                    elif any(s in f_lower for s in ["_fp1", "_fp2", "_fp3", "practice"]):
                        if "practice" in selected_cats:
                            classified["practice"].append(p)
                    elif any(s in f_lower for s in ["qualifying", "shootout", "quali"]):
                        if "qualifying" in selected_cats:
                            classified["qualifying"].append(p)
                    elif any(s in f_lower for s in ["race", "_sprint"]):
                        if "race" in selected_cats:
                            classified["race"].append(p)

        # Search data/processed
        if "processed" in selected_cats and proc_root.exists():
            for root, _, files in os.walk(proc_root):
                for f in files:
                    p = Path(root) / f
                    prot, _ = self.is_protected(p, include_historical=include_historical)
                    if not prot:
                        classified["processed"].append(p)

        return {k: sorted(list(set(v))) for k, v in classified.items()}

    def discover_grand_prix_data(
        self,
        track_key: Optional[str] = None,
        year: int = 2026,
        round_num: Optional[int] = None,
        include_historical: bool = False,
        track: Optional[str] = None,
    ) -> Dict[str, List[Path]]:
        """
        Discovers all artifacts belonging strictly to a specific Grand Prix event:
        - Collected session datasets for that round
        - Track-specific collected sessions
        - Event-specific prediction outputs
        - Event-specific cache entries
        """
        clean_track = (track or track_key or "").lower().strip()
        t_info = self.tracks_config.get(clean_track, {})
        resolved_round = round_num or t_info.get(f"round_{year}" if year != 2026 else "round_2026")

        event_files: Dict[str, List[Path]] = {
            "session_data": [],
            "track_data": [],
            "prediction_outputs": [],
            "cached_data": [],
        }

        # 1. Round session data in data/raw/season/{year}
        if resolved_round is not None:
            season_dir = self.root_dir / "data" / "raw" / "season" / str(year)
            if season_dir.exists():
                for f in season_dir.glob(f"{year}_round_{resolved_round}_*.csv"):
                    prot, _ = self.is_protected(f, include_historical=include_historical)
                    if not prot:
                        event_files["session_data"].append(f)

            # Cached OpenF1 grid
            grid_dir = self.root_dir / "data" / "raw" / "grids" / str(year)
            if grid_dir.exists():
                for f in grid_dir.glob(f"{year}_round_{resolved_round}_*.json"):
                    prot, _ = self.is_protected(f, include_historical=include_historical)
                    if not prot:
                        event_files["session_data"].append(f)

        # 2. Track data in data/raw/tracks/{track_key}/{year}
        if clean_track:
            track_dir = self.root_dir / "data" / "raw" / "tracks" / clean_track / str(year)
            if track_dir.exists():
                for root, _, files in os.walk(track_dir):
                    for f in files:
                        p = Path(root) / f
                        prot, _ = self.is_protected(p, include_historical=include_historical)
                        if not prot:
                            event_files["track_data"].append(p)

            # 3. Prediction outputs in outputs/{track_key}
            out_dir = self.root_dir / "outputs" / clean_track
            if out_dir.exists():
                for f in out_dir.iterdir():
                    if f.is_file() and f"{year}" in f.name:
                        prot, _ = self.is_protected(f, include_historical=include_historical)
                        if not prot:
                            event_files["prediction_outputs"].append(f)

            # 4. Cache files in data/cache/{year} matching track patterns
            cache_year = self.root_dir / "data" / "cache" / str(year)
            patterns = list(t_info.get("event_name_patterns", [clean_track]))
            patterns.append(clean_track.replace("_", " "))
            if cache_year.exists():
                for d in cache_year.iterdir():
                    if d.is_dir():
                        d_name_clean = d.name.lower().replace("_", " ")
                        if any(p.lower() in d_name_clean for p in patterns):
                            for root, _, files in os.walk(d):
                                for f in files:
                                    p = Path(root) / f
                                    prot, _ = self.is_protected(p, include_historical=include_historical)
                                    if not prot:
                                        event_files["cached_data"].append(p)
                                p = Path(root) / f
                                prot, _ = self.is_protected(p, include_historical=True)
                                if not prot:
                                    event_files["cached_data"].append(p)

        return {k: sorted(list(set(v))) for k, v in event_files.items()}

    def discover_all_project_generated_data(self, include_historical: bool = False) -> Dict[str, List[Path]]:
        """
        Discovers all project-generated data across caches, datasets, and predictions.
        Strictly preserves historical training data unless include_historical=True.
        """
        return {
            "cached_data": self.discover_cache_data(),
            "prediction_outputs": self.discover_prediction_outputs(),
            "collected_datasets": [
                f for flist in self.discover_collected_datasets(include_historical=include_historical).values() for f in flist
            ],
        }

    # =========================================================================
    # PREVIEW & EXECUTION
    # =========================================================================

    def build_deletion_preview(
        self,
        files: List[Path],
        operation_name: str,
        include_historical: bool = False,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Constructs a structured deletion preview with file counts, estimated storage,
        and required confirmation requirements.
        """
        valid_files: List[Path] = []
        excluded_files: List[Tuple[Path, str]] = []
        total_bytes = 0

        for f in files:
            try:
                prot, reason = self.is_protected(f, include_historical=include_historical)
                if prot:
                    excluded_files.append((f, reason))
                    continue

                res = self.validate_path(f)
                if res.is_file():
                    valid_files.append(res)
                    total_bytes += res.stat().st_size
            except Exception as e:
                excluded_files.append((f, str(e)))

        # Categorize valid files
        categories: Dict[str, int] = {}
        for vf in valid_files:
            rel = vf.relative_to(self.root_dir).as_posix()
            if "cache" in rel:
                cat = "Cache data"
            elif "outputs" in rel:
                cat = "Prediction outputs"
            elif "season" in rel:
                cat = "Season round datasets"
            elif "tracks" in rel:
                cat = "Circuit-specific datasets"
            elif "grids" in rel:
                cat = "Cached grid data"
            else:
                cat = "Other generated data"
            categories[cat] = categories.get(cat, 0) + 1

        req_phrase = None
        if "all" in operation_name.lower():
            req_phrase = "DELETE ALL GENERATED DATA"
        elif include_historical:
            req_phrase = "DELETE HISTORICAL DATA"

        return {
            "operation": operation_name,
            "target_files": valid_files,
            "target_paths": [p.relative_to(self.root_dir).as_posix() for p in valid_files],
            "file_count": len(valid_files),
            "total_bytes": total_bytes,
            "storage_formatted": self._format_size(total_bytes),
            "categories": categories,
            "excluded_count": len(excluded_files),
            "excluded_files": [(p.relative_to(self.root_dir).as_posix() if p.is_relative_to(self.root_dir) else str(p), r) for p, r in excluded_files],
            "historical_included": include_historical,
            "required_confirmation_phrase": req_phrase,
            "metadata": extra_metadata or {},
        }

    def execute_deletion_plan(
        self,
        preview: Dict[str, Any],
        confirmed: bool = False,
        confirmation_phrase: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes approved deletion plan with strict path re-validation.
        Logs operation to persistent deletion_audit.jsonl.
        """
        op_name = preview.get("operation", "Unknown Deletion")
        target_files: List[Path] = preview.get("target_files", [])
        hist_inc = preview.get("historical_included", False)
        req_phrase = preview.get("required_confirmation_phrase")

        if not confirmed:
            self.audit_logger.log_operation(
                operation_type=op_name,
                files_identified=preview.get("target_paths", []),
                files_deleted=[],
                files_excluded=[p for p, _ in preview.get("excluded_files", [])],
                failed_deletions=[],
                storage_reclaimed_bytes=0,
                historical_included=hist_inc,
                confirmed=False,
                status="CANCELLED",
                metadata=preview.get("metadata"),
            )
            return {
                "status": "CANCELLED",
                "message": "Operation cancelled by user.",
                "files_deleted_count": 0,
                "reclaimed_bytes": 0,
            }

        if req_phrase and (confirmation_phrase or "").strip() != req_phrase:
            return {
                "status": "REJECTED",
                "message": f"Confirmation phrase mismatch. Expected '{req_phrase}', got '{confirmation_phrase}'.",
                "files_deleted_count": 0,
                "reclaimed_bytes": 0,
            }

        deleted_paths: List[str] = []
        failed_deletions: List[Dict[str, str]] = []
        reclaimed_bytes = 0

        affected_dirs: Set[Path] = set()

        for f in target_files:
            try:
                # Re-validate immediately before deletion
                prot, reason = self.is_protected(f, include_historical=hist_inc)
                if prot:
                    failed_deletions.append({"path": str(f), "error": f"Protected: {reason}"})
                    continue

                res = self.validate_path(f)
                if res.is_file():
                    sz = res.stat().st_size
                    res.unlink()
                    reclaimed_bytes += sz
                    deleted_paths.append(res.relative_to(self.root_dir).as_posix())
                    affected_dirs.add(res.parent)
            except Exception as e:
                failed_deletions.append({"path": str(f), "error": str(e)})

        # Safely remove empty subdirectories up to root data/cache or outputs
        for d in sorted(list(affected_dirs), key=lambda x: len(x.parts), reverse=True):
            try:
                if d.exists() and d != self.root_dir and not any(d.iterdir()):
                    # Only remove if inside data/cache, outputs, or scratch
                    rel = d.relative_to(self.root_dir).parts
                    if rel[0] in ["outputs", "scratch"] or (len(rel) >= 2 and rel[0] == "data" and rel[1] == "cache"):
                        d.rmdir()
            except Exception:
                pass

        status = "COMPLETED"
        if failed_deletions:
            status = "PARTIAL_FAILURE" if deleted_paths else "FAILED"

        audit_entry = self.audit_logger.log_operation(
            operation_type=op_name,
            files_identified=preview.get("target_paths", []),
            files_deleted=deleted_paths,
            files_excluded=[p for p, _ in preview.get("excluded_files", [])],
            failed_deletions=failed_deletions,
            storage_reclaimed_bytes=reclaimed_bytes,
            historical_included=hist_inc,
            confirmed=True,
            status=status,
            metadata=preview.get("metadata"),
        )

        return {
            "status": status,
            "operation": op_name,
            "files_identified_count": len(target_files),
            "files_deleted_count": len(deleted_paths),
            "failed_count": len(failed_deletions),
            "failed_deletions": failed_deletions,
            "storage_reclaimed_bytes": reclaimed_bytes,
            "storage_reclaimed_formatted": self._format_size(reclaimed_bytes),
            "audit_log_path": str(self.audit_logger.log_path),
            "audit_entry": audit_entry,
        }

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Formats byte count to human-readable size string."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.2f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
