"""
Persistent Deletion Audit Logger
================================
Maintains an append-only, immutable audit trail of all data deletion operations
performed within APEXGRID Motorsport Intelligence Platform.

Audit log is stored at: logs/deletion_audit.jsonl
Audit records are permanently protected from deletion.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


class DeletionAuditLogger:
    """Manages immutable, append-only logging of data management operations."""

    def __init__(self, root_dir: Optional[Path] = None, log_file: Optional[Path] = None):
        self.root_dir = (root_dir or Path(__file__).resolve().parents[2]).resolve()
        if log_file:
            self.log_path = log_file.resolve()
        else:
            self.log_path = (self.root_dir / "logs" / "deletion_audit.jsonl").resolve()

        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_operation(
        self,
        operation_type: str,
        files_identified: List[str],
        files_deleted: List[str],
        files_excluded: List[str],
        failed_deletions: List[Dict[str, str]],
        storage_reclaimed_bytes: int,
        historical_included: bool = False,
        confirmed: bool = True,
        status: str = "COMPLETED",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Appends a structured audit record to deletion_audit.jsonl.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation_type": operation_type,
            "status": status,
            "confirmed": confirmed,
            "historical_data_included": historical_included,
            "storage_reclaimed_bytes": storage_reclaimed_bytes,
            "storage_reclaimed_mb": round(storage_reclaimed_bytes / (1024 * 1024), 3),
            "files_identified_count": len(files_identified),
            "files_deleted_count": len(files_deleted),
            "files_excluded_count": len(files_excluded),
            "failed_deletions_count": len(failed_deletions),
            "files_deleted": files_deleted,
            "files_excluded": files_excluded,
            "failed_deletions": failed_deletions,
            "metadata": metadata or {},
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        return entry

    def get_audit_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Reads recent audit history entries."""
        if not self.log_path.exists():
            return []

        records = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            pass

        return records[-limit:]

