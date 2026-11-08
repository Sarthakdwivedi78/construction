"""Local SQLite persistence for inspection report history."""
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from retrieval import ROOT

DEFAULT_DATABASE = Path(os.environ.get(
    "SAFETY_HISTORY_DB",
    ROOT / "outputs" / "inspection_reports" / "history.sqlite3",
))


def save_inspection(report, database_path=DEFAULT_DATABASE):
    if not isinstance(report, dict) or not isinstance(report.get("inspection_id"), str):
        raise ValueError("Inspection report needs a string inspection_id.")
    saved = dict(report)
    saved.setdefault("created_at", datetime.now(timezone.utc).isoformat(timespec="microseconds"))
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(saved, ensure_ascii=False, separators=(",", ":"))
    with closing(sqlite3.connect(database_path, timeout=10)) as connection:
        with connection:
            connection.execute("PRAGMA busy_timeout = 10000")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS inspections (
                    inspection_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    site_type TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    report_json TEXT NOT NULL
                )
            """)
            connection.execute("""
                INSERT INTO inspections (inspection_id, created_at, site_type, risk_level, report_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(inspection_id) DO NOTHING
            """, (saved["inspection_id"], saved["created_at"],
                  str(saved.get("site_type", "Unspecified")),
                  str(saved.get("overall_risk_level", "Undetermined")), payload))
    return saved


def load_history(limit=100, database_path=DEFAULT_DATABASE):
    if not isinstance(limit, int) or limit < 1 or limit > 1000:
        raise ValueError("History limit must be between 1 and 1000.")
    database_path = Path(database_path)
    if not database_path.is_file():
        return []
    with closing(sqlite3.connect(database_path, timeout=10)) as connection:
        connection.execute("PRAGMA busy_timeout = 10000")
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='inspections'"
        ).fetchone()
        if not table:
            return []
        rows = connection.execute(
            "SELECT report_json FROM inspections ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [json.loads(row[0]) for row in rows]
