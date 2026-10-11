"""SQLite persistence. Each operation owns and closes its connection."""

from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3

from .tracer import Trace


class SQLiteStore:
    def __init__(self, path: str | Path, *, read_only: bool = False):
        self.path = Path(path).expanduser().resolve()
        self.read_only = read_only
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError(f"unsupported database version: {version}")
            if read_only:
                if version != 1:
                    raise ValueError("this is not an llm-trace database")
            else:
                with connection:
                    connection.execute("""
                        CREATE TABLE IF NOT EXISTS traces (
                            id TEXT PRIMARY KEY,
                            name TEXT NOT NULL,
                            started_at TEXT NOT NULL,
                            status TEXT NOT NULL,
                            duration_ms REAL NOT NULL,
                            payload TEXT NOT NULL
                        )
                    """)
                    connection.execute("""
                        CREATE INDEX IF NOT EXISTS traces_started_at
                        ON traces(started_at DESC, id DESC)
                    """)
                    connection.execute("PRAGMA user_version = 1")

    def _connect(self) -> sqlite3.Connection:
        if self.read_only:
            connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        else:
            connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def save(self, trace: Trace) -> None:
        """Insert a finished trace. Saving an existing ID is an error."""
        if self.read_only:
            raise ValueError("store is read-only")
        if trace.status not in ("ok", "error") or any(
            span.status not in ("ok", "error") for span in trace.spans
        ):
            raise ValueError("finish the trace and all its spans before saving")
        payload = json.dumps(trace.to_dict(), allow_nan=False)
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO traces VALUES (?, ?, ?, ?, ?, ?)",
                (trace.id, trace.name, trace.started_at, trace.status, trace.duration_ms, payload),
            )

    def list_recent(self, *, limit: int = 20, status: str | None = None) -> list[dict]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        if status not in (None, "ok", "error"):
            raise ValueError("status must be ok or error")
        where = " WHERE status = ?" if status else ""
        parameters = (status, limit) if status else (limit,)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT id, name, started_at, status, duration_ms FROM traces"
                + where + " ORDER BY started_at DESC, id DESC LIMIT ?", parameters,
            ).fetchall()
        return [dict(row) for row in rows]

    def get(self, trace_id: str) -> dict:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT payload FROM traces WHERE id = ?", (trace_id,)).fetchone()
        if row is None:
            raise KeyError(f"trace not found: {trace_id}")
        data = json.loads(row["payload"])
        if data.get("schema_version") != 1:
            raise ValueError("unsupported trace schema version")
        return data

    def resolve_id(self, prefix: str) -> str:
        """Resolve a hexadecimal prefix; never guess when it matches twice."""
        if not re.fullmatch(r"[0-9a-fA-F]{4,32}", prefix):
            raise ValueError("trace ID must be 4 to 32 hexadecimal characters")
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT id FROM traces WHERE id LIKE ? LIMIT 2", (prefix.lower() + "%",),
            ).fetchall()
        if not rows:
            raise KeyError(f"trace not found: {prefix}")
        if len(rows) != 1:
            raise ValueError("trace ID prefix matches multiple traces; use a longer prefix")
        return rows[0]["id"]
