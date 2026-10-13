import json
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from llm_trace import SQLiteStore, Span, Tracer


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "traces.sqlite3"
        self.store = SQLiteStore(self.path)
        self.tracer = Tracer()

    def record(self, name="request", error=False):
        try:
            with self.tracer.trace(name) as trace:
                with self.tracer.span("generate", kind="model", model="demo") as span:
                    span.set_usage(input_tokens=10, output_tokens=5)
                    if error:
                        raise ValueError()
        except ValueError:
            pass
        self.store.save(trace)
        return trace

    def test_roundtrip_from_another_store(self):
        trace = self.record()
        reader = SQLiteStore(self.path, read_only=True)
        self.assertEqual(reader.get(trace.id), trace.to_dict())
        self.assertEqual(reader.resolve_id(trace.id[:8].upper()), trace.id)
        self.assertEqual(reader.list_recent()[0]["id"], trace.id)
        with self.assertRaisesRegex(ValueError, "read-only"):
            reader.save(trace)

    def test_newest_first_and_status_filter(self):
        first = self.record()
        second = self.record(error=True)
        self.assertEqual([row["id"] for row in self.store.list_recent()], [second.id, first.id])
        self.assertEqual(len(self.store.list_recent(limit=1)), 1)
        self.assertEqual(self.store.list_recent(status="error")[0]["id"], second.id)
        self.assertEqual(self.store.list_recent(status="ok")[0]["id"], first.id)

    def test_sql_metacharacters_are_stored_as_text(self):
        name = "Robert'); DROP TABLE traces; --"
        trace = self.record(name=name)
        self.assertEqual(self.store.get(trace.id)["name"], name)
        self.assertEqual(len(self.store.list_recent()), 1)

    def test_duplicate_id_does_not_overwrite(self):
        trace = self.record()
        trace.name = "changed"
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.save(trace)
        self.assertEqual(self.store.get(trace.id)["name"], "request")

    def test_unfinished_traces_are_rejected(self):
        with self.tracer.trace("request") as trace:
            with self.assertRaisesRegex(ValueError, "finish the trace"):
                self.store.save(trace)
        trace.spans.append(Span(
            id="a" * 32, parent_id=None, name="pending", kind="step", started_at=trace.started_at,
        ))
        with self.assertRaises(ValueError):
            self.store.save(trace)
        self.assertEqual(self.store.list_recent(), [])

    def test_missing_or_ambiguous_ids(self):
        first = self.record()
        with self.assertRaises(KeyError):
            self.store.get("missing")
        for invalid in ("abc", "%", "not-an-id", "a" * 33):
            with self.assertRaises(ValueError):
                self.store.resolve_id(invalid)
        with self.tracer.trace("second") as second:
            pass
        second.id = first.id[:8] + ("0" if first.id[8] != "0" else "1") + first.id[9:]
        self.store.save(second)
        with self.assertRaisesRegex(ValueError, "multiple traces"):
            self.store.resolve_id(first.id[:8])
        self.assertEqual(self.store.resolve_id(first.id), first.id)

    def test_invalid_listing_arguments(self):
        for invalid in (0, -1, 1001, True):
            with self.assertRaises(ValueError):
                self.store.list_recent(limit=invalid)
        with self.assertRaises(ValueError):
            self.store.list_recent(status="running")

    def test_concurrent_requests_persist_without_losing_traces(self):
        # A store can be shared: it creates a separate connection per operation.
        with ThreadPoolExecutor(max_workers=4) as workers:
            traces = list(workers.map(self.record, [f"request-{n}" for n in range(12)]))
        rows = self.store.list_recent()
        self.assertEqual({row["id"] for row in rows}, {trace.id for trace in traces})
        for trace in traces:
            self.assertEqual(self.store.get(trace.id), trace.to_dict())

    def test_missing_read_only_database_does_not_create_files(self):
        path = Path(self.directory.name) / "missing" / "db.sqlite3"
        with self.assertRaises(sqlite3.OperationalError):
            SQLiteStore(path, read_only=True)
        self.assertFalse(path.parent.exists())

    def test_newer_database_version_is_not_modified(self):
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("PRAGMA user_version = 99")
        for read_only in (False, True):
            with self.assertRaisesRegex(ValueError, "unsupported database version"):
                SQLiteStore(self.path, read_only=read_only)
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 99)

    def test_unknown_trace_schema_is_rejected(self):
        trace = self.record()
        data = trace.to_dict()
        data["schema_version"] = 99
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("UPDATE traces SET payload = ? WHERE id = ?", (json.dumps(data), trace.id))
        with self.assertRaisesRegex(ValueError, "unsupported trace schema"):
            self.store.get(trace.id)

    def test_version_number_alone_does_not_claim_another_apps_database(self):
        path = Path(self.directory.name) / "other.sqlite3"
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("CREATE TABLE settings (value TEXT)")
            connection.execute("PRAGMA user_version = 1")
        for read_only in (False, True):
            with self.assertRaisesRegex(ValueError, "not an llm-trace database"):
                SQLiteStore(path, read_only=read_only)
        with closing(sqlite3.connect(path)) as connection:
            self.assertEqual(connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall(), [("settings",)])
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)

    def test_concurrent_store_initialization_uses_one_complete_schema(self):
        path = Path(self.directory.name) / "new.sqlite3"
        with ThreadPoolExecutor(max_workers=4) as workers:
            stores = list(workers.map(SQLiteStore, [path] * 8))
        self.assertTrue(all(store.list_recent() == [] for store in stores))
        with closing(sqlite3.connect(path)) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)
