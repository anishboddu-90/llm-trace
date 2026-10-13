import json
from contextlib import closing
from pathlib import Path
import subprocess
import sqlite3
import sys
from tempfile import TemporaryDirectory
import unittest

from llm_trace import SQLiteStore
from llm_trace.demo import run_demo


class PackagedDemoTests(unittest.TestCase):
    def test_demo_runs_outside_checkout_and_traces_can_be_inspected(self):
        with TemporaryDirectory() as directory:
            db = Path(directory) / "demo.sqlite3"

            def cli(*args):
                return subprocess.run(
                    [sys.executable, "-m", "llm_trace", "--db", str(db), *args],
                    cwd=directory, text=True, capture_output=True,
                )

            result = cli("demo")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("No API calls.", result.stdout)
            ids = {line.split(": ", 1)[0]: line.split(": ", 1)[1]
                   for line in result.stdout.splitlines() if not line.startswith("Offline demo:")}
            self.assertEqual(set(ids), {"normal", "slow-generation", "recovered-lookup", "failed-lookup"})
            summary = cli("summary", ids["recovered-lookup"], "--json")
            self.assertEqual(summary.returncode, 0, summary.stderr)
            data = json.loads(summary.stdout)
            self.assertEqual(data["status"], "ok")
            self.assertEqual(len(data["failed_spans"]), 1)
            comparison = cli("compare", ids["normal"], ids["slow-generation"], "--json")
            self.assertEqual(comparison.returncode, 0, comparison.stderr)
            self.assertEqual(json.loads(comparison.stdout)["input_tokens"]["delta"], 0)
            report = Path(directory) / "trace.html"
            rendered = cli("report", ids["recovered-lookup"], "--output", str(report))
            self.assertEqual(rendered.returncode, 0, rendered.stderr)
            self.assertIn("Cached input: 80", report.read_text())

    def test_demo_appends_runs_and_records_automatic_cost_estimates(self):
        with TemporaryDirectory() as directory:
            db = Path(directory) / "demo.sqlite3"
            first, second = run_demo(db), run_demo(db)
            store = SQLiteStore(db, read_only=True)
            self.assertEqual(len(store.list_recent()), 8)
            self.assertTrue(set(first.values()).isdisjoint(second.values()))
            for name, expected in (("normal", 0.0002), ("recovered-lookup", 0.00014)):
                trace = store.get(first[name])
                model = next(span for span in trace["spans"] if span["kind"] == "model")
                self.assertEqual(model["estimated_cost_usd"], expected)
                self.assertEqual(model["token_pricing"]["cached_input_per_million"], "0.25")
                self.assertNotIn("Synthetic input.", json.dumps(trace))
                self.assertNotIn("Synthetic demo response.", json.dumps(trace))

    def test_bad_database_path_returns_cli_error_without_traceback(self):
        with TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-m", "llm_trace", "--db", directory, "demo"],
                cwd=directory, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertNotIn("Traceback", result.stderr)

    def test_demo_does_not_modify_an_unrelated_sqlite_database(self):
        with TemporaryDirectory() as directory:
            db = Path(directory) / "notes.sqlite3"
            with closing(sqlite3.connect(db)) as connection, connection:
                connection.execute("CREATE TABLE notes (body TEXT)")
                connection.execute("INSERT INTO notes VALUES ('keep this')")
            result = subprocess.run(
                [sys.executable, "-m", "llm_trace", "--db", str(db), "demo"],
                cwd=directory, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("unrelated schema", result.stderr)
            with closing(sqlite3.connect(db)) as connection:
                self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 0)
                self.assertEqual(connection.execute("SELECT body FROM notes").fetchone()[0], "keep this")
                self.assertEqual(connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall(), [("notes",)])
