from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from llm_trace import SQLiteStore, summarize


class DemoTests(unittest.TestCase):
    def test_demo_persists_all_cases_and_exposes_the_slow_call(self):
        with TemporaryDirectory() as directory:
            db = Path(directory) / "folder with spaces" / "demo.sqlite3"
            example = Path(__file__).resolve().parents[1] / "examples" / "demo.py"
            result = subprocess.run(
                [sys.executable, str(example), "--db", str(db)], capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            store = SQLiteStore(db, read_only=True)
            rows = {row["name"]: row for row in store.list_recent()}
            self.assertEqual(set(rows), {"normal", "slow-generation", "recovered-lookup", "failed-lookup"})
            slow = store.get(rows["slow-generation"]["id"])
            # Check the deliberate delay itself; run-to-run comparisons are noisy on CI.
            slow_model = next(s for s in slow["spans"] if s["kind"] == "model")
            self.assertGreaterEqual(slow_model["duration_ms"], 75)
            recovered = summarize(store.get(rows["recovered-lookup"]["id"]))
            self.assertEqual(recovered["status"], "ok")
            self.assertEqual(len(recovered["failed_spans"]), 1)
            failed = summarize(store.get(rows["failed-lookup"]["id"]))
            self.assertEqual(failed["status"], "error")
            self.assertEqual(failed["model_calls"], 0)
            self.assertEqual(failed["error_type"], "LookupError")
