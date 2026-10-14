import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from llm_trace.benchmark import _stats, run_benchmark
from llm_trace.storage import SQLiteStore


class BenchmarkTests(unittest.TestCase):
    def test_statistics_keep_raw_samples_and_negative_deltas(self):
        samples = list(range(-10, 10))
        data = _stats(samples)
        self.assertEqual(data["median_ms"], -0.5)
        self.assertEqual(data["mean_ms"], -0.5)
        self.assertEqual(data["p95_ms"], 8)
        self.assertEqual(data["min_ms"], -10)
        self.assertEqual(data["max_ms"], 9)
        self.assertEqual(data["samples_ms"], samples)

    def test_runs_equal_work_verifies_writes_and_excludes_warmup(self):
        original_save = SQLiteStore.save
        paths = []

        def checked_save(store, trace):
            original_save(store, trace)
            stored = store.get(trace.id)
            self.assertEqual(stored["status"], "ok")
            self.assertEqual(len(stored["spans"]), 3)
            self.assertTrue(all(span["status"] == "ok" for span in stored["spans"]))
            paths.append(store.path)

        with patch.object(SQLiteStore, "save", checked_save):
            result = run_benchmark(rounds=4, warmup=2, spans=3, work=0)
        self.assertEqual(result["checksum"], 6)
        self.assertEqual(result["saved_traces"], 6)
        self.assertEqual(len(paths), 6)
        self.assertFalse(paths[0].exists())
        self.assertGreater(result["database_bytes"], 0)
        for case in result["cases"].values():
            self.assertEqual(len(case["samples_ms"]), 4)
        self.assertEqual(len(result["save"]["samples_ms"]), 4)
        for name, data in result["paired_overhead"].items():
            expected = [current - baseline for current, baseline in zip(
                result["cases"][name]["samples_ms"], result["cases"]["baseline"]["samples_ms"])]
            self.assertEqual(data["samples_ms"], expected)

    def test_swallowed_save_failure_invalidates_measurements_and_cleans_up(self):
        paths = []

        def failed_save(store, trace):
            paths.append(store.path)
            raise sqlite3.OperationalError("simulated unavailable disk")

        with patch.object(SQLiteStore, "save", failed_save):
            with self.assertLogs("llm_trace.tracer", level="WARNING"):
                with self.assertRaisesRegex(ValueError, "not saved"):
                    run_benchmark(rounds=1, warmup=0, spans=1, work=0)
        self.assertFalse(paths[0].exists())

    def test_invalid_or_excessive_configuration_fails_before_storage_setup(self):
        invalid = ({"rounds": 0}, {"rounds": True}, {"warmup": -1}, {"spans": 101},
                   {"work": 1.5}, {"rounds": 10000, "spans": 100},
                   {"rounds": 1000, "spans": 10, "work": 10000})
        with patch("llm_trace.benchmark.SQLiteStore") as store:
            for config in invalid:
                with self.subTest(config=config), self.assertRaises(ValueError):
                    run_benchmark(**config)
            store.assert_not_called()

    def test_installed_cli_leaves_selected_database_untouched(self):
        with TemporaryDirectory() as directory:
            selected = Path(directory) / "existing.sqlite3"
            selected.write_bytes(b"caller-owned file")
            result = subprocess.run(
                [sys.executable, "-m", "llm_trace", "--db", str(selected), "benchmark",
                 "--rounds", "3", "--warmup", "0", "--spans", "1", "--work", "0", "--json"],
                cwd=directory, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertEqual(data["saved_traces"], 3)
            self.assertEqual(data["checksum"], 1)
            self.assertEqual(selected.read_bytes(), b"caller-owned file")
            self.assertEqual(list(Path(directory).iterdir()), [selected])

    def test_cli_invalid_configuration_is_a_readable_error(self):
        result = subprocess.run(
            [sys.executable, "-m", "llm_trace", "benchmark", "--rounds", "0"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("rounds must be", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
