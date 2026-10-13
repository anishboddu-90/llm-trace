import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from llm_trace import SQLiteStore, Tracer


class CLITests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "traces.sqlite3"
        store = SQLiteStore(self.path)
        tracer = Tracer()
        with tracer.trace("question") as self.trace:
            with tracer.span("answer"):
                with tracer.span("generate", kind="model", model="demo", prompt_version="v1") as call:
                    call.set_usage(input_tokens=10, output_tokens=2, estimated_cost_usd=0.0001)
        store.save(self.trace)

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "llm_trace", "--db", str(self.path), *args],
            capture_output=True, text=True, check=False,
        )

    def test_list_and_show(self):
        listing = self.run_cli("list")
        self.assertEqual(listing.returncode, 0, listing.stderr)
        self.assertIn(self.trace.id, listing.stdout)
        shown = self.run_cli("show", self.trace.id[:8])
        self.assertEqual(shown.returncode, 0, shown.stderr)
        self.assertIn("    model generate", shown.stdout)
        self.assertIn("tokens=10 in/2 out", shown.stdout)
        self.assertIn("prompt=v1", shown.stdout)
        self.assertIn("cost=$0.000100", shown.stdout)

    def test_export_is_plain_json(self):
        result = self.run_cli("export", self.trace.id)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), self.trace.to_dict())
        self.assertEqual(result.stderr, "")

    def test_summary_and_json(self):
        result = self.run_cli("summary", self.trace.id[:8], "--slowest", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Recorded tokens: 10 in / 2 out", result.stdout)
        self.assertIn("Usage coverage: 1/1", result.stdout)
        self.assertIn("Failed spans: 0", result.stdout)
        result = self.run_cli("summary", self.trace.id, "--json", "--slowest", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["recorded_input_tokens"], 10)
        self.assertEqual(len(data["slowest_spans"]), 1)
        invalid = self.run_cli("summary", self.trace.id, "--slowest", "0")
        self.assertEqual(invalid.returncode, 1)

    def test_invalid_id_and_limit_return_errors(self):
        for args in (("show", "zzzz"), ("show", "00000000"), ("list", "--limit", "0")):
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 1)
                self.assertIn("llm-trace:", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_missing_database_returns_error_without_creating_it(self):
        self.path = self.path.with_name("missing.sqlite3")
        result = self.run_cli("list")
        self.assertEqual(result.returncode, 1)
        self.assertFalse(self.path.exists())

    def test_comparison_reports_changed_prompt_and_handles_partial_data(self):
        tracer = Tracer()
        with tracer.trace("candidate") as candidate:
            with tracer.span("answer"):
                with tracer.span("generate", kind="model", model="new-model", prompt_version="v2"):
                    pass
        SQLiteStore(self.path).save(candidate)
        result = self.run_cli("compare", self.trace.id[:8], candidate.id[:8])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Prompt versions: v1 -> v2", result.stdout)
        self.assertIn("Models: demo -> new-model", result.stdout)
        self.assertIn("Input tokens: not comparable", result.stdout)
        result = self.run_cli("compare", self.trace.id, candidate.id, "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["candidate"]["trace_id"], candidate.id)
        self.assertIsNone(data["input_tokens"]["delta"])
        self.assertEqual(len(data["calls"]), 2)
        missing = self.run_cli("compare", self.trace.id, "00000000")
        self.assertEqual(missing.returncode, 1)

    def test_terminal_control_characters_in_names_are_escaped(self):
        tracer = Tracer()
        with tracer.trace("unsafe\n\x1b[31m") as trace:
            with tracer.span("tool\rname", kind="tool"):
                pass
        SQLiteStore(self.path).save(trace)
        result = self.run_cli("show", trace.id)
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("\x1b", result.stdout)
        self.assertIn("unsafe\\n\\u001b[31m", result.stdout)
        self.assertIn("tool\\rname", result.stdout)
