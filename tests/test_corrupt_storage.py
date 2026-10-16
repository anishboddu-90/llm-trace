from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from llm_trace import SQLiteStore, Tracer


class CorruptStorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "traces.sqlite3"
        self.store = SQLiteStore(self.path)
        tracer = Tracer()
        with tracer.trace("request") as self.trace:
            with tracer.span("retrieve", kind="tool"):
                with tracer.span("generate", kind="model", model="example") as call:
                    call.set_usage(input_tokens=10, output_tokens=5)
        self.store.save(self.trace)

    def replace_payload(self, data):
        self.replace_raw(json.dumps(data))

    def replace_raw(self, payload):
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("UPDATE traces SET payload = ? WHERE id = ?", (payload, self.trace.id))

    def test_non_objects_and_invalid_json_are_readable_errors(self):
        for payload in ("[]", "null", "42", '"private value"', "{private value", b"\xff", "[" * 2000):
            with self.subTest(payload_type=type(payload).__name__):
                self.replace_raw(payload)
                with self.assertRaisesRegex(ValueError, "invalid trace") as raised:
                    self.store.get(self.trace.id)
                self.assertNotIn("private value", str(raised.exception))

    def test_missing_required_fields_and_wrong_types_are_rejected(self):
        original = self.trace.to_dict()
        variants = []
        for key in ("name", "duration_ms", "spans", "error_type"):
            data = deepcopy(original)
            del data[key]
            variants.append(data)
        for key, value in (("name", []), ("status", "running"), ("spans", {}), ("error_type", 42)):
            variants.append({**original, key: value})
        for key, value in (("id", []), ("parent_id", []), ("name", None), ("model", {}),
                           ("status", "running"), ("kind", "other"), ("input_tokens", True),
                           ("output_tokens", -1), ("cached_input_tokens", 11),
                           ("estimated_cost_usd", True), ("token_pricing", [])):
            data = deepcopy(original)
            data["spans"][1][key] = value
            variants.append(data)
        data = deepcopy(original)
        del data["spans"][0]["input_tokens"]
        variants.append(data)
        variants.append({**original, "spans": [None]})
        for index, data in enumerate(variants):
            with self.subTest(index=index):
                self.replace_payload(data)
                with self.assertRaisesRegex(ValueError, "invalid trace"):
                    self.store.get(self.trace.id)

    def test_negative_nonfinite_and_unrepresentable_numbers_are_rejected(self):
        for value in (-1, float("nan"), float("inf"), 10 ** 400, "0.5", True):
            for field in ("duration_ms", "estimated_cost_usd"):
                with self.subTest(value_type=type(value).__name__, field=field):
                    data = self.trace.to_dict()
                    data["spans"][1][field] = value
                    self.replace_payload(data)
                    with self.assertRaisesRegex(ValueError, "finite, non-negative"):
                        self.store.get(self.trace.id)

    def test_duplicate_missing_and_cyclic_parents_are_rejected(self):
        original = self.trace.to_dict()
        duplicate = deepcopy(original)
        duplicate["spans"][1]["id"] = duplicate["spans"][0]["id"]
        missing = deepcopy(original)
        missing["spans"][0]["parent_id"] = "0" * 32
        cyclic = deepcopy(original)
        cyclic["spans"][0]["parent_id"] = cyclic["spans"][1]["id"]
        for data in (duplicate, missing, cyclic):
            self.replace_payload(data)
            with self.assertRaisesRegex(ValueError, "invalid trace"):
                self.store.get(self.trace.id)

    def test_deep_parent_chain_does_not_use_python_recursion(self):
        data = self.trace.to_dict()
        template = data["spans"][0]
        data["spans"] = [{**template, "id": f"{index + 1:032x}",
                          "parent_id": f"{index:032x}" if index else None}
                         for index in range(1200)]
        data["spans"].reverse()
        self.replace_payload(data)
        self.assertEqual(self.store.get(self.trace.id), data)

    def test_older_optional_fields_and_unknown_additions_remain_readable(self):
        data = self.trace.to_dict()
        for span in data["spans"]:
            del span["cached_input_tokens"]
            del span["token_pricing"]
        data["future_field"] = {"version": 2}
        self.replace_payload(data)
        self.assertEqual(self.store.get(self.trace.id), data)

    def test_invalid_pricing_snapshot_is_rejected(self):
        for pricing in ({}, {"input_per_million": "NaN", "output_per_million": "1"},
                        {"input_per_million": "1", "output_per_million": "-2"}):
            data = self.trace.to_dict()
            data["spans"][1]["token_pricing"] = pricing
            self.replace_payload(data)
            with self.assertRaisesRegex(ValueError, "token_pricing"):
                self.store.get(self.trace.id)

    def test_index_and_payload_disagreement_is_rejected(self):
        data = self.trace.to_dict()
        data["id"] = "0" * 32
        self.replace_payload(data)
        with self.assertRaisesRegex(ValueError, "metadata does not match"):
            self.store.get(self.trace.id)

    def test_invalid_unicode_is_rejected_before_report_output_is_opened(self):
        data = self.trace.to_dict()
        data["spans"][0]["name"] = "\ud800"
        self.replace_payload(data)
        report = Path(self.directory.name) / "invalid.html"
        result = subprocess.run(
            [sys.executable, "-m", "llm_trace", "--db", str(self.path), "report",
             self.trace.id, "--output", str(report)], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("valid Unicode", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(report.exists())

    def test_mutated_trace_is_rejected_before_writing(self):
        self.trace.id = "1" * 32
        self.trace.spans[1].duration_ms = "private value"
        with self.assertRaisesRegex(ValueError, "duration_ms"):
            self.store.save(self.trace)
        self.assertEqual(len(self.store.list_recent()), 1)

    def test_inspection_commands_do_not_crash_leak_contents_or_write_reports(self):
        report = Path(self.directory.name) / "report.html"
        commands = [("show", self.trace.id), ("export", self.trace.id),
                    ("summary", self.trace.id, "--json"),
                    ("report", self.trace.id, "--output", str(report)),
                    ("compare", self.trace.id, self.trace.id, "--json")]
        data = self.trace.to_dict()
        data["spans"][1]["duration_ms"] = "private value"
        for payload in ("[]", "{private value", json.dumps(data)):
            self.replace_raw(payload)
            before = self.path.read_bytes()
            for command in commands:
                with self.subTest(command=command[0], payload=payload[:10]):
                    result = subprocess.run(
                        [sys.executable, "-m", "llm_trace", "--db", str(self.path), *command],
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 1)
                    self.assertEqual(result.stdout, "")
                    self.assertIn("llm-trace: invalid trace", result.stderr)
                    self.assertNotIn("Traceback", result.stderr)
                    self.assertNotIn("private value", result.stderr)
                    self.assertFalse(report.exists())
                    self.assertEqual(self.path.read_bytes(), before)

    def test_list_validates_indexed_fields_before_printing(self):
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("UPDATE traces SET duration_ms = 'private value'")
        result = subprocess.run(
            [sys.executable, "-m", "llm_trace", "--db", str(self.path), "list"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("duration_ms", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn("private value", result.stderr)
