from contextlib import redirect_stderr, redirect_stdout
from html.parser import HTMLParser
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from llm_trace import SQLiteStore, Tracer
from llm_trace.cli import main
from llm_trace.report import render_report


class Markup(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.tags = []
        self.feed(html)

    def handle_starttag(self, tag, attributes):
        self.tags.append((tag, dict(attributes)))


class ReportTests(unittest.TestCase):
    def record(self):
        tracer = Tracer()
        with tracer.trace("request") as trace:
            with tracer.span("answer"):
                with tracer.span("generate", kind="model", model="demo", prompt_version="v1") as call:
                    call.set_usage(input_tokens=10, output_tokens=4, estimated_cost_usd=0.002)
                try:
                    with tracer.span("lookup", kind="tool"):
                        raise LookupError()
                except LookupError:
                    pass
        return trace

    def test_report_preserves_hierarchy_usage_and_recovered_failures(self):
        document = render_report(self.record().to_dict())
        self.assertIn("10 in / 4 out", document)
        self.assertIn("Usage on 1/1 model calls", document)
        self.assertIn("$0.002000", document)
        self.assertIn("1 failed span", document)
        self.assertIn("LookupError", document)
        self.assertIn("padding-left:16px", document)
        self.assertIn("inclusive durations", document)

    def test_all_text_fields_are_escaped_and_no_executable_tags_are_created(self):
        trace = self.record().to_dict()
        attack = '</title><script>alert("x")</script><img src="https://evil.test/">'
        trace["name"] = trace["id"] = trace["started_at"] = trace["error_type"] = attack
        for span in trace["spans"]:
            for field in ("name", "model", "prompt_version", "error_type", "kind", "status"):
                span[field] = attack
        document = render_report(trace)
        tags = Markup(document).tags
        self.assertFalse(any(tag in ("script", "img", "iframe", "link") for tag, _ in tags))
        self.assertFalse(any(key in ("src", "href") or key.startswith("on") for _, attrs in tags for key in attrs))
        self.assertIn("&lt;script&gt;", document)
        self.assertIn("Content-Security-Policy", document)

    def test_empty_trace_and_missing_measurements(self):
        tracer = Tracer()
        with tracer.trace("empty") as trace:
            pass
        document = render_report(trace.to_dict())
        self.assertIn("No spans recorded.", document)
        self.assertIn("Not recorded", document)
        self.assertNotIn("width:nan", document)
        with tracer.trace("zero-duration") as trace:
            with tracer.span("call") as call:
                pass
        call.duration_ms = 0
        self.assertIn("width:0.00%", render_report(trace.to_dict()))

    def test_cli_creates_report_and_refuses_to_overwrite_files_or_database(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            trace = self.record()
            db = path / "traces.sqlite3"
            SQLiteStore(db).save(trace)
            report = path / "trace.html"
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(main(["--db", str(db), "report", trace.id[:8], "--output", str(report)]), 0)
                original = report.read_bytes()
                self.assertEqual(main(["--db", str(db), "report", trace.id, "--output", str(report)]), 1)
                self.assertEqual(report.read_bytes(), original)
                self.assertEqual(main(["--db", str(db), "report", trace.id, "--output", str(db)]), 1)
            self.assertEqual(SQLiteStore(db, read_only=True).get(trace.id), trace.to_dict())
