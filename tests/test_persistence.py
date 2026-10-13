import asyncio
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from llm_trace import SQLiteStore, Tracer


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = SQLiteStore(Path(self.directory.name) / "traces.sqlite3")

    def test_completed_trace_is_saved_once_with_all_spans(self):
        sink = Mock(wraps=self.store)
        tracer = Tracer(store=sink)
        with tracer.trace("request") as trace:
            with tracer.span("generate", kind="model"):
                self.assertEqual(self.store.list_recent(), [])
        sink.save.assert_called_once_with(trace)
        self.assertEqual(self.store.get(trace.id), trace.to_dict())
        self.assertEqual(trace.status, "ok")

    def test_failed_request_is_saved_and_original_exception_survives(self):
        tracer = Tracer(store=self.store)
        error = ValueError("private application message")
        with self.assertRaises(ValueError) as caught:
            with tracer.trace("request") as trace:
                with tracer.span("tool", kind="tool"):
                    raise error
        self.assertIs(caught.exception, error)
        stored = self.store.get(trace.id)
        self.assertEqual(stored["status"], "error")
        self.assertEqual(stored["spans"][0]["error_type"], "ValueError")

    def test_save_error_does_not_break_success_or_replace_app_error(self):
        sink = Mock()
        sink.save.side_effect = sqlite3.OperationalError("private database path")
        tracer = Tracer(store=sink)
        with self.assertLogs("llm_trace.tracer", level="WARNING") as logs:
            with tracer.trace("successful") as first:
                pass
            error = LookupError("private application message")
            with self.assertRaises(LookupError) as caught:
                with tracer.trace("failed") as second:
                    raise error
        self.assertIs(caught.exception, error)
        self.assertEqual(first.status, "ok")
        self.assertEqual(second.status, "error")
        self.assertEqual(sink.save.call_count, 2)
        self.assertIn("OperationalError", " ".join(logs.output))
        self.assertNotIn("private", " ".join(logs.output))

    def test_context_is_reset_before_export_and_export_time_is_excluded(self):
        observed = []
        independent = Tracer()

        class Sink:
            def save(self, trace):
                observed.append(trace.duration_ms)
                with independent.trace("export-work"):
                    pass

        tracer = Tracer(store=Sink())
        with patch("llm_trace.tracer.perf_counter", side_effect=[10, 10.02, 10.02, 11]):
            with tracer.trace("request") as trace:
                pass
        self.assertAlmostEqual(trace.duration_ms, 20)
        self.assertEqual(observed, [trace.duration_ms])

    def test_logging_failure_cannot_break_request(self):
        sink = Mock()
        sink.save.side_effect = OSError("private message")
        with patch("llm_trace.tracer._logger.warning", side_effect=RuntimeError("handler failed")):
            with Tracer(store=sink).trace("request") as trace:
                pass
        self.assertEqual(trace.status, "ok")

    def test_default_tracer_keeps_explicit_save_behavior(self):
        with Tracer().trace("manual") as trace:
            pass
        self.assertEqual(self.store.list_recent(), [])
        self.store.save(trace)
        self.assertEqual(self.store.get(trace.id), trace.to_dict())


class AsyncPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_is_saved_without_replacing_it(self):
        with TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / "traces.sqlite3")
            tracer = Tracer(store=store)
            with self.assertRaises(asyncio.CancelledError):
                with tracer.trace("request") as trace:
                    with tracer.span("call"):
                        raise asyncio.CancelledError()
            self.assertEqual(store.get(trace.id)["error_type"], "CancelledError")
            with tracer.trace("next") as next_trace:
                await asyncio.sleep(0)
            self.assertEqual(store.get(next_trace.id)["status"], "ok")
