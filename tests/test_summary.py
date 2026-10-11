import unittest

from llm_trace import Tracer
from llm_trace.summary import summarize


class SummaryTests(unittest.TestCase):
    def test_known_totals_keep_missing_measurements_visible(self):
        tracer = Tracer()
        with tracer.trace("request") as trace:
            with tracer.span("first", kind="model") as first:
                first.set_usage(input_tokens=10, output_tokens=5, estimated_cost_usd=0.002)
            with tracer.span("second", kind="model") as second:
                second.set_usage(input_tokens=20, output_tokens=3)
            with tracer.span("third", kind="model"):
                pass
        result = summarize(trace.to_dict())
        self.assertEqual(result["model_calls"], 3)
        self.assertEqual(result["recorded_input_tokens"], 30)
        self.assertEqual(result["recorded_output_tokens"], 8)
        self.assertEqual(result["model_calls_missing_usage"], 1)
        self.assertEqual(result["recorded_estimated_cost_usd"], 0.002)
        self.assertEqual(result["model_calls_missing_cost"], 2)

    def test_unknown_usage_and_measured_zero_are_distinct(self):
        tracer = Tracer()
        with tracer.trace("request") as trace:
            with tracer.span("call", kind="model") as call:
                pass
        result = summarize(trace.to_dict())
        self.assertIsNone(result["recorded_input_tokens"])
        self.assertIsNone(result["recorded_estimated_cost_usd"])
        with tracer.trace("zero-usage") as zero_trace:
            with tracer.span("call", kind="model") as zero_call:
                zero_call.set_usage(input_tokens=0, output_tokens=0, estimated_cost_usd=0.0)
        result = summarize(zero_trace.to_dict())
        self.assertEqual(result["recorded_input_tokens"], 0)
        self.assertEqual(result["recorded_estimated_cost_usd"], 0.0)

    def test_recovered_failure_is_in_successful_request_summary(self):
        tracer = Tracer()
        with tracer.trace("request") as trace:
            try:
                with tracer.span("lookup", kind="tool") as failed:
                    raise LookupError()
            except LookupError:
                pass
        result = summarize(trace.to_dict())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["failed_spans"], [{
            "id": failed.id, "name": "lookup", "kind": "tool", "error_type": "LookupError",
        }])

    def test_slowest_calls_rank_without_summing_nested_durations(self):
        tracer = Tracer()
        with tracer.trace("request") as trace:
            with tracer.span("parent") as parent:
                with tracer.span("child") as child:
                    pass
        # Use fixed values so this test doesn't depend on scheduler timing.
        trace.duration_ms, parent.duration_ms, child.duration_ms = 110, 100, 80
        result = summarize(trace.to_dict(), slowest=1)
        self.assertEqual(result["duration_ms"], 110)
        self.assertEqual(len(result["slowest_spans"]), 1)
        self.assertEqual(result["slowest_spans"][0]["id"], parent.id)

    def test_empty_trace_and_invalid_limit(self):
        tracer = Tracer()
        with tracer.trace("request") as trace:
            pass
        result = summarize(trace.to_dict())
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(result["failed_spans"], [])
        self.assertEqual(result["slowest_spans"], [])
        for invalid in (0, -1, 101, True):
            with self.assertRaises(ValueError):
                summarize(trace.to_dict(), slowest=invalid)
