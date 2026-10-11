import asyncio
import json
import unittest

from llm_trace import Tracer


class TracerTests(unittest.TestCase):
    def test_nested_spans_and_usage(self):
        tracer = Tracer()
        with tracer.trace("question") as trace:
            with tracer.span("answer") as parent:
                with tracer.span("generate", kind="model", model="demo", prompt_version="v1") as child:
                    child.set_usage(input_tokens=10, output_tokens=5, estimated_cost_usd=0.002)
        data = json.loads(json.dumps(trace.to_dict(), allow_nan=False))
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(child.parent_id, parent.id)
        self.assertEqual(child.input_tokens, 10)
        self.assertEqual(child.estimated_cost_usd, 0.002)
        self.assertEqual(trace.status, "ok")
        self.assertGreaterEqual(trace.duration_ms, parent.duration_ms)
        self.assertGreaterEqual(parent.duration_ms, child.duration_ms)

    def test_error_is_recorded_and_original_exception_survives(self):
        tracer = Tracer()
        error = ValueError("sensitive exception message")
        with self.assertRaises(ValueError) as caught:
            with tracer.trace("request") as trace:
                with tracer.span("lookup", kind="tool") as span:
                    raise error
        self.assertIs(caught.exception, error)
        self.assertEqual(trace.error_type, "ValueError")
        self.assertEqual(span.status, "error")
        self.assertNotIn("sensitive exception message", json.dumps(trace.to_dict()))
        with tracer.trace("next") as next_trace:
            pass
        self.assertEqual(next_trace.status, "ok")

    def test_caught_tool_failure_keeps_request_successful(self):
        tracer = Tracer()
        with tracer.trace("request") as trace:
            try:
                with tracer.span("lookup", kind="tool"):
                    raise LookupError()
            except LookupError:
                pass
        self.assertEqual(trace.status, "ok")
        self.assertEqual(trace.spans[0].status, "error")

    def test_context_ownership(self):
        tracer = Tracer()
        with self.assertRaises(RuntimeError):
            with tracer.span("orphan"):
                pass
        with tracer.trace("request"):
            with self.assertRaises(RuntimeError):
                with tracer.trace("nested"):
                    pass
            with self.assertRaises(RuntimeError):
                with Tracer().span("wrong tracer"):
                    pass

    def test_invalid_usage_does_not_partially_update(self):
        tracer = Tracer()
        with tracer.trace("request"):
            with tracer.span("generate", kind="model") as span:
                span.set_usage(input_tokens=1, output_tokens=2)
                for bad in (-1, 1.5, True, "3"):
                    with self.subTest(count=bad), self.assertRaises(ValueError):
                        span.set_usage(input_tokens=10, output_tokens=bad)
                for bad in (-1, float("nan"), float("inf"), True, "0.1"):
                    with self.subTest(cost=bad), self.assertRaises(ValueError):
                        span.set_usage(input_tokens=10, output_tokens=20, estimated_cost_usd=bad)
                self.assertEqual(span.input_tokens, 1)
            with self.assertRaises(RuntimeError):
                span.set_usage(input_tokens=1, output_tokens=2)
            with tracer.span("tool", kind="tool") as tool:
                with self.assertRaises(ValueError):
                    tool.set_usage(input_tokens=1, output_tokens=2)

    def test_invalid_metadata(self):
        tracer = Tracer()
        with self.assertRaises(ValueError):
            with tracer.trace(" "):
                pass
        with tracer.trace("request") as trace:
            for args in ({"kind": "unknown"}, {"kind": "tool", "model": "demo"}, {"model": ""}):
                with self.assertRaises(ValueError):
                    with tracer.span("call", **args):
                        pass
        self.assertEqual(trace.spans, [])


class AsyncTracerTests(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_requests_do_not_share_parents(self):
        tracer = Tracer()

        async def request(name):
            with tracer.trace(name) as trace:
                with tracer.span(name) as parent:
                    await asyncio.sleep(0)
                    with tracer.span("child"):
                        await asyncio.sleep(0)
            self.assertEqual(trace.spans[1].parent_id, parent.id)
            return trace

        first, second = await asyncio.gather(request("one"), request("two"))
        self.assertNotEqual(first.id, second.id)
        self.assertTrue(set(s.id for s in first.spans).isdisjoint(s.id for s in second.spans))

    async def test_parallel_siblings_share_only_their_real_parent(self):
        tracer = Tracer()

        async def call():
            with tracer.span("child") as span:
                await asyncio.sleep(0)
            return span

        with tracer.trace("request"):
            with tracer.span("parent") as parent:
                children = await asyncio.gather(call(), call())
        self.assertEqual([s.parent_id for s in children], [parent.id, parent.id])

    async def test_task_cannot_start_span_after_trace_ends(self):
        tracer = Tracer()
        gate = asyncio.Event()

        async def late():
            await gate.wait()
            with tracer.span("too late"):
                pass

        with tracer.trace("request"):
            task = asyncio.create_task(late())
        gate.set()
        with self.assertRaisesRegex(RuntimeError, "trace has ended"):
            await task

    async def test_cancellation_is_recorded(self):
        tracer = Tracer()
        with self.assertRaises(asyncio.CancelledError):
            with tracer.trace("request") as trace:
                with tracer.span("call") as span:
                    raise asyncio.CancelledError()
        self.assertEqual(trace.status, "error")
        self.assertEqual(span.error_type, "CancelledError")
