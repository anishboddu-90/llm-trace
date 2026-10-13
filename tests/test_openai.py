import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

from llm_trace import Tracer
from llm_trace.integrations.openai import AsyncOpenAIResponses, OpenAIResponses


def response(**updates):
    data = dict(model="resolved-model", status="completed", usage=SimpleNamespace(input_tokens=10, output_tokens=4),
                output_text="private-output", instructions="private-instructions", metadata={"secret": "private-metadata"})
    data.update(updates)
    return SimpleNamespace(**data)


class OpenAIAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tracer = Tracer()
        self.result = response()
        self.client = SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=self.result)))
        self.adapter = OpenAIResponses(self.tracer, self.client)

    def test_response_identity_arguments_usage_and_no_content_capture(self):
        arguments = {"model": "alias", "input": "private-input", "metadata": {"key": "private-key"}, "max_output_tokens": 20}
        with self.tracer.trace("request") as trace:
            with self.tracer.span("answer") as parent:
                result = self.adapter.create(prompt_version="v2", **arguments)
        self.assertIs(result, self.result)
        self.client.responses.create.assert_called_once_with(**arguments)
        span = trace.spans[1]
        self.assertEqual(span.parent_id, parent.id)
        self.assertEqual((span.model, span.prompt_version, span.input_tokens, span.output_tokens), ("resolved-model", "v2", 10, 4))
        self.assertIsNone(span.estimated_cost_usd)
        stored = json.dumps(trace.to_dict())
        for secret in ("private-input", "private-output", "private-instructions", "private-metadata", "private-key"):
            self.assertNotIn(secret, stored)

    def test_provider_exception_is_not_replaced(self):
        error = TimeoutError("private provider message")
        self.client.responses.create.side_effect = error
        with self.assertRaises(TimeoutError) as caught:
            with self.tracer.trace("request") as trace:
                self.adapter.create(model="alias", input="private-input")
        self.assertIs(caught.exception, error)
        self.assertEqual(trace.spans[0].error_type, "TimeoutError")
        self.assertEqual(trace.status, "error")

    def test_missing_and_invalid_usage_stays_unknown(self):
        for usage in (None, SimpleNamespace(input_tokens=1), SimpleNamespace(input_tokens=True, output_tokens=2),
                      SimpleNamespace(input_tokens=-1, output_tokens=2)):
            with self.subTest(usage=usage):
                self.client.responses.create.return_value = response(usage=usage)
                with self.tracer.trace("request") as trace:
                    self.adapter.create(model="alias")
                self.assertIsNone(trace.spans[0].input_tokens)
                self.assertEqual(trace.spans[0].status, "ok")

    def test_returned_failures_are_visible_without_changing_response(self):
        for status, expected in (("failed", "ResponseFailed"), ("incomplete", "ResponseIncomplete")):
            with self.subTest(status=status):
                result = response(status=status)
                self.client.responses.create.return_value = result
                with self.tracer.trace("request") as trace:
                    self.assertIs(self.adapter.create(model="alias"), result)
                self.assertEqual(trace.spans[0].status, "error")
                self.assertEqual(trace.spans[0].error_type, expected)
                self.assertEqual(trace.status, "ok")  # Application can inspect and recover from the response.

    def test_unsupported_modes_and_missing_trace_do_not_call_provider(self):
        for args in ({"stream": True}, {"background": True}):
            with self.assertRaises(ValueError):
                self.adapter.create(model="alias", **args)
        with self.assertRaises(RuntimeError):
            self.adapter.create(model="alias")
        self.client.responses.create.assert_not_called()

    def test_manual_error_marking_enforces_span_lifecycle(self):
        with self.tracer.trace("request") as trace:
            with self.tracer.span("tool", kind="tool") as span:
                with self.assertRaises(ValueError):
                    span.record_error(" ")
                span.record_error("RejectedResult")
        self.assertEqual(span.status, "error")
        self.assertEqual(trace.status, "ok")
        with self.assertRaises(RuntimeError):
            span.record_error("TooLate")


class AsyncOpenAIAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_request_returns_original_response(self):
        tracer = Tracer()
        result = response()
        client = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(return_value=result)))
        adapter = AsyncOpenAIResponses(tracer, client)
        with tracer.trace("request") as trace:
            self.assertIs(await adapter.create(model="alias", input="private-input", prompt_version="v1"), result)
        client.responses.create.assert_awaited_once_with(model="alias", input="private-input")
        self.assertEqual(trace.spans[0].output_tokens, 4)

    async def test_cancellation_and_context_cleanup(self):
        tracer = Tracer()
        client = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(side_effect=asyncio.CancelledError)))
        with self.assertRaises(asyncio.CancelledError):
            with tracer.trace("request") as trace:
                await AsyncOpenAIResponses(tracer, client).create(model="alias")
        self.assertEqual(trace.spans[0].error_type, "CancelledError")
        with tracer.trace("next"):
            pass
