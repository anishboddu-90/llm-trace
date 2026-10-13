"""Optional SDK contract checks. MockTransport handles every HTTP request locally."""

import json
from importlib import import_module
import unittest

from llm_trace import TokenPricing, Tracer
from llm_trace.integrations.openai import AsyncOpenAIResponses, OpenAIResponses

try:
    import openai
    from openai import AsyncOpenAI, OpenAI
    # SDK 3 uses httpx2; older SDKs use httpx. Match the installed SDK's transport.
    httpx = import_module("httpx2" if int(openai.__version__.split(".")[0]) >= 3 else "httpx")
except ImportError:
    httpx = None


def payload(status="completed"):
    return {
        "id": "resp_local_fixture", "object": "response", "created_at": 1,
        "status": status, "model": "fixture-model", "output": [],
        "usage": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15,
                  "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
        "error": {"code": "server_error", "message": "private-error"} if status == "failed" else None,
    }


@unittest.skipIf(httpx is None, "install openai to run optional SDK checks")
class OpenAISDKTests(unittest.TestCase):
    def test_actual_sdk_parsing_and_forwarding(self):
        received = []

        def handle(request):
            received.append(json.loads(request.content))
            return httpx.Response(200, json=payload())

        with OpenAI(api_key="local-test", base_url="https://local.invalid/v1", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
            tracer = Tracer()
            with tracer.trace("sdk-test") as trace:
                result = OpenAIResponses(tracer, client, pricing=TokenPricing(1, 2, "0.5")).create(model="fixture-model", input="private-input", prompt_version="v1")
        self.assertEqual(result.id, "resp_local_fixture")
        self.assertEqual(received, [{"model": "fixture-model", "input": "private-input"}])
        self.assertEqual(trace.spans[0].input_tokens, 12)
        self.assertEqual(trace.spans[0].cached_input_tokens, 0)
        self.assertEqual(trace.spans[0].estimated_cost_usd, 0.000018)
        self.assertNotIn("private-input", json.dumps(trace.to_dict()))

    def test_actual_sdk_returned_failure_does_not_capture_provider_message(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload("failed")))
        with OpenAI(api_key="local-test", base_url="https://local.invalid/v1", max_retries=0,
                    http_client=httpx.Client(transport=transport)) as client:
            tracer = Tracer()
            with tracer.trace("sdk-test") as trace:
                result = OpenAIResponses(tracer, client).create(model="fixture-model", input="private-input")
        self.assertEqual(result.status, "failed")
        self.assertEqual(trace.spans[0].status, "error")
        self.assertNotIn("private-error", json.dumps(trace.to_dict()))


@unittest.skipIf(httpx is None, "install openai to run optional SDK checks")
class AsyncOpenAISDKTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_async_sdk_parsing(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload()))
        async with AsyncOpenAI(api_key="local-test", base_url="https://local.invalid/v1", max_retries=0,
                               http_client=httpx.AsyncClient(transport=transport)) as client:
            tracer = Tracer()
            with tracer.trace("sdk-test") as trace:
                result = await AsyncOpenAIResponses(tracer, client, pricing=TokenPricing(1, 2)).create(model="fixture-model", input="private-input")
        self.assertEqual(result.id, "resp_local_fixture")
        self.assertEqual(trace.spans[0].output_tokens, 3)
        self.assertEqual(trace.spans[0].estimated_cost_usd, 0.000018)
