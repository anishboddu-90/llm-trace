"""Trace non-streaming OpenAI Responses calls using a caller-owned SDK client."""

from typing import Any

from ..tracer import Span, Tracer


def _validate(kwargs: dict) -> None:
    if kwargs.get("stream"):
        raise ValueError("streaming is not supported; use stream=False")
    if kwargs.get("background"):
        raise ValueError("background responses are not supported; use background=False")


def _record(span: Span, response: Any) -> None:
    # Read only an allowlist. Never serialize a response or request body.
    model = getattr(response, "model", None)
    if isinstance(model, str) and model.strip():
        span.model = model
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if all(type(count) is int and count >= 0 for count in (input_tokens, output_tokens)):
        span.set_usage(input_tokens=input_tokens, output_tokens=output_tokens)
    status = getattr(response, "status", None)
    if status == "failed":
        span.record_error("ResponseFailed")
    elif status == "incomplete":
        span.record_error("ResponseIncomplete")


class OpenAIResponses:
    """A small wrapper around `client.responses.create`, without an SDK dependency."""

    def __init__(self, tracer: Tracer, client: Any):
        self.tracer = tracer
        self.client = client

    def create(self, *, prompt_version: str | None = None, **kwargs: Any) -> Any:
        _validate(kwargs)
        with self.tracer.span(
            "openai.responses.create", kind="model", model=kwargs.get("model"),
            prompt_version=prompt_version,
        ) as span:
            response = self.client.responses.create(**kwargs)
            _record(span, response)
            return response


class AsyncOpenAIResponses:
    """The same adapter for a caller-owned `AsyncOpenAI` client."""

    def __init__(self, tracer: Tracer, client: Any):
        self.tracer = tracer
        self.client = client

    async def create(self, *, prompt_version: str | None = None, **kwargs: Any) -> Any:
        _validate(kwargs)
        with self.tracer.span(
            "openai.responses.create", kind="model", model=kwargs.get("model"),
            prompt_version=prompt_version,
        ) as span:
            response = await self.client.responses.create(**kwargs)
            _record(span, response)
            return response
