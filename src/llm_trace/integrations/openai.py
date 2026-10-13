"""Trace non-streaming OpenAI Responses calls using a caller-owned SDK client."""

from typing import Any

from ..tracer import Span, Tracer
from ..pricing import TokenPricing


def _validate(kwargs: dict) -> None:
    if kwargs.get("stream"):
        raise ValueError("streaming is not supported; use stream=False")
    if kwargs.get("background"):
        raise ValueError("background responses are not supported; use background=False")


def _record(span: Span, response: Any, pricing: TokenPricing | None) -> None:
    # Read only an allowlist. Never serialize a response or request body.
    model = getattr(response, "model", None)
    if isinstance(model, str) and model.strip():
        span.model = model
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if pricing is not None:
        span.token_pricing = pricing.to_dict()
    if all(type(count) is int and count >= 0 for count in (input_tokens, output_tokens)):
        cached = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", None)
        if type(cached) is not int or not 0 <= cached <= input_tokens:
            cached = None
        cost = pricing.estimate(input_tokens=input_tokens, output_tokens=output_tokens,
                                cached_input_tokens=cached) if pricing is not None else None
        span.set_usage(input_tokens=input_tokens, output_tokens=output_tokens,
                       cached_input_tokens=cached, estimated_cost_usd=cost)
    status = getattr(response, "status", None)
    if status == "failed":
        span.record_error("ResponseFailed")
    elif status == "incomplete":
        span.record_error("ResponseIncomplete")


class OpenAIResponses:
    """A small wrapper around `client.responses.create`, without an SDK dependency."""

    def __init__(self, tracer: Tracer, client: Any, *, pricing: TokenPricing | None = None):
        if pricing is not None and not isinstance(pricing, TokenPricing):
            raise TypeError("pricing must be TokenPricing or None")
        self.tracer = tracer
        self.client = client
        self.pricing = pricing

    def create(self, *, prompt_version: str | None = None, **kwargs: Any) -> Any:
        _validate(kwargs)
        with self.tracer.span(
            "openai.responses.create", kind="model", model=kwargs.get("model"),
            prompt_version=prompt_version,
        ) as span:
            response = self.client.responses.create(**kwargs)
            _record(span, response, self.pricing)
            return response


class AsyncOpenAIResponses:
    """The same adapter for a caller-owned `AsyncOpenAI` client."""

    def __init__(self, tracer: Tracer, client: Any, *, pricing: TokenPricing | None = None):
        if pricing is not None and not isinstance(pricing, TokenPricing):
            raise TypeError("pricing must be TokenPricing or None")
        self.tracer = tracer
        self.client = client
        self.pricing = pricing

    async def create(self, *, prompt_version: str | None = None, **kwargs: Any) -> Any:
        _validate(kwargs)
        with self.tracer.span(
            "openai.responses.create", kind="model", model=kwargs.get("model"),
            prompt_version=prompt_version,
        ) as span:
            response = await self.client.responses.create(**kwargs)
            _record(span, response, self.pricing)
            return response
