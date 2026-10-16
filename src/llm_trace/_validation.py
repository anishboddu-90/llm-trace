"""Validate persisted records before inspection, without echoing their contents."""

import math
import re

from .pricing import TokenPricing


def _fail(field: str, requirement: str) -> None:
    raise ValueError(f"invalid trace: {field} {requirement}")


def _text(value, field: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not value.strip():
        _fail(field, "must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeError:
        _fail(field, "must contain valid Unicode text")


def _identifier(value, field: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{32}", value) is None:
        _fail(field, "must be a 32-character hexadecimal ID")


def _number(value, field: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    try:
        valid = type(value) in (int, float) and value >= 0 and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        _fail(field, "must be a finite, non-negative number")


def _required(data: dict, keys: tuple[str, ...], field: str) -> None:
    for key in keys:
        if key not in data:
            _fail(f"{field}.{key}", "is missing")


def validate_header(data: dict) -> None:
    """Check the indexed fields used by list and trace lookup."""
    _required(data, ("id", "name", "started_at", "status", "duration_ms"), "trace")
    _identifier(data["id"], "trace.id")
    for field in ("name", "started_at"):
        _text(data[field], f"trace.{field}")
    if data["status"] not in ("ok", "error"):
        _fail("trace.status", "must be ok or error")
    _number(data["duration_ms"], "trace.duration_ms")


def validate_trace(data) -> None:
    """Accept schema 1, including older records without cache/pricing fields.

    Unknown fields are retained for forward compatibility. Parent checks are
    iterative so a deeply nested but valid trace doesn't exhaust the Python stack.
    """
    if not isinstance(data, dict):
        _fail("record", "must be an object")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("unsupported trace schema version")
    validate_header(data)
    _required(data, ("error_type", "spans"), "trace")
    _text(data["error_type"], "trace.error_type", optional=True)
    if not isinstance(data["spans"], list):
        _fail("trace.spans", "must be a list")
    parents = {}
    for index, span in enumerate(data["spans"]):
        field = f"spans[{index}]"
        if not isinstance(span, dict):
            _fail(field, "must be an object")
        _required(span, ("id", "parent_id", "name", "kind", "started_at", "duration_ms",
                         "status", "error_type", "model", "prompt_version", "input_tokens",
                         "output_tokens", "estimated_cost_usd"), field)
        _identifier(span["id"], f"{field}.id")
        _identifier(span["parent_id"], f"{field}.parent_id", optional=True)
        if span["id"] in parents:
            _fail("span IDs", "must be unique")
        parents[span["id"]] = span["parent_id"]
        for name in ("name", "started_at", "error_type", "model", "prompt_version"):
            _text(span[name], f"{field}.{name}", optional=name in ("error_type", "model", "prompt_version"))
        if span["kind"] not in ("step", "tool", "model"):
            _fail(f"{field}.kind", "must be step, tool, or model")
        if span["status"] not in ("ok", "error"):
            _fail(f"{field}.status", "must be ok or error")
        _number(span["duration_ms"], f"{field}.duration_ms")
        _number(span["estimated_cost_usd"], f"{field}.estimated_cost_usd", optional=True)
        for name in ("input_tokens", "output_tokens", "cached_input_tokens"):
            count = span.get(name)
            if count is not None and (type(count) is not int or count < 0):
                _fail(f"{field}.{name}", "must be a non-negative integer or null")
        cached = span.get("cached_input_tokens")
        if cached is not None and (span["input_tokens"] is None or cached > span["input_tokens"]):
            _fail(f"{field}.cached_input_tokens", "must not exceed recorded input tokens")
        pricing = span.get("token_pricing")
        if pricing is not None:
            if not isinstance(pricing, dict):
                _fail(f"{field}.token_pricing", "must be an object or null")
            _required(pricing, ("input_per_million", "output_per_million"), f"{field}.token_pricing")
            rates = {name: pricing.get(name) for name in (
                "input_per_million", "output_per_million", "cached_input_per_million")}
            try:
                TokenPricing(**rates)
            except ValueError:
                _fail(f"{field}.token_pricing", "must contain finite, non-negative rates")

    checked = set()
    for span_id in parents:
        path = set()
        while span_id is not None and span_id not in checked:
            if span_id not in parents:
                _fail("span parents", "must refer to recorded spans")
            if span_id in path:
                _fail("span parents", "must not contain a cycle")
            path.add(span_id)
            span_id = parents[span_id]
        checked.update(path)
