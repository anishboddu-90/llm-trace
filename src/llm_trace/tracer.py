"""Context-local tracing primitives. All times are measured in milliseconds."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
import logging
from time import perf_counter
from typing import Iterator, Literal, Protocol
from uuid import uuid4

SpanKind = Literal["model", "tool", "step"]


def _name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("name must be a non-empty string")
    return value


@dataclass
class Span:
    id: str
    parent_id: str | None
    name: str
    kind: SpanKind
    started_at: str
    duration_ms: float = 0.0
    status: str = "running"
    error_type: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    estimated_cost_usd: float | None = None
    token_pricing: dict[str, str | None] | None = None

    def record_error(self, error_type: str) -> None:
        """Mark a returned failure without raising an application exception."""
        if self.status != "running":
            raise RuntimeError("cannot change a finished span")
        self.error_type = _name(error_type)

    def set_usage(
        self, *, input_tokens: int, output_tokens: int,
        estimated_cost_usd: float | None = None,
        cached_input_tokens: int | None = None,
    ) -> None:
        """Use provider-reported counts and an optional caller-calculated cost."""
        if self.status != "running":
            raise RuntimeError("cannot change a finished span")
        if self.kind != "model":
            raise ValueError("token usage belongs on a model span")
        for count in (input_tokens, output_tokens):
            if type(count) is not int or count < 0:
                raise ValueError("token counts must be non-negative integers")
        if cached_input_tokens is not None:
            if type(cached_input_tokens) is not int or not 0 <= cached_input_tokens <= input_tokens:
                raise ValueError("cached input tokens must be between zero and input tokens")
        if estimated_cost_usd is not None:
            if (type(estimated_cost_usd) not in (int, float)
                    or not math.isfinite(estimated_cost_usd)
                    or estimated_cost_usd < 0):
                raise ValueError("cost must be a finite, non-negative number")
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cached_input_tokens = cached_input_tokens
        self.estimated_cost_usd = estimated_cost_usd


@dataclass
class Trace:
    id: str
    name: str
    started_at: str
    duration_ms: float = 0.0
    status: str = "running"
    error_type: str | None = None
    spans: list[Span] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a JSON-compatible snapshot of the trace."""
        return {"schema_version": 1, **asdict(self)}


@dataclass(frozen=True)
class _Active:
    tracer: "Tracer"
    trace: Trace
    parent: Span | None = None


_active: ContextVar[_Active | None] = ContextVar("llm_trace_active", default=None)
_logger = logging.getLogger(__name__)


class TraceSink(Protocol):
    def save(self, trace: Trace) -> None: ...


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class Tracer:
    def __init__(self, *, store: TraceSink | None = None):
        self.store = store

    def _save(self, trace: Trace) -> None:
        if self.store is None:
            return
        try:
            self.store.save(trace)
        except Exception as error:
            # Export failures must not replace an application result or exception.
            # Only record the type: a storage error message may contain secrets.
            try:
                _logger.warning("Trace %s could not be saved (%s)", trace.id, type(error).__name__)
            except Exception:
                pass  # A failing logging handler must not break the application either.

    @contextmanager
    def trace(self, name: str) -> Iterator[Trace]:
        """Start a request. Use ordinary `with` blocks in sync or async code."""
        if _active.get() is not None:
            raise RuntimeError("start a span inside a trace, not another trace")
        trace = Trace(id=uuid4().hex, name=_name(name), started_at=_timestamp())
        token = _active.set(_Active(self, trace))
        started = perf_counter()
        try:
            yield trace
        except BaseException as error:
            trace.status = "error"
            trace.error_type = type(error).__name__
            raise
        else:
            trace.status = "ok"
        finally:
            trace.duration_ms = (perf_counter() - started) * 1000
            _active.reset(token)
            self._save(trace)

    @contextmanager
    def span(
        self, name: str, *, kind: SpanKind = "step",
        model: str | None = None, prompt_version: str | None = None,
    ) -> Iterator[Span]:
        active = _active.get()
        if active is None or active.tracer is not self:
            raise RuntimeError("span must be inside this tracer's trace")
        if active.trace.status != "running":
            raise RuntimeError("trace has ended; await tasks before leaving it")
        if active.parent is not None and active.parent.status != "running":
            raise RuntimeError("parent span has ended; await its tasks before leaving it")
        if kind not in ("model", "tool", "step"):
            raise ValueError("kind must be model, tool, or step")
        for value in (model, prompt_version):
            if value is not None:
                _name(value)
        if kind != "model" and (model is not None or prompt_version is not None):
            raise ValueError("model and prompt_version belong on a model span")
        span = Span(
            id=uuid4().hex, parent_id=active.parent.id if active.parent else None,
            name=_name(name), kind=kind, started_at=_timestamp(),
            model=model, prompt_version=prompt_version,
        )
        active.trace.spans.append(span)
        token = _active.set(_Active(self, active.trace, span))
        started = perf_counter()
        try:
            yield span
        except BaseException as error:
            span.status = "error"
            span.error_type = type(error).__name__
            raise
        else:
            span.status = "error" if span.error_type is not None else "ok"
        finally:
            span.duration_ms = (perf_counter() - started) * 1000
            _active.reset(token)
