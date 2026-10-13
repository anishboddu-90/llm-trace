"""An offline walkthrough with artificial delays, usage counts, and token prices."""

from pathlib import Path
from time import sleep
from types import SimpleNamespace

from .integrations.openai import OpenAIResponses
from .pricing import TokenPricing
from .storage import SQLiteStore
from .tracer import Tracer


class _DemoResponses:
    def __init__(self, delay: float, cached: int):
        self.delay = delay
        self.cached = cached

    def create(self, **kwargs):
        sleep(self.delay)
        return SimpleNamespace(
            model="fake-local", status="completed", output_text="Synthetic demo response.",
            usage=SimpleNamespace(input_tokens=120, output_tokens=40,
                                  input_tokens_details=SimpleNamespace(cached_tokens=self.cached)),
        )


def run_demo(db: str | Path) -> dict[str, str]:
    """Append four synthetic traces and return their saved IDs."""
    store = SQLiteStore(db)
    tracer = Tracer(store=store)
    # Fictional USD rates, used only to demonstrate cost accounting.
    rates = TokenPricing(input_per_million=1, output_per_million=2, cached_input_per_million="0.25")
    ids = {}
    for name, delay, cached, fail_lookup, recover in (
        ("normal", 0.01, 0, False, False),
        ("slow-generation", 0.08, 0, False, False),
        ("recovered-lookup", 0.01, 80, True, True),
        ("failed-lookup", 0.01, 0, True, False),
    ):
        client = SimpleNamespace(responses=_DemoResponses(delay, cached))
        responses = OpenAIResponses(tracer, client, pricing=rates)
        try:
            with tracer.trace(name) as trace:
                with tracer.span("answer"):
                    try:
                        with tracer.span("retrieve", kind="tool"):
                            sleep(0.002)
                            if fail_lookup:
                                raise LookupError("simulated unavailable index")
                    except LookupError:
                        if not recover:
                            raise
                        with tracer.span("fallback-context", kind="tool"):
                            sleep(0.001)
                    responses.create(model="fake-local", input="Synthetic input.", prompt_version="demo-v1")
        except LookupError:
            pass  # The failed request was saved automatically when its trace closed.
        store.get(trace.id)  # Don't report a successful demo if a trace couldn't be saved.
        ids[name] = trace.id
    return ids
