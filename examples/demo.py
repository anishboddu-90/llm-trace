"""Generate synthetic traces locally. No provider account or API key needed."""

import argparse
from pathlib import Path
from time import sleep

from llm_trace import SQLiteStore, Tracer


def run(db: Path) -> dict[str, str]:
    tracer = Tracer()
    store = SQLiteStore(db)
    ids = {}
    for name, delay, fail_lookup, recover in (
        ("normal", 0.01, False, False),
        ("slow-generation", 0.08, False, False),
        ("recovered-lookup", 0.01, True, True),
        ("failed-lookup", 0.01, True, False),
    ):
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
                    with tracer.span("generate", kind="model", model="fake-local", prompt_version="demo-v1") as call:
                        sleep(delay)
                        # Synthetic counts and price: these are not provider measurements.
                        call.set_usage(input_tokens=120, output_tokens=40, estimated_cost_usd=0.0002)
        except LookupError:
            # Keep the failed request so it can be inspected next to successful ones.
            pass
        finally:
            store.save(trace)
        ids[name] = trace.id
    return ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("demo.sqlite3"))
    args = parser.parse_args()
    for name, trace_id in run(args.db).items():
        print(f"{name}: {trace_id}")
