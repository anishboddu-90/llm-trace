"""Measure local tracing overhead with a fixed CPU workload and disposable storage."""

from contextlib import closing
from datetime import datetime, timezone
import math
from pathlib import Path
import platform
import sqlite3
import statistics
from tempfile import TemporaryDirectory
from time import perf_counter_ns

from .storage import SQLiteStore
from .tracer import Trace, Tracer


def _stats(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.mean(ordered),
        "p95_ms": ordered[math.ceil(len(ordered) * 0.95) - 1],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "samples_ms": samples,
    }


def _work(iterations: int, seed: int) -> int:
    value = seed
    for _ in range(iterations):
        value = (value * 1664525 + 1013904223) & 0xFFFFFFFF
    return value


class _TimedStore:
    def __init__(self, store: SQLiteStore):
        self.store = store
        self.samples: list[float] = []

    def save(self, trace: Trace) -> None:
        started = perf_counter_ns()
        self.store.save(trace)
        # Only successful writes count. Tracer deliberately swallows ordinary save errors.
        self.samples.append((perf_counter_ns() - started) / 1_000_000)


def run_benchmark(*, rounds: int = 200, warmup: int = 20,
                  spans: int = 5, work: int = 200) -> dict:
    """Compare identical work without tracing, in memory, and with synchronous saving.

    Each round rotates the case order. Startup and warmup are excluded; the SQLite
    case includes serialization, connection setup, insert, and commit on every request.
    No model calls are made and no existing trace database is opened.
    """
    bounds = {"rounds": (rounds, 1, 10000), "warmup": (warmup, 0, 10000),
              "spans": (spans, 1, 100), "work": (work, 0, 100000)}
    for name, (value, minimum, maximum) in bounds.items():
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    if (rounds + warmup) * spans > 100000 or (rounds + warmup) * spans * work * 3 > 50000000:
        raise ValueError("benchmark too large; reduce rounds, warmup, spans, or work")

    with TemporaryDirectory(prefix="llm-trace-benchmark-") as directory:
        path = Path(directory) / "benchmark.sqlite3"
        sink = _TimedStore(SQLiteStore(path))
        memory = Tracer()
        persisted = Tracer(store=sink)
        with closing(sqlite3.connect(path)) as connection:
            journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
            synchronous = connection.execute("PRAGMA synchronous").fetchone()[0]

        def baseline() -> int:
            return sum(_work(work, index + 1) for index in range(spans))

        def traced(tracer: Tracer) -> int:
            checksum = 0
            with tracer.trace("benchmark-request"):
                for index in range(spans):
                    with tracer.span("cpu-step", kind="step"):
                        checksum += _work(work, index + 1)
            return checksum

        cases = {"baseline": baseline, "memory": lambda: traced(memory),
                 "sqlite": lambda: traced(persisted)}
        names = list(cases)
        samples: dict[str, list[float]] = {name: [] for name in names}
        expected_checksum = baseline()
        for round_index in range(warmup + rounds):
            offset = round_index % len(names)
            for name in names[offset:] + names[:offset]:
                writes_before = len(sink.samples)
                started = perf_counter_ns()
                checksum = cases[name]()
                elapsed = (perf_counter_ns() - started) / 1_000_000
                if checksum != expected_checksum:
                    raise ValueError(f"{name} benchmark workload changed its result")
                if name == "sqlite" and len(sink.samples) != writes_before + 1:
                    raise ValueError("benchmark trace was not saved; measurements discarded")
                if round_index >= warmup:
                    samples[name].append(elapsed)

        with closing(sqlite3.connect(path)) as connection:
            saved = connection.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
        if saved != warmup + rounds:
            raise ValueError("benchmark trace count is incorrect; measurements discarded")
        return {
            "schema_version": 1,
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "environment": {"python": platform.python_version(), "system": platform.system(),
                            "release": platform.release(), "machine": platform.machine(),
                            "sqlite": sqlite3.sqlite_version,
                            "journal_mode": journal_mode, "synchronous": synchronous},
            "config": {"rounds": rounds, "warmup": warmup, "spans": spans, "work": work},
            "checksum": expected_checksum,
            "saved_traces": saved,
            "database_bytes": path.stat().st_size,
            "cases": {name: _stats(values) for name, values in samples.items()},
            "save": _stats(sink.samples[warmup:]),
            "paired_overhead": {
                name: _stats([current - plain for current, plain in zip(samples[name], samples["baseline"])])
                for name in ("memory", "sqlite")
            },
        }


def format_benchmark(result: dict) -> str:
    config = result["config"]
    lines = [
        f"Local overhead benchmark: {config['rounds']} rounds, {config['warmup']} warmup, "
        f"{config['spans']} spans/request, {config['work']} CPU iterations/span",
        "Case                 Median ms     P95 ms",
    ]
    for name, data in result["cases"].items():
        lines.append(f"{name:20} {data['median_ms']:9.4f}  {data['p95_ms']:9.4f}")
    save = result["save"]
    lines.append(f"{'save (within sqlite)':20} {save['median_ms']:9.4f}  {save['p95_ms']:9.4f}")
    lines.append("Paired overhead vs baseline (median ms): " + ", ".join(
        f"{name} {data['median_ms']:+.4f}" for name, data in result["paired_overhead"].items()))
    lines.append(f"Verified {result['saved_traces']} saved traces including warmup; temporary database removed.")
    lines.append("Full request timings include saving. CPU microbenchmark only; results depend on your machine.")
    return "\n".join(lines)
