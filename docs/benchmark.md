# Tracing overhead

Run the benchmark after installing the package:

```sh
llm-trace benchmark
llm-trace benchmark --rounds 200 --warmup 20 --spans 20 --work 200
llm-trace benchmark --json > overhead.json
```

It makes no API calls and uses a temporary SQLite database, which is removed when the run ends. The global `--db` option is ignored for this command; your application's database is never opened. The base package still needs no runtime dependencies.

## What it measures

Three cases execute the same deterministic integer workload:

- Baseline: work without tracing.
- Memory: one request trace and a configurable number of sequential step spans, without saving.
- SQLite: the same traced request, saved through `Tracer(store=...)` before the outer timer stops.

`--work` controls the integer-loop iterations per span. `--spans` controls the number of steps per request. These are CPU steps, with no simulated network latency, token counts, or model responses. Each case must return the same checksum.

The cases run once per round, rotating their order between rounds. Startup, database creation, and the configured warmup rounds are excluded. Every request in the SQLite case opens its own connection and commits its trace, using the package's existing storage behavior. The benchmark times `save()` separately, including serialization, connection setup, the insert, and commit. That time is already included in the SQLite request time; do not add it again. The measuring wrapper itself adds a small amount of timing overhead.

Every SQLite request must save successfully, and the final row count must match the measured and warmup rounds. A swallowed save error invalidates the benchmark. Setup errors also fail the command.

JSON output includes configuration, Python and SQLite versions, storage settings, database size before cleanup, all measured samples, and median, mean, nearest-rank p95, minimum, and maximum times. Paired overhead subtracts the baseline from the corresponding traced sample in each round. The median of those differences can differ from the difference of the case medians. Negative samples are kept: operating-system scheduling and other activity can affect these short measurements.

## One local measurement

Measured October 14, 2026, on a MacBook Pro with an Apple M1 Pro and 16 GB memory. Python 3.13.1, Darwin 25.4.0, SQLite 3.45.3, arm64. The database used the default `delete` journal and `synchronous=2` (FULL). Both configurations used 200 measured rounds, 20 warmup rounds, and 200 CPU iterations per span. Other applications were running; the machine was not isolated for benchmarking.

All times below are milliseconds. These are observations from one run of each configuration, not performance guarantees.

| Measurement | 5 spans median | 5 spans p95 | 20 spans median | 20 spans p95 |
| --- | ---: | ---: | ---: | ---: |
| Baseline request | 0.071 | 0.243 | 0.276 | 0.582 |
| Traced in memory | 0.133 | 0.509 | 0.453 | 0.813 |
| Traced and saved | 1.188 | 4.725 | 1.823 | 3.903 |
| Save alone, within traced request | 1.024 | 4.182 | 1.342 | 3.130 |
| Paired memory overhead | 0.058 | 0.361 | 0.152 | 0.463 |
| Paired traced-and-saved overhead | 1.100 | 4.654 | 1.527 | 3.534 |

Each configuration verified 220 saved traces. The databases occupied 950,272 and 1,851,392 bytes respectively, including indexes, warmup traces, and SQLite page allocation. Those totals aren't estimates of bytes per production request.

Raw measurements: [five spans](benchmarks/2026-10-14-5-spans.json) and [twenty spans](benchmarks/2026-10-14-20-spans.json).

Database saving dominated the median overhead here. It is synchronous, including when tracing an async application, and runs after the trace's own duration stops. Use the full request timings above when accounting for that write; saved trace durations alone omit it.

This benchmark doesn't measure concurrent requests, large nested traces, SDK adapters, provider retries, a growing production database, other disks, or end-to-end application latency. Run it on your target machine and measure the actual application before deciding whether synchronous saving is acceptable. The existing-application case study remains separate work.
