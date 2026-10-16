# llm-trace

A small Python tool for figuring out where an LLM request spent its time and which calls failed.

Record request traces and nested model/tool calls, then inspect them from your terminal. It has no runtime dependencies and doesn't need an API key.

## Try it

Requires Python 3.11 or newer.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

To see it work before instrumenting your application:

```sh
llm-trace --db demo.sqlite3 demo
llm-trace --db demo.sqlite3 list
llm-trace --db demo.sqlite3 summary <slow-generation-id>
llm-trace --db demo.sqlite3 show <recovered-lookup-id>
```

The demo prints IDs for four requests: normal, slow generation, recovered lookup, and failed lookup. Copy an ID into the last two commands. It runs from the installed package, makes no API calls, and appends new traces when repeated. Storage accepts empty databases and existing llm-trace databases; it rejects unrelated SQLite schemas. The demo uses artificial delays, token counts, and prices. It shows automatic saving, the Responses adapter, cache-aware cost estimates, and inspection; it isn't a benchmark or a real model integration. `python examples/demo.py --db demo.sqlite3` still works from a checkout.

Existing databases must have the expected column types, required fields, and trace ID primary key. Column order can differ. A version number or matching column names alone won't pass the storage check; incompatible layouts are rejected without modification.

Saved traces are checked before inspection: required fields, value types, finite non-negative timings and costs, and valid span parents. Invalid records return an error naming the field, without printing its contents or changing the database. Older traces without cache counts or pricing snapshots remain readable. Unknown added fields are preserved. Saving also checks the record, which catches invalid edits to a finished trace before writing it.

## Instrument a request

```python
from llm_trace import SQLiteStore, Tracer

tracer = Tracer(store=SQLiteStore("traces.sqlite3"))

with tracer.trace("answer-question") as trace:
    with tracer.span("retrieve", kind="tool"):
        documents = ["some context"]  # Replace with your retrieval call.

    with tracer.span("generate", kind="model", model="your-model", prompt_version="v1") as call:
        # Make your model call here, then copy its reported token usage.
        call.set_usage(input_tokens=120, output_tokens=40)

print(trace.id)
```

```sh
llm-trace list
llm-trace show <trace-id>
llm-trace summary <trace-id>
llm-trace summary <trace-id> --json --slowest 3
llm-trace export <trace-id> > trace.json
llm-trace report <trace-id> --output trace.html
llm-trace compare <baseline-id> <candidate-id>
llm-trace compare <baseline-id> <candidate-id> --json
llm-trace --db /path/to/traces.sqlite3 list --status error --limit 10
```

`show`, `summary`, and `export` accept a unique ID prefix of at least four characters. An ambiguous prefix returns an error. Commands only read existing databases; they won't silently create an empty one at the wrong path.

`report` accepts the same ID prefixes and writes a standalone HTML file. Open it directly in your browser. It shows nested calls, duration bars, usage coverage, and failures; it has no JavaScript, external fonts, or network requests. The output path must be new: existing files are never overwritten. Reports include recorded names and model metadata.

## Compare two runs

```sh
llm-trace --db demo.sqlite3 compare <normal-id> <slow-generation-id>
```

`compare` reads two traces from the selected database. It reports differences in request duration, failed calls, tokens, and estimated cost. Token and cost differences are only calculated when both runs have complete measurements; missing values stay unknown. A trace with no model calls has zero model usage and cost.

Calls match by their full path of parent names and kinds, so two `generate` calls under different steps stay separate. Repeated calls at the same path compare their mean inclusive duration and retain their counts. Model and prompt-version changes are shown alongside matched calls. Added and removed paths are listed separately. Renaming a call makes it a different path.

Compare equivalent workloads when investigating a change. Two individual runs aren't a statistical regression test, and inclusive call durations can overlap. `--json` provides the same comparison data for scripts. Percentage changes are undefined when the baseline is zero.

From Python, use `compare_traces(baseline_dict, candidate_dict)` to compare traces loaded from any stores.

## Trace lifecycle

For automatic saving, pass a store when creating your tracer:

```python
tracer = Tracer(store=SQLiteStore("traces.sqlite3"))
with tracer.trace("answer"):
    with tracer.span("retrieve", kind="tool"):
        documents = retrieve(question)
```

Completed traces are saved once, including requests that raise or are cancelled. Ordinary save failures are logged through `llm_trace.tracer` and preserve your application's result or exception. Logs include the trace ID and failure type, not the storage error message. Store setup errors still raise normally. Saving is synchronous and happens after request timing ends, so reported latency excludes the database write. For high-volume async apps, the synchronous write needs to be accounted for separately.

Use either automatic or explicit saving for a trace; combining both tries to save the same ID twice.

`save()` accepts completed traces and opens its own SQLite connection. The same ID can't be saved twice. Save explicitly after leaving the trace; a database write is separate from your model call. If a request raises, save its completed trace from your application's exception handler or a `finally` block.

Use the same `with` blocks in async functions. Await child tasks before leaving their trace or parent span. Each async request keeps its own context.

Exceptions are re-raised after recording their type. If your application catches a tool failure and recovers, the request can finish successfully while the failed tool call stays visible.

## What gets recorded

Names, parent IDs, UTC start times, elapsed milliseconds, status, exception types, and any model name, prompt version, usage, or cost you provide. Cost is optional; the OpenAI adapter can estimate it from rates you supply. There is no built-in pricing table. Missing usage stays unknown rather than becoming zero.

Prompts, responses, tool arguments, and exception messages aren't captured. Names and model metadata are stored as provided, so keep secrets out of those fields.

`summary` ranks the slowest spans, lists failed calls even if the request recovered, and totals recorded usage and cost. It also shows how many model calls have those measurements. A partial total isn't an estimate for the whole request.

Span times include their children. A parent and child can both appear in the slowest list, and parallel calls can overlap. Use the request duration for wall time; don't add span durations together.

## OpenAI Responses

If you already use the OpenAI Python SDK, wrap its Responses calls:

```python
from llm_trace.integrations.openai import OpenAIResponses

tracer = Tracer(store=SQLiteStore("traces.sqlite3"))
responses = OpenAIResponses(tracer, client)  # Your existing OpenAI client.
with tracer.trace("answer") as trace:
    result = responses.create(model=model_name, input=question, prompt_version="v2")
```

`AsyncOpenAIResponses` works the same way with an `AsyncOpenAI` client; await its `create()` method. The wrapper returns the original SDK response, forwards request arguments, and records the returned model and input/output token counts when available. `prompt_version` belongs to the tracer and isn't sent to OpenAI. Request and response content aren't recorded.

Only non-streaming, foreground calls are supported. Streaming and background calls are rejected before contacting the provider. Timing covers the SDK call, including its internal retries; individual attempts aren't separated. Returned `failed` or `incomplete` responses mark the model span as failed but still return to your application. Exceptions and cancellation propagate unchanged.

For token-cost estimates, pass rates to either adapter:

```python
from llm_trace import TokenPricing

# Fictional rates for this example, in USD per million tokens.
rates = TokenPricing(input_per_million="2", output_per_million="10", cached_input_per_million="0.5")
responses = OpenAIResponses(tracer, client, pricing=rates)
```

Use rates appropriate to the model you call. One rate configuration applies to every call through that wrapper. The estimate charges uncached input, cached input, and output separately. If you omit the cached-input rate, all input uses the regular input rate. If a cached rate is configured but the cache count is missing or invalid, cost stays unknown. Missing token usage also leaves cost unknown. No rates means no estimate.

Estimates cover reported tokens at your supplied rates. The rate snapshot and valid cached-input count are saved with the model span; `show` and HTML reports display the cache count. There is no automatic price lookup or accounting for other provider charges. Existing saved traces remain readable; older traces have no cache count or rate snapshot.

The adapter uses the [official Responses API fields](https://developers.openai.com/api/reference/python/resources/responses/methods/create). Tests use local fixtures, not live API calls.

There is an opt-in live example. Install `openai` separately, set `OPENAI_API_KEY` in your environment, then run `python examples/openai_request.py --model <your-model>`. This makes a real API call using your account. The base package doesn't install or import the OpenAI SDK.

## Development

To measure local tracing and database-write overhead:

```sh
llm-trace benchmark
llm-trace benchmark --spans 20 --json > overhead.json
```

This runs the same CPU workload without tracing, with tracing in memory, and with SQLite saving. It makes no model calls and removes its temporary database afterward. Full request timings include the save; the write is also measured separately. See [the method and local measurements](docs/benchmark.md) for raw samples and limitations.

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
```

Tests use Python's standard library and cover trace isolation, storage, summaries, comparisons, HTML reports, provider adapters, the demo, benchmark failures, and the command-line interface.

Installing `openai` also enables three SDK contract tests. Their HTTP transport is mocked locally; they don't use real credentials or contact OpenAI.

Before a release, run [the build and clean-install check](docs/release-check.md). It builds a wheel from the source distribution, installs it in a fresh environment, and runs the tests and installed commands from outside the checkout. The check needs the separate `build` development tool; the base package still has no runtime dependencies.

## Current scope

Local traces, SQLite storage, terminal inspection, standalone HTML reports, and an explicit OpenAI Responses adapter. No live dashboard, streaming support, OpenTelemetry export, or output evaluations yet.

Before the first release: a reproducible case study on an existing app and a clean-install trial by someone outside the project. Local trace and storage overhead measurements are available above; they still need checking against an actual application workload.
