# llm-trace

A small Python tool for figuring out where an LLM request spent its time and which calls failed.

Record request traces and nested model/tool calls, then inspect them from your terminal. It has no runtime dependencies and doesn't need an API key.

## Try it

Requires Python 3.11 or newer.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

To see it work before instrumenting your application:

```sh
python examples/demo.py --db demo.sqlite3
llm-trace --db demo.sqlite3 list
llm-trace --db demo.sqlite3 summary <slow-generation-id>
llm-trace --db demo.sqlite3 show <recovered-lookup-id>
```

The demo prints IDs for four requests: normal, slow generation, recovered lookup, and failed lookup. Copy an ID into the last two commands. It uses artificial delays, token counts, and prices. It shows how to inspect a problem; it isn't a benchmark or a real model integration.

## Instrument a request

```python
from llm_trace import SQLiteStore, Tracer

tracer = Tracer()

with tracer.trace("answer-question") as trace:
    with tracer.span("retrieve", kind="tool"):
        documents = ["some context"]  # Replace with your retrieval call.

    with tracer.span("generate", kind="model", model="your-model", prompt_version="v1") as call:
        # Make your model call here, then copy its reported token usage.
        call.set_usage(input_tokens=120, output_tokens=40)

SQLiteStore("traces.sqlite3").save(trace)
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

`save()` accepts completed traces and opens its own SQLite connection. The same ID can't be saved twice. Save explicitly after leaving the trace; a database write is separate from your model call. If a request raises, save its completed trace from your application's exception handler or a `finally` block.

Use the same `with` blocks in async functions. Await child tasks before leaving their trace or parent span. Each async request keeps its own context.

Exceptions are re-raised after recording their type. If your application catches a tool failure and recovers, the request can finish successfully while the failed tool call stays visible.

## What gets recorded

Names, parent IDs, UTC start times, elapsed milliseconds, status, exception types, and any model name, prompt version, usage, or cost you provide. Cost is optional and caller-calculated; there is no built-in pricing table. Missing usage stays unknown rather than becoming zero.

Prompts, responses, tool arguments, and exception messages aren't captured. Names and model metadata are stored as provided, so keep secrets out of those fields.

`summary` ranks the slowest spans, lists failed calls even if the request recovered, and totals recorded usage and cost. It also shows how many model calls have those measurements. A partial total isn't an estimate for the whole request.

Span times include their children. A parent and child can both appear in the slowest list, and parallel calls can overlap. Use the request duration for wall time; don't add span durations together.

## OpenAI Responses

If you already use the OpenAI Python SDK, wrap its Responses calls:

```python
from llm_trace.integrations.openai import OpenAIResponses

responses = OpenAIResponses(tracer, client)  # Your existing OpenAI client.
with tracer.trace("answer") as trace:
    result = responses.create(model=model_name, input=question, prompt_version="v2")
SQLiteStore("traces.sqlite3").save(trace)
```

`AsyncOpenAIResponses` works the same way with an `AsyncOpenAI` client; await its `create()` method. The wrapper returns the original SDK response, forwards request arguments, and records the returned model and input/output token counts when available. `prompt_version` belongs to the tracer and isn't sent to OpenAI. Request and response content aren't recorded.

Only non-streaming, foreground calls are supported. Streaming and background calls are rejected before contacting the provider. Timing covers the SDK call, including its internal retries; individual attempts aren't separated. Returned `failed` or `incomplete` responses mark the model span as failed but still return to your application. Exceptions and cancellation propagate unchanged. This adapter doesn't estimate cost.

The adapter uses the [official Responses API fields](https://developers.openai.com/api/reference/python/resources/responses/methods/create). Tests use local fixtures, not live API calls.

There is an opt-in live example. Install `openai` separately, set `OPENAI_API_KEY` in your environment, then run `python examples/openai_request.py --model <your-model>`. This makes a real API call using your account. The base package doesn't install or import the OpenAI SDK.

## Development

```sh
python -m unittest discover -s tests -v
```

Tests use Python's standard library and cover trace isolation, storage, summaries, comparisons, HTML reports, provider adapters, the demo, and the command-line interface.

Installing `openai` also enables three SDK contract tests. Their HTTP transport is mocked locally; they don't use real credentials or contact OpenAI.

## Current scope

Local traces, SQLite storage, terminal inspection, standalone HTML reports, and an explicit OpenAI Responses adapter. No live dashboard, streaming support, OpenTelemetry export, or output evaluations yet.
