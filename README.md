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
llm-trace --db /path/to/traces.sqlite3 list --status error --limit 10
```

`show`, `summary`, and `export` accept a unique ID prefix of at least four characters. An ambiguous prefix returns an error. Commands only read existing databases; they won't silently create an empty one at the wrong path.

`save()` accepts completed traces and opens its own SQLite connection. The same ID can't be saved twice. Save explicitly after leaving the trace; a database write is separate from your model call. If a request raises, save its completed trace from your application's exception handler or a `finally` block.

Use the same `with` blocks in async functions. Await child tasks before leaving their trace or parent span. Each async request keeps its own context.

Exceptions are re-raised after recording their type. If your application catches a tool failure and recovers, the request can finish successfully while the failed tool call stays visible.

## What gets recorded

Names, parent IDs, UTC start times, elapsed milliseconds, status, exception types, and any model name, prompt version, usage, or cost you provide. Cost is optional and caller-calculated; there is no built-in pricing table. Missing usage stays unknown rather than becoming zero.

Prompts, responses, tool arguments, and exception messages aren't captured. Names and model metadata are stored as provided, so keep secrets out of those fields.

`summary` ranks the slowest spans, lists failed calls even if the request recovered, and totals recorded usage and cost. It also shows how many model calls have those measurements. A partial total isn't an estimate for the whole request.

Span times include their children. A parent and child can both appear in the slowest list, and parallel calls can overlap. Use the request duration for wall time; don't add span durations together.

## Development

```sh
python -m unittest discover -s tests -v
```

Tests use Python's standard library and cover trace isolation, storage, summaries, the demo, and the command-line interface.

## Current scope

Local traces, SQLite storage, and terminal inspection. No dashboard, automatic provider instrumentation, OpenTelemetry export, or output evaluations yet. The next step is to instrument a real application and see which missing details make debugging harder.
