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
llm-trace export <trace-id> > trace.json
llm-trace --db /path/to/traces.sqlite3 list --status error --limit 10
```

`show` and `export` accept a unique ID prefix of at least four characters. An ambiguous prefix returns an error. Commands only read existing databases; they won't silently create an empty one at the wrong path.

`save()` accepts completed traces and opens its own SQLite connection. The same ID can't be saved twice. Save explicitly after leaving the trace; a database write is separate from your model call. If a request raises, save its completed trace from your application's exception handler or a `finally` block.

Use the same `with` blocks in async functions. Await child tasks before leaving their trace or parent span. Each async request keeps its own context.

Exceptions are re-raised after recording their type. If your application catches a tool failure and recovers, the request can finish successfully while the failed tool call stays visible.

## What gets recorded

Names, parent IDs, UTC start times, elapsed milliseconds, status, exception types, and any model name, prompt version, usage, or cost you provide. Cost is optional and caller-calculated; there is no built-in pricing table. Missing usage stays unknown rather than becoming zero.

Prompts, responses, tool arguments, and exception messages aren't captured. Names and model metadata are stored as provided, so keep secrets out of those fields.

## Development

```sh
python -m unittest discover -s tests -v
```

Next: a runnable example and a summary of slow calls, failures, and known token usage. A dashboard and provider integrations can come after the trace format is useful on its own.
