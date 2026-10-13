"""Opt-in live example. Requires the OpenAI SDK and your own API credentials."""

import argparse
import os

from llm_trace import SQLiteStore, Tracer
from llm_trace.integrations.openai import OpenAIResponses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="a model available to your API account")
    parser.add_argument("--db", default="traces.sqlite3")
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("set OPENAI_API_KEY before running this live example")
    from openai import OpenAI

    tracer = Tracer()
    store = SQLiteStore(args.db)
    with OpenAI() as client:
        try:
            with tracer.trace("openai-example") as trace:
                result = OpenAIResponses(tracer, client).create(
                    model=args.model, input="Explain what a trace span is in one sentence.",
                    max_output_tokens=100, prompt_version="example-v1",
                )
                print(result.output_text)
        finally:
            store.save(trace)
            print(f"Trace: {trace.id}")


if __name__ == "__main__":
    main()
