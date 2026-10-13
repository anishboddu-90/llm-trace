"""Generate synthetic traces locally. No provider account or API key needed."""

import argparse
from pathlib import Path
from llm_trace.demo import run_demo as run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("demo.sqlite3"))
    args = parser.parse_args()
    for name, trace_id in run(args.db).items():
        print(f"{name}: {trace_id}")
