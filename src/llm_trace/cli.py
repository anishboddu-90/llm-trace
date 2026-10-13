"""Inspect saved traces from the terminal."""

import argparse
import json
from pathlib import Path
import sqlite3
import sys

from .storage import SQLiteStore
from .summary import summarize
from .report import render_report


def _display(value: str) -> str:
    # Make newlines and terminal control characters visible rather than executing them.
    return json.dumps(value, ensure_ascii=True)[1:-1]


def _show(trace: dict) -> str:
    lines = [
        f"{_display(trace['name'])}  {trace['status']}  {trace['duration_ms']:.2f} ms",
        f"ID: {trace['id']}",
        f"Started: {trace['started_at']}",
    ]
    if trace["error_type"]:
        lines.append(f"Error: {_display(trace['error_type'])}")
    spans = {span["id"]: span for span in trace["spans"]}
    for span in trace["spans"]:
        depth, parent, seen = 0, span["parent_id"], {span["id"]}
        while parent in spans and parent not in seen:
            seen.add(parent)
            depth += 1
            parent = spans[parent]["parent_id"]
        label = f"{span['kind']} {_display(span['name'])}  {span['status']}  {span['duration_ms']:.2f} ms"
        if span["model"]:
            label += f"  model={_display(span['model'])}"
        if span["prompt_version"]:
            label += f"  prompt={_display(span['prompt_version'])}"
        if span["input_tokens"] is not None:
            label += f"  tokens={span['input_tokens']} in/{span['output_tokens']} out"
        if span["estimated_cost_usd"] is not None:
            label += f"  cost=${span['estimated_cost_usd']:.6f}"
        if span["error_type"]:
            label += f"  error={_display(span['error_type'])}"
        lines.append("  " * (depth + 1) + label)
    if not spans:
        lines.append("No spans recorded.")
    return "\n".join(lines)


def _summary(data: dict) -> str:
    lines = [
        f"{_display(data['name'])}  {data['status']}  {data['duration_ms']:.2f} ms",
        f"Spans: {data['span_count']}  Model calls: {data['model_calls']}",
    ]
    if data["error_type"]:
        lines.append(f"Error: {_display(data['error_type'])}")
    if data["model_calls"]:
        if data["recorded_input_tokens"] is None:
            lines.append("Recorded tokens: unknown")
        else:
            lines.append(f"Recorded tokens: {data['recorded_input_tokens']} in / {data['recorded_output_tokens']} out")
        lines.append(f"Usage coverage: {data['model_calls_with_usage']}/{data['model_calls']} model calls")
        if data["recorded_estimated_cost_usd"] is None:
            lines.append("Recorded estimated cost: unknown")
        else:
            lines.append(f"Recorded estimated cost: ${data['recorded_estimated_cost_usd']:.6f}")
        lines.append(f"Cost coverage: {data['model_calls_with_cost']}/{data['model_calls']} model calls")
    lines.append(f"Failed spans: {len(data['failed_spans'])}")
    for failed in data["failed_spans"]:
        lines.append(f"  {failed['kind']} {_display(failed['name'])}: {_display(failed['error_type'] or 'unknown')}")
    lines.append("Slowest spans (inclusive; overlapping times are not added):")
    for span in data["slowest_spans"]:
        lines.append(f"  {span['duration_ms']:9.2f} ms  {span['kind']} {_display(span['name'])}  {span['status']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-trace", description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("traces.sqlite3"), help="trace database path")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="list recent traces")
    listing.add_argument("--limit", type=int, default=20)
    listing.add_argument("--status", choices=("ok", "error"))
    for name in ("show", "export"):
        command = commands.add_parser(name, help="inspect a trace" if name == "show" else "write trace JSON to stdout")
        command.add_argument("trace_id", help="full ID or a unique prefix of at least four characters")
    summary = commands.add_parser("summary", help="show slow calls, failures, and recorded usage")
    summary.add_argument("trace_id", help="full ID or a unique prefix of at least four characters")
    summary.add_argument("--slowest", type=int, default=5, help="number of spans to rank (1 to 100)")
    summary.add_argument("--json", action="store_true", help="write summary JSON to stdout")
    report = commands.add_parser("report", help="save a standalone HTML report")
    report.add_argument("trace_id", help="full ID or a unique prefix of at least four characters")
    report.add_argument("--output", required=True, type=Path, help="new HTML file path (never overwrites)")
    args = parser.parse_args(argv)
    try:
        store = SQLiteStore(args.db, read_only=True)
        if args.command == "list":
            traces = store.list_recent(limit=args.limit, status=args.status)
            if not traces:
                print("No traces recorded.")
            for trace in traces:
                print(f"{trace['id']}  {trace['status']:5}  {trace['duration_ms']:9.2f} ms  {_display(trace['name'])}")
        else:
            trace = store.get(store.resolve_id(args.trace_id))
            if args.command == "summary":
                data = summarize(trace, slowest=args.slowest)
                print(json.dumps(data, indent=2, allow_nan=False) if args.json else _summary(data))
            elif args.command == "report":
                document = render_report(trace)
                with args.output.open("x", encoding="utf-8") as output:
                    output.write(document)
                print(f"Report: {_display(str(args.output))}")
            else:
                print(json.dumps(trace, indent=2, allow_nan=False) if args.command == "export" else _show(trace))
    except (OSError, sqlite3.Error, ValueError, KeyError) as error:
        message = error.args[0] if isinstance(error, KeyError) else str(error)
        print(f"llm-trace: {_display(str(message))}", file=sys.stderr)
        return 1
    return 0
