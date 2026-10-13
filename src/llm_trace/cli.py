"""Inspect saved traces from the terminal."""

import argparse
import json
from pathlib import Path
import sqlite3
import sys

from .storage import SQLiteStore
from .summary import summarize
from .report import render_report
from .comparison import compare_traces
from .demo import run_demo


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
        if span.get("cached_input_tokens") is not None:
            label += f"  cached={span['cached_input_tokens']} in"
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


def _comparison(data: dict) -> str:
    before, after = data["baseline"], data["candidate"]
    duration = data["duration_ms"]
    lines = [
        f"Baseline: {_display(before['name'])}  {before['trace_id']}",
        f"Candidate: {_display(after['name'])}  {after['trace_id']}",
        f"Status: {before['status']} -> {after['status']}",
        f"Request duration: {duration['baseline']:.2f} -> {duration['candidate']:.2f} ms ({duration['delta']:+.2f} ms)",
        f"Failed spans: {data['failed_spans']['baseline']} -> {data['failed_spans']['candidate']}",
    ]
    for field, label in (("input_tokens", "Input tokens"), ("output_tokens", "Output tokens"),
                         ("estimated_cost_usd", "Estimated cost (USD)")):
        metric = data[field]
        if metric["delta"] is None:
            lines.append(f"{label}: not comparable (missing measurements)")
        elif field == "estimated_cost_usd":
            lines.append(f"{label}: {metric['baseline']:.6f} -> {metric['candidate']:.6f} ({metric['delta']:+.6f})")
        else:
            lines.append(f"{label}: {metric['baseline']} -> {metric['candidate']} ({metric['delta']:+d})")
    lines.append("Calls (inclusive mean duration; groups can overlap):")
    for group in data["calls"]:
        path = " / ".join(f"{node['kind']} {node['name']}" for node in group["path"])
        lines.append(f"  {_display(path)}")
        base, new = group["baseline"], group["candidate"]
        if group["presence"] != "matched":
            side = new if new else base
            lines.append(f"    {group['presence']}: {side['count']} calls, mean {side['mean_duration_ms']:.2f} ms")
        else:
            delta = group["mean_duration_ms"]["delta"]
            lines.append(f"    mean {base['mean_duration_ms']:.2f} -> {new['mean_duration_ms']:.2f} ms ({delta:+.2f} ms); calls {base['count']} -> {new['count']}; failed {base['failed_calls']} -> {new['failed_calls']}")
            for field, label in (("models", "Models"), ("prompt_versions", "Prompt versions")):
                if base[field] != new[field]:
                    old = _display(", ".join(base[field])) or "not recorded"
                    current = _display(", ".join(new[field])) or "not recorded"
                    lines.append(f"    {label}: {old} -> {current}")
    lines.append("This compares two individual runs, not a statistical regression test.")
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
    comparison = commands.add_parser("compare", help="compare two saved runs")
    comparison.add_argument("baseline_id", help="baseline trace ID or unique prefix")
    comparison.add_argument("candidate_id", help="candidate trace ID or unique prefix")
    comparison.add_argument("--json", action="store_true", help="write comparison JSON to stdout")
    commands.add_parser("demo", help="append four offline example traces (no API calls)")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            ids = run_demo(args.db)
            print("Offline demo: artificial delays, token counts, and USD rates. No API calls.")
            for name, trace_id in ids.items():
                print(f"{name}: {trace_id}")
            return 0
        store = SQLiteStore(args.db, read_only=True)
        if args.command == "list":
            traces = store.list_recent(limit=args.limit, status=args.status)
            if not traces:
                print("No traces recorded.")
            for trace in traces:
                print(f"{trace['id']}  {trace['status']:5}  {trace['duration_ms']:9.2f} ms  {_display(trace['name'])}")
        elif args.command == "compare":
            baseline = store.get(store.resolve_id(args.baseline_id))
            candidate = store.get(store.resolve_id(args.candidate_id))
            data = compare_traces(baseline, candidate)
            print(json.dumps(data, indent=2, allow_nan=False) if args.json else _comparison(data))
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
