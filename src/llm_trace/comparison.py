"""Compare two individual runs. These differences are not a benchmark verdict."""

from .summary import summarize


def _difference(baseline, candidate) -> dict:
    known = baseline is not None and candidate is not None
    delta = candidate - baseline if known else None
    return {"baseline": baseline, "candidate": candidate, "delta": delta,
            "percent_change": delta / baseline * 100 if known and baseline != 0 else None}


def _complete(data: dict, field: str, missing: str):
    if data[missing]:
        return None
    # No model calls means no model token usage or model cost in this trace.
    return data[field] if data["model_calls"] else 0


def _groups(trace: dict) -> dict:
    spans = {span["id"]: span for span in trace["spans"]}
    if len(spans) != len(trace["spans"]):
        raise ValueError("trace contains duplicate span IDs")
    groups = {}
    for span in trace["spans"]:
        path, seen, node = [], set(), span
        while node is not None:
            if node["id"] in seen:
                raise ValueError("trace contains a cycle in span parents")
            seen.add(node["id"])
            path.append((node["kind"], node["name"]))
            parent = node["parent_id"]
            if parent is not None and parent not in spans:
                raise ValueError("trace contains a missing parent span")
            node = spans[parent] if parent is not None else None
        key = tuple(reversed(path))
        groups.setdefault(key, []).append(span)
    return {
        path: {
            "count": len(calls),
            "mean_duration_ms": sum(call["duration_ms"] for call in calls) / len(calls),
            "failed_calls": sum(call["status"] == "error" for call in calls),
            "models": sorted({call["model"] for call in calls if call["model"] is not None}),
            "prompt_versions": sorted({call["prompt_version"] for call in calls if call["prompt_version"] is not None}),
        }
        for path, calls in groups.items()
    }


def compare_traces(baseline: dict, candidate: dict) -> dict:
    """Match kind/name paths and compare per-call means, keeping repeats visible."""
    before, after = summarize(baseline), summarize(candidate)
    base_groups, candidate_groups = _groups(baseline), _groups(candidate)
    calls = []
    for path in sorted(base_groups.keys() | candidate_groups.keys()):
        base, new = base_groups.get(path), candidate_groups.get(path)
        calls.append({
            "path": [{"kind": kind, "name": name} for kind, name in path],
            "presence": "matched" if base and new else "added" if new else "removed",
            "baseline": base, "candidate": new,
            "mean_duration_ms": _difference(base["mean_duration_ms"] if base else None,
                                             new["mean_duration_ms"] if new else None),
        })
    return {
        "baseline": {key: before[key] for key in ("trace_id", "name", "status")},
        "candidate": {key: after[key] for key in ("trace_id", "name", "status")},
        "duration_ms": _difference(before["duration_ms"], after["duration_ms"]),
        "failed_spans": _difference(len(before["failed_spans"]), len(after["failed_spans"])),
        "input_tokens": _difference(_complete(before, "recorded_input_tokens", "model_calls_missing_usage"),
                                    _complete(after, "recorded_input_tokens", "model_calls_missing_usage")),
        "output_tokens": _difference(_complete(before, "recorded_output_tokens", "model_calls_missing_usage"),
                                     _complete(after, "recorded_output_tokens", "model_calls_missing_usage")),
        "estimated_cost_usd": _difference(_complete(before, "recorded_estimated_cost_usd", "model_calls_missing_cost"),
                                           _complete(after, "recorded_estimated_cost_usd", "model_calls_missing_cost")),
        "calls": calls,
    }
