"""Summaries of recorded data, without inferring missing usage or prices."""


def summarize(trace: dict, *, slowest: int = 5) -> dict:
    """Rank inclusive span durations. Never add them to estimate wall time."""
    if type(slowest) is not int or not 1 <= slowest <= 100:
        raise ValueError("slowest must be between 1 and 100")
    models = [span for span in trace["spans"] if span["kind"] == "model"]
    with_usage = [span for span in models if span["input_tokens"] is not None and span["output_tokens"] is not None]
    with_cost = [span for span in models if span["estimated_cost_usd"] is not None]
    return {
        "trace_id": trace["id"],
        "name": trace["name"],
        "status": trace["status"],
        "error_type": trace["error_type"],
        "duration_ms": trace["duration_ms"],
        "span_count": len(trace["spans"]),
        "model_calls": len(models),
        "model_calls_with_usage": len(with_usage),
        "model_calls_missing_usage": len(models) - len(with_usage),
        "recorded_input_tokens": sum(span["input_tokens"] for span in with_usage) if with_usage else None,
        "recorded_output_tokens": sum(span["output_tokens"] for span in with_usage) if with_usage else None,
        "model_calls_with_cost": len(with_cost),
        "model_calls_missing_cost": len(models) - len(with_cost),
        "recorded_estimated_cost_usd": sum(span["estimated_cost_usd"] for span in with_cost) if with_cost else None,
        "failed_spans": [
            {key: span[key] for key in ("id", "name", "kind", "error_type")}
            for span in trace["spans"] if span["status"] == "error"
        ],
        "slowest_spans": [
            {key: span[key] for key in ("id", "name", "kind", "duration_ms", "status")}
            for span in sorted(trace["spans"], key=lambda span: span["duration_ms"], reverse=True)[:slowest]
        ],
    }
