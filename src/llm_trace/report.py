"""A self-contained, read-only HTML view of one saved trace."""

from html import escape

from .summary import summarize


def _text(value) -> str:
    return escape(str(value), quote=True)


def render_report(trace: dict) -> str:
    data = summarize(trace)
    spans = {span["id"]: span for span in trace["spans"]}
    longest = max((span["duration_ms"] for span in trace["spans"]), default=0) or 1
    rows = []
    for span in trace["spans"]:
        depth, parent, seen = 0, span["parent_id"], {span["id"]}
        while parent in spans and parent not in seen:
            seen.add(parent)
            depth += 1
            parent = spans[parent]["parent_id"]
        metadata = []
        if span["model"] is not None:
            metadata.append(f"Model: {_text(span['model'])}")
        if span["prompt_version"] is not None:
            metadata.append(f"Prompt: {_text(span['prompt_version'])}")
        if span["error_type"] is not None:
            metadata.append(f"Error: {_text(span['error_type'])}")
        if span.get("cached_input_tokens") is not None:
            metadata.append(f"Cached input: {_text(span['cached_input_tokens'])}")
        usage = "Not recorded" if span["input_tokens"] is None else f"{span['input_tokens']} in / {span['output_tokens']} out"
        cost = "Not recorded" if span["estimated_cost_usd"] is None else f"${span['estimated_cost_usd']:.6f}"
        state = "error" if span["status"] == "error" else "ok"
        width = max(0, min(100, span["duration_ms"] / longest * 100))
        rows.append(f"""<tr>
          <td><div class="call" style="padding-left:{min(depth, 20) * 16}px">
            <span class="kind">{_text(span['kind'])}</span><strong>{_text(span['name'])}</strong>
            <div class="metadata">{' &middot; '.join(metadata)}</div></div></td>
          <td><span class="badge {state}">{_text(span['status'])}</span></td>
          <td class="duration"><strong>{span['duration_ms']:.2f} ms</strong>
            <div class="track" aria-hidden="true"><div class="bar {state}" style="width:{width:.2f}%"></div></div></td>
          <td>{_text(usage)}<div class="metadata">{_text(cost)}</div></td>
        </tr>""")
    if not rows:
        rows.append('<tr><td colspan="4" class="empty">No spans recorded.</td></tr>')
    failures = "".join(
        f"<li><strong>{_text(span['name'])}</strong> <span class='metadata'>{_text(span['kind'])} / {_text(span['error_type'] or 'Unknown error')}</span></li>"
        for span in data["failed_spans"]
    ) or '<li class="metadata">No failed spans.</li>'
    tokens = "Not recorded" if data["recorded_input_tokens"] is None else f"{data['recorded_input_tokens']} in / {data['recorded_output_tokens']} out"
    cost = "Not recorded" if data["recorded_estimated_cost_usd"] is None else f"${data['recorded_estimated_cost_usd']:.6f}"
    failed_count = len(data["failed_spans"])
    failure_label = "failed span" if failed_count == 1 else "failed spans"
    error = f"<p class='trace-error'>Request error: {_text(data['error_type'])}</p>" if data["error_type"] else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>{_text(trace['name'])} | llm-trace</title>
<style>
* {{ box-sizing:border-box }}
body {{ margin:0; background:#f4f6f8; color:#182536; font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif }}
main {{ max-width:1120px; margin:0 auto; padding:42px 24px 60px }}
.eyebrow {{ color:#586a80; font-size:12px; letter-spacing:1.5px; text-transform:uppercase }}
h1 {{ font-size:32px; line-height:1.2; margin:12px 0; overflow-wrap:anywhere }}
h2 {{ font-size:18px; margin:0 0 16px }}
.metadata {{ color:#586a80; font-size:12px; overflow-wrap:anywhere }}
.ids {{ margin:0 0 28px; font-family:ui-monospace,monospace }}
.cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:24px }}
.card,section {{ background:white; border:1px solid #dce3ea; border-radius:10px }}
.card {{ padding:18px }}
.label {{ font-size:12px; color:#586a80; margin-bottom:7px }}
.value {{ font-size:22px; font-weight:650; overflow-wrap:anywhere }}
section {{ padding:22px; margin-top:20px }}
.scroll {{ overflow-x:auto }}
table {{ border-collapse:collapse; width:100%; min-width:670px; font-size:13px }}
th {{ text-align:left; color:#586a80; font-size:11px; text-transform:uppercase; letter-spacing:.6px; padding:0 12px 12px }}
td {{ padding:16px 12px; border-top:1px solid #e8edf2; vertical-align:top }}
th:first-child,td:first-child {{ padding-left:0; width:42% }}
.kind {{ color:#586a80; font-size:10px; text-transform:uppercase; display:block; margin-bottom:3px }}
.call strong {{ overflow-wrap:anywhere }}
.badge {{ display:inline-block; border-radius:5px; font-size:11px; padding:3px 8px }}
.badge.ok {{ color:#245d48; background:#e8f4ee }}
.badge.error {{ color:#9b3137; background:#fbecee }}
.duration {{ min-width:150px }}
.track {{ height:5px; background:#edf1f6; border-radius:5px; margin-top:9px }}
.bar {{ height:5px; border-radius:5px; background:#557fd7 }}
.bar.error {{ background:#c65360 }}
.note {{ color:#586a80; font-size:12px; margin:18px 0 0 }}
ul {{ list-style:none; padding:0; margin:0 }}
li+li {{ margin-top:10px }}
.trace-error {{ color:#9b3137 }}
.empty {{ color:#586a80; padding:28px 0 }}
footer {{ margin-top:22px; color:#586a80; font-size:12px }}
@media(max-width:700px) {{ main {{ padding:24px 16px }} .cards {{ grid-template-columns:repeat(2,1fr) }} h1 {{ font-size:26px }} section {{ padding:16px }} }}
</style></head><body><main>
<div class="eyebrow">llm-trace / local report</div><h1>{_text(trace['name'])}</h1>
<p class="metadata ids">{_text(trace['id'])}<br>{_text(trace['started_at'])}</p>{error}
<div class="cards">
  <div class="card"><div class="label">Request duration</div><div class="value">{trace['duration_ms']:.2f} ms</div><div class="metadata">Wall time</div></div>
  <div class="card"><div class="label">Request status</div><div class="value">{_text(trace['status'])}</div><div class="metadata">{failed_count} {failure_label}</div></div>
  <div class="card"><div class="label">Recorded tokens</div><div class="value">{_text(tokens)}</div><div class="metadata">Usage on {data['model_calls_with_usage']}/{data['model_calls']} model calls</div></div>
  <div class="card"><div class="label">Recorded estimated cost</div><div class="value">{_text(cost)}</div><div class="metadata">Cost on {data['model_calls_with_cost']}/{data['model_calls']} model calls</div></div>
</div>
<section><h2>Calls <span class="metadata">{len(spans)} recorded</span></h2>
<div class="scroll"><table><thead><tr><th scope="col">Call</th><th scope="col">Status</th><th scope="col">Duration</th><th scope="col">Tokens / cost</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<p class="note">Bars compare inclusive durations, not positions on a timeline. Parent spans include their children; parallel calls can overlap. Don't add these times together.</p></section>
<section><h2>Failed calls</h2><ul>{failures}</ul></section>
<footer>Partial token and cost totals cover recorded measurements only. This file has no scripts or external resources. Names and model metadata are included as recorded.</footer>
</main></body></html>"""
