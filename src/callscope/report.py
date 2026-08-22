"""Report rendering: JSON for machines, text for terminals, HTML for humans.

The JSON document is the contract; the text and HTML views are strictly derived
from the same :class:`~callscope.schema.CallReport` so the three can never
disagree about a score.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from callscope.schema import CallReport, CriterionResult, SemanticResult

#: The report formats :func:`write_reports` knows how to render. The CLI
#: validates ``--formats`` against this, so the two cannot drift apart.
REPORT_FORMATS: tuple[str, ...] = ("json", "txt", "html")


def render_json(report: CallReport, *, indent: int = 2) -> str:
    """Serialize the full report. This is the machine-readable contract."""
    return json.dumps(report.to_dict(), indent=indent, ensure_ascii=False, sort_keys=False)


def render_text(report: CallReport, *, width: int = 78) -> str:
    """A plain-text summary suitable for a terminal or an email body."""
    out: list[str] = []
    rule = "=" * width
    thin = "-" * width

    out.append(rule)
    out.append(f"callscope report: {report.call_id}")
    out.append(rule)
    out.append(f"source            {report.source_path}")
    out.append(f"duration          {_hms(report.duration_seconds)}")
    out.append(f"transcript        {report.transcript.backend}"
               + (f" / {report.transcript.model}" if report.transcript.model else ""))
    out.append(f"diarization       {report.metadata.get('diarization_backend', 'unknown')}")
    out.append("")

    sem = report.semantic
    out.append(thin)
    out.append(f"SEMANTIC TRACK -- {sem.rubric_name} (judge: {sem.judge_backend})")
    out.append(thin)
    out.append(f"overall           {sem.score_percent:.1f}%  "
               f"({_passed(sem)}/{len(sem.criteria)} criteria passed)")
    out.append("")
    for c in sem.criteria:
        mark = "PASS" if c.passed else "FAIL"
        out.append(f"  [{mark}] {c.name}  {c.score:g}/{c.max_score:g} (weight {c.weight:g})")
        out.append(f"         {c.rationale}")
        for ev in c.evidence[:2]:
            speaker = ev.speaker or "?"
            out.append(f"         @{_hms(ev.start)} {speaker}: \"{_ellipsize(ev.quote, 60)}\"")
    out.append("")

    p = report.paralinguistics
    out.append(thin)
    out.append("PARALINGUISTIC TRACK -- computed from the audio signal")
    out.append(thin)
    out.append(f"speech / silence  {_hms(p.speech_seconds)} / {_hms(p.silence_seconds)}"
               f"  ({p.silence_ratio * 100:.1f}% silent)")
    out.append(f"longest silence   {p.longest_silence_seconds:.2f}s"
               f"   (p50 {p.silence_p50_seconds:.2f}s, p90 {p.silence_p90_seconds:.2f}s)")
    out.append(f"dead air events   {len(p.dead_air_events)}")
    out.append(f"overlap           {p.overlap_seconds:.2f}s across {len(p.overlap_events)} events")
    out.append("interruptions     "
               + ", ".join(f"{k} {v}" for k, v in sorted(p.interruptions.items())))
    latency = (f"{p.mean_response_latency_seconds:.2f}s"
               if p.mean_response_latency_seconds is not None else "n/a")
    out.append(f"mean response lag {latency}")
    out.append("")
    for sp in p.speakers:
        out.append(f"  {sp.speaker}")
        out.append(f"    talk time     {_hms(sp.talk_time_seconds)}"
                   f"  ({sp.talk_time_ratio * 100:.1f}% of speech)")
        out.append(f"    turns         {sp.turn_count}"
                   f"  (mean {sp.mean_turn_seconds:.2f}s, longest {sp.longest_turn_seconds:.2f}s)")
        out.append(f"    speech rate   {sp.syllable_rate_hz:.2f} syll/s"
                   + (f", {sp.words_per_minute:.0f} wpm" if sp.words_per_minute else ""))
        if sp.f0_mean_hz is not None:
            out.append(f"    pitch         {sp.f0_mean_hz:.1f} Hz"
                       f"  (sd {sp.f0_std_hz:.1f} Hz / {sp.f0_std_semitones:.2f} st)")
        else:
            out.append("    pitch         no voiced frames")

    if report.warnings:
        out.append("")
        out.append(thin)
        out.append("WARNINGS")
        out.append(thin)
        for w in report.warnings:
            out.append(f"  - {w}")

    out.append(rule)
    return "\n".join(out)


def render_html(report: CallReport) -> str:
    """A single self-contained HTML page. No external assets, no network."""
    sem = report.semantic
    p = report.paralinguistics
    e = html.escape

    criteria_rows = "\n".join(
        f"""      <tr class="{'pass' if c.passed else 'fail'}">
        <td>{e(c.name)}</td>
        <td class="num">{c.score:g} / {c.max_score:g}</td>
        <td class="num">{c.weight:g}</td>
        <td>{e(c.rationale)}</td>
        <td>{_evidence_html(c)}</td>
      </tr>"""
        for c in sem.criteria
    )

    speaker_rows = "\n".join(
        f"""      <tr>
        <td>{e(sp.speaker)}</td>
        <td class="num">{sp.talk_time_seconds:.1f}s</td>
        <td class="num">{sp.talk_time_ratio * 100:.1f}%</td>
        <td class="num">{sp.turn_count}</td>
        <td class="num">{sp.syllable_rate_hz:.2f}</td>
        <td class="num">{'-' if sp.words_per_minute is None else f'{sp.words_per_minute:.0f}'}</td>
        <td class="num">{'-' if sp.f0_mean_hz is None else f'{sp.f0_mean_hz:.1f}'}</td>
        <td class="num">{'-' if sp.f0_std_semitones is None else f'{sp.f0_std_semitones:.2f}'}</td>
      </tr>"""
        for sp in p.speakers
    )

    warnings_html = ""
    if report.warnings:
        items = "\n".join(f"<li>{e(w)}</li>" for w in report.warnings)
        warnings_html = f'<section><h2>Warnings</h2><ul class="warn">{items}</ul></section>'

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>callscope — {e(report.call_id)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, sans-serif;
         margin: 0 auto; max-width: 62rem; padding: 2rem 1rem; }}
  h1 {{ font-size: 1.4rem; margin: 0 0 .25rem; }}
  h2 {{ font-size: 1.05rem; margin: 2rem 0 .5rem;
        border-bottom: 1px solid currentColor; padding-bottom: .25rem; opacity: .85; }}
  .sub {{ opacity: .7; font-size: .875rem; margin: 0 0 1.5rem; }}
  .score {{ font-size: 2.5rem; font-weight: 600; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .875rem; }}
  th, td {{ text-align: left; padding: .5rem .6rem; vertical-align: top;
            border-bottom: 1px solid rgba(128,128,128,.3); }}
  th {{ font-weight: 600; opacity: .75; font-size: .8rem;
        text-transform: uppercase; letter-spacing: .04em; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
  tr.pass td:first-child {{ border-left: 3px solid #2f855a; }}
  tr.fail td:first-child {{ border-left: 3px solid #c53030; }}
  dl.metrics {{ display: grid; grid-template-columns: auto 1fr; gap: .3rem 1rem;
                font-size: .875rem; margin: 0; }}
  dl.metrics dt {{ opacity: .7; }}
  dl.metrics dd {{ margin: 0; font-variant-numeric: tabular-nums; }}
  .ev {{ font-size: .8rem; opacity: .8; margin: 0; padding-left: 1rem; }}
  ul.warn {{ font-size: .875rem; }}
  footer {{ margin-top: 3rem; font-size: .8rem; opacity: .6; }}
  .wrap {{ overflow-x: auto; }}
</style>
</head>
<body>
<h1>callscope — {e(report.call_id)}</h1>
<p class="sub">{e(report.source_path)} · {_hms(report.duration_seconds)} ·
transcript: {e(report.transcript.backend)} ·
diarization: {e(str(report.metadata.get('diarization_backend', 'unknown')))}</p>

<section>
  <h2>Semantic track — {e(sem.rubric_name)}</h2>
  <p class="score">{sem.score_percent:.1f}%</p>
  <p class="sub">{_passed(sem)} of {len(sem.criteria)} criteria passed ·
     judge: {e(sem.judge_backend)}</p>
  <div class="wrap"><table>
    <thead><tr><th>Criterion</th><th>Score</th><th>Weight</th>
      <th>Rationale</th><th>Evidence</th></tr></thead>
    <tbody>
{criteria_rows}
    </tbody>
  </table></div>
</section>

<section>
  <h2>Paralinguistic track — computed from the audio</h2>
  <dl class="metrics">
    <dt>Speech</dt><dd>{p.speech_seconds:.1f}s</dd>
    <dt>Silence</dt><dd>{p.silence_seconds:.1f}s ({p.silence_ratio * 100:.1f}%)</dd>
    <dt>Longest silence</dt><dd>{p.longest_silence_seconds:.2f}s</dd>
    <dt>Silence p50 / p90</dt>
      <dd>{p.silence_p50_seconds:.2f}s / {p.silence_p90_seconds:.2f}s</dd>
    <dt>Dead air events</dt><dd>{len(p.dead_air_events)}</dd>
    <dt>Overlap</dt><dd>{p.overlap_seconds:.2f}s over {len(p.overlap_events)} events</dd>
    <dt>Interruptions</dt>
      <dd>{e(', '.join(f'{k}: {v}' for k, v in sorted(p.interruptions.items())))}</dd>
    <dt>Mean response lag</dt><dd>{
      'n/a' if p.mean_response_latency_seconds is None
      else f'{p.mean_response_latency_seconds:.2f}s'}</dd>
  </dl>
  <div class="wrap"><table>
    <thead><tr><th>Speaker</th><th>Talk time</th><th>Share</th><th>Turns</th>
      <th>Syll/s</th><th>WPM</th><th>F0 Hz</th><th>F0 sd (st)</th></tr></thead>
    <tbody>
{speaker_rows}
    </tbody>
  </table></div>
</section>

{warnings_html}

<footer>{_footer_note(sem)}</footer>
</body>
</html>
"""


def write_reports(
    report: CallReport,
    out_dir: str | Path,
    *,
    formats: tuple[str, ...] = ("json", "txt", "html"),
) -> dict[str, Path]:
    """Write the selected report formats into ``out_dir``.

    Returns a mapping of format name to written path.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(report.call_id)
    renderers = _RENDERERS

    written: dict[str, Path] = {}
    for fmt in formats:
        renderer = renderers.get(fmt)
        if renderer is None:
            raise ValueError(f"unknown report format {fmt!r}; choose from {sorted(renderers)}")
        path = directory / f"{stem}.{fmt}"
        path.write_text(renderer(report), encoding="utf-8")
        written[fmt] = path
    return written


#: Bound to REPORT_FORMATS at import time so the CLI's advertised choices and
#: the renderers that actually exist cannot drift apart silently.
_RENDERERS = {"json": render_json, "txt": render_text, "html": render_html}
if set(_RENDERERS) != set(REPORT_FORMATS):  # pragma: no cover - import-time invariant
    raise RuntimeError("REPORT_FORMATS and the renderer table disagree")


def _evidence_html(criterion: CriterionResult) -> str:
    if not criterion.evidence:
        return "<span style='opacity:.5'>—</span>"
    parts = [
        f"<p class='ev'>@{_hms(ev.start)} "
        f"{html.escape(ev.speaker or '?')}: “{html.escape(_ellipsize(ev.quote, 90))}”</p>"
        for ev in criterion.evidence[:3]
    ]
    return "\n".join(parts)


def _footer_note(sem: SemanticResult) -> str:
    """The locality claim, qualified when the LLM judge actually ran."""
    base = "Generated locally by callscope. No audio, transcript, or metric left this machine"
    if sem.judge_backend == "llm":
        return base + " except the transcript excerpts sent to the configured LLM judge."
    return base + "."


def _passed(sem: SemanticResult) -> int:
    return sum(1 for c in sem.criteria if c.passed)


def _hms(seconds: float) -> str:
    """Format seconds as ``M:SS`` (or ``H:MM:SS`` past an hour)."""
    total = int(round(max(seconds, 0.0)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _ellipsize(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _safe_stem(call_id: str) -> str:
    """Filesystem-safe stem. Call ids often come from filenames or CRM keys."""
    keep = [c if (c.isalnum() or c in "-_.") else "_" for c in call_id]
    stem = "".join(keep).strip("._") or "call"
    return stem[:120]
