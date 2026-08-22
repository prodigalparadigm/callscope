"""Report generation: the JSON contract and the derived text/HTML views."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from callscope.paralinguistics import analyze_paralinguistics
from callscope.report import render_html, render_json, render_text, write_reports
from callscope.rubric import parse_rubric
from callscope.schema import CallReport, SpeakerTurn, Transcript, TranscriptSegment
from callscope.scoring import JudgeContext, score_transcript


@pytest.fixture
def report(call_audio) -> CallReport:
    turns = [
        SpeakerTurn(0.0, 6.0, "SPEAKER_00"),
        SpeakerTurn(6.5, 10.0, "SPEAKER_01"),
        SpeakerTurn(14.0, 20.0, "SPEAKER_00"),
    ]
    transcript = Transcript(
        segments=[
            TranscriptSegment(0.0, 6.0, "Thanks for calling, this is Alex.", "SPEAKER_00"),
            TranscriptSegment(6.5, 10.0, "My order has not arrived.", "SPEAKER_01"),
            TranscriptSegment(14.0, 20.0, "I will reship it at no charge. Goodbye.",
                              "SPEAKER_00"),
        ],
        backend="fixture",
        model="test-model",
        language="en",
    )
    rubric = parse_rubric({
        "id": "r", "name": "Test rubric",
        "criteria": [
            {"id": "greeting", "name": "Greeting", "patterns": [r"thanks for calling"]},
            {"id": "resolution", "name": "Resolution", "patterns": [r"no charge"],
             "weight": 2.0},
            {"id": "missing", "name": "Never satisfied", "patterns": [r"impossible"]},
        ],
    })
    semantic = score_transcript(
        rubric, JudgeContext(transcript, turns, call_audio.duration, "unit-test")
    )
    return CallReport(
        call_id="unit-test",
        source_path="/tmp/unit-test.wav",
        duration_seconds=call_audio.duration,
        semantic=semantic,
        paralinguistics=analyze_paralinguistics(call_audio, turns, transcript=transcript),
        transcript=transcript,
        turns=turns,
        metadata={"diarization_backend": "cluster", "rubric_id": "r"},
        warnings=["a sample warning"],
    )


# --- JSON: the contract ----------------------------------------------------

def test_json_is_valid_and_round_trips(report: CallReport):
    parsed = json.loads(render_json(report))
    assert parsed["call_id"] == "unit-test"
    assert parsed["semantic"]["rubric_id"] == "r"
    assert len(parsed["semantic"]["criteria"]) == 3


def test_json_exposes_computed_scores(report: CallReport):
    """A consumer must not have to re-derive the rollup from the criteria."""
    parsed = json.loads(render_json(report))
    assert "score" in parsed["semantic"]
    assert "score_percent" in parsed["semantic"]
    assert 0.0 <= parsed["semantic"]["score"] <= 1.0
    assert parsed["semantic"]["criteria"][0]["weighted_score"] >= 0.0


def test_json_carries_evidence_timestamps(report: CallReport):
    parsed = json.loads(render_json(report))
    greeting = next(c for c in parsed["semantic"]["criteria"] if c["criterion_id"] == "greeting")
    assert greeting["evidence"]
    assert greeting["evidence"][0]["start"] == pytest.approx(0.0)
    assert greeting["evidence"][0]["speaker"] == "SPEAKER_00"


def test_json_carries_the_full_paralinguistic_profile(report: CallReport):
    profile = json.loads(render_json(report))["paralinguistics"]
    for key in (
        "duration_seconds", "speech_seconds", "silence_seconds", "silence_ratio",
        "longest_silence_seconds", "silence_p50_seconds", "silence_p90_seconds",
        "dead_air_events", "overlap_seconds", "overlap_events", "interruptions",
        "mean_response_latency_seconds", "speakers",
    ):
        assert key in profile, f"missing paralinguistic field {key}"
    assert len(profile["speakers"]) == 2
    for key in ("talk_time_ratio", "syllable_rate_hz", "f0_mean_hz", "f0_std_semitones"):
        assert key in profile["speakers"][0]


def test_json_records_provenance(report: CallReport):
    parsed = json.loads(render_json(report))
    assert parsed["metadata"]["diarization_backend"] == "cluster"
    assert parsed["transcript"]["backend"] == "fixture"
    assert parsed["transcript"]["model"] == "test-model"


def test_json_has_no_nan_or_infinity(report: CallReport):
    """NaN is not valid JSON; a consumer's parser will reject the whole document."""
    raw = render_json(report)
    assert "NaN" not in raw and "Infinity" not in raw
    json.loads(raw)  # strict by default in Python, but be explicit


# --- text view -------------------------------------------------------------

def test_text_report_shows_both_tracks(report: CallReport):
    text = render_text(report)
    assert "SEMANTIC TRACK" in text
    assert "PARALINGUISTIC TRACK" in text


def test_text_report_marks_pass_and_fail(report: CallReport):
    text = render_text(report)
    assert "[PASS] Greeting" in text
    assert "[FAIL] Never satisfied" in text


def test_text_report_includes_evidence_with_timestamps(report: CallReport):
    assert "@0:00 SPEAKER_00:" in render_text(report)


def test_text_report_surfaces_warnings(report: CallReport):
    text = render_text(report)
    assert "WARNINGS" in text and "a sample warning" in text


def test_text_report_survives_an_empty_transcript(call_audio):
    rubric = parse_rubric({"id": "r", "criteria": [{"id": "a", "patterns": ["x"]}]})
    empty = Transcript()
    semantic = score_transcript(rubric, JudgeContext(empty, [], 5.0, "empty"))
    report = CallReport(
        call_id="empty", source_path="x.wav", duration_seconds=5.0, semantic=semantic,
        paralinguistics=analyze_paralinguistics(call_audio, []),
        transcript=empty, turns=[],
    )
    text = render_text(report)
    assert "no voiced frames" in text
    assert "0.0%" in text


# --- HTML view -------------------------------------------------------------

def test_html_is_self_contained(report: CallReport):
    """No external assets means the report opens on an air-gapped machine."""
    html = render_html(report)
    assert html.lstrip().startswith("<!doctype html>")
    assert "src=\"http" not in html
    assert "href=\"http" not in html
    assert "<script" not in html


def test_html_escapes_content(report: CallReport):
    report.call_id = "<script>alert(1)</script>"
    html = render_html(report)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_html_contains_both_tracks(report: CallReport):
    html = render_html(report)
    assert "Semantic track" in html
    assert "Paralinguistic track" in html
    assert "SPEAKER_00" in html and "SPEAKER_01" in html


def test_html_footer_states_the_locality_claim(report: CallReport):
    assert "left this" in render_html(report)


def test_html_footer_qualifies_the_claim_when_an_llm_judge_ran(report: CallReport):
    report.semantic.judge_backend = "llm"
    assert "sent to the configured LLM judge" in render_html(report)


# --- writing ---------------------------------------------------------------

def test_write_reports_produces_all_three(report: CallReport, tmp_path: Path):
    written = write_reports(report, tmp_path)
    assert set(written) == {"json", "txt", "html"}
    for path in written.values():
        assert path.exists() and path.stat().st_size > 0


def test_write_reports_honours_a_format_subset(report: CallReport, tmp_path: Path):
    written = write_reports(report, tmp_path, formats=("json",))
    assert set(written) == {"json"}
    assert not (tmp_path / "unit-test.html").exists()


def test_unknown_format_rejected(report: CallReport, tmp_path: Path):
    with pytest.raises(ValueError, match="unknown report format"):
        write_reports(report, tmp_path, formats=("pdf",))


def test_call_ids_are_made_filesystem_safe(report: CallReport, tmp_path: Path):
    report.call_id = "../../etc/passwd"
    written = write_reports(report, tmp_path, formats=("json",))
    assert written["json"].parent == tmp_path
    assert "/" not in written["json"].name


def test_output_directory_is_created(report: CallReport, tmp_path: Path):
    target = tmp_path / "deep" / "nested"
    write_reports(report, target, formats=("json",))
    assert target.exists()


def test_all_three_views_agree_on_the_score(report: CallReport):
    """Text and HTML are derived views; they must never disagree with the JSON."""
    expected = f"{report.semantic.score_percent:.1f}%"
    assert expected in render_text(report)
    assert expected in render_html(report)
    assert json.loads(render_json(report))["semantic"]["score_percent"] == pytest.approx(
        report.semantic.score_percent
    )
