"""End-to-end pipeline and CLI. These are the tests that prove the thing runs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from callscope.cli import main
from callscope.pipeline import PipelineConfig, analyze_batch, analyze_call
from callscope.rubric import parse_rubric
from callscope.schema import SPEAKER_LABELS
from callscope.transcribe import TranscriptionConfig


@pytest.fixture
def config(support_rubric, transcript_json: Path) -> PipelineConfig:
    return PipelineConfig(
        rubric=support_rubric,
        transcription=TranscriptionConfig(backend="fixture", fixture_path=transcript_json),
    )


# --- the happy path --------------------------------------------------------

@pytest.mark.requires_ffmpeg
def test_full_pipeline_produces_a_complete_report(call_wav: Path, config: PipelineConfig):
    report = analyze_call(call_wav, config)

    assert report.duration_seconds == pytest.approx(20.0, abs=0.1)
    assert report.turns, "diarization produced no turns"
    assert {t.speaker for t in report.turns} <= set(SPEAKER_LABELS)
    assert report.transcript.segments, "transcript was not loaded"
    assert all(s.speaker for s in report.transcript.segments), "segments unattributed"

    assert report.semantic.criteria
    assert 0.0 <= report.semantic.score <= 1.0
    assert len(report.paralinguistics.speakers) == 2
    assert report.paralinguistics.speech_seconds > 0

    assert report.metadata["diarization_backend"] == "cluster"
    assert report.metadata["transcription_backend"] == "fixture"
    assert report.metadata["elapsed_seconds"] >= 0


@pytest.mark.requires_ffmpeg
def test_pipeline_scores_the_scripted_call_as_expected(call_wav: Path, config: PipelineConfig):
    """The scripted transcript satisfies most support criteria; assert the ones we control."""
    report = analyze_call(call_wav, config)
    by_id = {c.criterion_id: c for c in report.semantic.criteria}
    assert by_id["greeting"].passed, by_id["greeting"].rationale
    assert by_id["resolution"].passed, by_id["resolution"].rationale
    assert by_id["closing"].passed, by_id["closing"].rationale
    assert by_id["prohibited_language"].passed
    assert report.semantic.score > 0.5


@pytest.mark.requires_ffmpeg
def test_evidence_points_back_into_the_call(call_wav: Path, config: PipelineConfig):
    report = analyze_call(call_wav, config)
    for criterion in report.semantic.criteria:
        for evidence in criterion.evidence:
            assert 0.0 <= evidence.start < evidence.end <= report.duration_seconds + 1e-6


@pytest.mark.requires_ffmpeg
def test_the_two_tracks_are_independent(call_wav: Path, support_rubric):
    """Removing the transcript must not degrade the paralinguistic track at all."""
    with_transcript = analyze_call(call_wav, PipelineConfig(
        rubric=support_rubric,
        transcription=TranscriptionConfig(backend="null"),
    ))
    assert with_transcript.transcript.segments == []
    assert with_transcript.semantic.score == 0.0
    # ... and yet:
    profile = with_transcript.paralinguistics
    assert profile.speech_seconds > 0
    assert profile.speaker(SPEAKER_LABELS[0]).f0_mean_hz is not None
    assert profile.speaker(SPEAKER_LABELS[0]).syllable_rate_hz > 0
    assert any("transcript is empty" in w for w in with_transcript.warnings)


@pytest.mark.requires_ffmpeg
def test_pipeline_is_deterministic(call_wav: Path, config: PipelineConfig):
    first = analyze_call(call_wav, config)
    second = analyze_call(call_wav, config)
    assert first.semantic.score == second.semantic.score
    assert first.turns == second.turns
    assert first.paralinguistics.speech_seconds == second.paralinguistics.speech_seconds


def test_skip_normalization_accepts_canonical_wav(call_wav: Path, config: PipelineConfig):
    """Runs with no ffmpeg subprocess at all -- the input is already canonical."""
    config.skip_normalization = True
    report = analyze_call(call_wav, config)
    assert report.duration_seconds == pytest.approx(20.0, abs=0.01)


def test_call_id_defaults_to_the_filename(call_wav: Path, config: PipelineConfig):
    config.skip_normalization = True
    assert analyze_call(call_wav, config).call_id == call_wav.stem


def test_explicit_call_id_wins(call_wav: Path, config: PipelineConfig):
    config.skip_normalization = True
    config.call_id = "CRM-4471"
    assert analyze_call(call_wav, config).call_id == "CRM-4471"


# --- failure handling ------------------------------------------------------

def test_missing_rubric_is_rejected_up_front(call_wav: Path):
    with pytest.raises(ValueError, match="rubric is required"):
        analyze_call(call_wav, PipelineConfig())


def test_missing_input_raises(tmp_path: Path, config: PipelineConfig):
    with pytest.raises(FileNotFoundError):
        analyze_call(tmp_path / "absent.wav", config)


def test_batch_isolates_failures(call_wav: Path, tmp_path: Path, config: PipelineConfig):
    """One corrupt file in a nightly batch must not take down the run."""
    config.skip_normalization = True
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"RIFF____WAVEnope")

    reports, failures = analyze_batch([call_wav, broken, call_wav], config)
    assert len(reports) == 2
    assert len(failures) == 1
    assert failures[0][0] == str(broken)


def test_rubric_path_is_loaded_when_no_object_is_given(
    call_wav: Path, transcript_json: Path
):
    from callscope.cli import DEFAULT_RUBRIC

    report = analyze_call(call_wav, PipelineConfig(
        rubric_path=DEFAULT_RUBRIC,
        transcription=TranscriptionConfig(backend="fixture", fixture_path=transcript_json),
        skip_normalization=True,
    ))
    assert report.metadata["rubric_id"] == "support_call_v1"


def test_a_domain_agnostic_rubric_scores_the_same_call(
    call_wav: Path, transcript_json: Path
):
    """Same binary, same audio, entirely different criteria: nothing is hardcoded."""
    custom = parse_rubric({
        "id": "shipping_v1", "name": "Shipping-specific",
        "criteria": [
            {"id": "reship", "name": "Offered a reship", "patterns": [r"reship"]},
            {"id": "apology", "name": "Apologized", "patterns": [r"\bsorry\b|\bapolog"]},
        ],
    })
    report = analyze_call(call_wav, PipelineConfig(
        rubric=custom,
        transcription=TranscriptionConfig(backend="fixture", fixture_path=transcript_json),
        skip_normalization=True,
    ))
    ids = {c.criterion_id for c in report.semantic.criteria}
    assert ids == {"reship", "apology"}
    by_id = {c.criterion_id: c for c in report.semantic.criteria}
    assert by_id["reship"].passed          # the script does say "reshipping"
    assert not by_id["apology"].passed     # the script never apologizes


# --- CLI -------------------------------------------------------------------

def test_cli_doctor_reports_capabilities(capsys):
    """doctor must describe the machine it is on, ffmpeg present or not."""
    code = main(["doctor"])
    out = capsys.readouterr().out
    assert "whisper preference" in out
    assert "pyannote" in out
    # Exit 1 is the correct, documented answer when ffmpeg is missing.
    assert code == (0 if shutil.which("ffmpeg") else 1)


def test_cli_doctor_json_is_parseable(capsys):
    main(["doctor", "--json"])
    info = json.loads(capsys.readouterr().out)
    assert info["ffmpeg"] is bool(shutil.which("ffmpeg"))
    assert "fixture" in info["transcription_backends_available"]
    assert info["transcription_backend_selected"] in info["transcription_backends_available"]


def test_cli_doctor_exit_code_signals_a_missing_ffmpeg(monkeypatch, capsys):
    """A CI gate reads the exit code, not the prose."""
    monkeypatch.setattr("callscope.audio.shutil.which", lambda name: None)
    assert main(["doctor"]) == 1
    assert "MISSING (required)" in capsys.readouterr().out


# --- CLI usage errors ------------------------------------------------------

def test_cli_rejects_an_unknown_report_format(tmp_path: Path, capsys):
    """Caught before any audio is decoded, and as a message rather than a traceback."""
    code = main(["analyze", "x.wav", "--out", str(tmp_path), "--formats", "pdf,json"])
    assert code == 2
    assert "unknown report format" in capsys.readouterr().err


def test_cli_rejects_an_empty_format_list(tmp_path: Path, capsys):
    assert main(["analyze", "x.wav", "--out", str(tmp_path), "--formats", " , "]) == 2
    assert "--formats is empty" in capsys.readouterr().err


def test_cli_reports_an_unwritable_output_directory(
    tmp_path: Path, call_wav: Path, transcript_json: Path, capsys
):
    """A read-only report directory is a message and an exit code, not a traceback."""
    blocked = tmp_path / "readonly"
    blocked.mkdir()
    blocked.chmod(0o500)
    try:
        code = main([
            "analyze", str(call_wav), "--transcript", str(transcript_json),
            "--out", str(blocked), "--skip-normalization", "--quiet",
        ])
    finally:
        blocked.chmod(0o700)
    assert code == 1
    assert "could not write reports" in capsys.readouterr().err


def test_cli_rejects_a_transcript_and_a_model_backend_together(
    tmp_path: Path, transcript_json: Path, capsys
):
    """Silently ignoring --whisper would be worse than refusing the combination."""
    code = main([
        "analyze", "x.wav", "--out", str(tmp_path),
        "--transcript", str(transcript_json), "--whisper", "mlx",
    ])
    assert code == 2
    assert "conflict" in capsys.readouterr().err


@pytest.mark.requires_ffmpeg
def test_cli_demo_runs_end_to_end(tmp_path: Path, capsys):
    """The zero-argument path a reviewer will actually try first."""
    assert main(["demo", "--out", str(tmp_path)]) == 0

    out = capsys.readouterr().out
    assert "SEMANTIC TRACK" in out and "PARALINGUISTIC TRACK" in out

    for name in ("demo_call.json", "demo_call.txt", "demo_call.html"):
        assert (tmp_path / name).exists(), f"{name} was not written"

    payload = json.loads((tmp_path / "demo_call.json").read_text())
    assert payload["call_id"] == "demo_call"
    assert payload["paralinguistics"]["speakers"]


@pytest.mark.requires_ffmpeg
def test_cli_analyze_writes_reports(tmp_path: Path, call_wav: Path, transcript_json: Path):
    code = main([
        "analyze", str(call_wav),
        "--transcript", str(transcript_json),
        "--out", str(tmp_path),
        "--formats", "json",
        "--quiet",
    ])
    assert code == 0
    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text())["semantic"]["criteria"]


def test_cli_analyze_reports_a_bad_file_without_crashing(tmp_path: Path, capsys):
    missing = tmp_path / "nope.wav"
    assert main(["analyze", str(missing), "--out", str(tmp_path), "--quiet"]) == 1
    assert "nope.wav" in capsys.readouterr().err


def test_cli_rejects_an_unknown_backend(tmp_path: Path):
    with pytest.raises(SystemExit):
        main(["analyze", "x.wav", "--whisper", "telepathy"])


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "callscope" in capsys.readouterr().out


def test_transcription_failure_degrades_instead_of_ending_the_run(
    call_wav: Path, support_rubric, tmp_path: Path
):
    """A broken transcript file must still yield a full paralinguistic profile."""
    broken = tmp_path / "broken.json"
    broken.write_text("{ not json at all", encoding="utf-8")

    report = analyze_call(call_wav, PipelineConfig(
        rubric=support_rubric,
        transcription=TranscriptionConfig(backend="fixture", fixture_path=broken),
        skip_normalization=True,
    ))
    assert report.transcript.backend == "failed"
    assert report.semantic.score == 0.0
    assert any("transcription failed" in w for w in report.warnings)
    assert report.paralinguistics.speech_seconds > 0
    assert report.paralinguistics.speaker(SPEAKER_LABELS[0]).f0_mean_hz is not None
    assert report.turns


def test_report_metadata_records_the_diarization_diagnostic(
    call_wav: Path, config: PipelineConfig
):
    """`separation` is how a consumer knows whether to trust per-speaker numbers."""
    config.skip_normalization = True
    metadata = analyze_call(call_wav, config).metadata
    assert metadata["diarization_speakers"] == 2
    assert 0.0 <= metadata["diarization_separation"] <= 1.0


def test_cluster_diarizer_cannot_report_overlap(call_wav: Path, config: PipelineConfig):
    """Pins the limitation the README states: disjoint turns mean structural zeros.

    The built-in clusterer assigns each VAD segment to exactly one speaker, so
    the turns it emits cannot overlap and the overlap/interruption metrics are
    always zero -- however much cross-talk the audio actually contains. The
    fixture call has a deliberate 0.6 s overlap at 9.4-10.0 s. If this test ever
    fails, the diarizer learned to emit overlapping turns and the README's
    limitations section is out of date.
    """
    config.skip_normalization = True
    report = analyze_call(call_wav, config)
    assert report.paralinguistics.overlap_seconds == 0.0
    assert report.paralinguistics.overlap_events == []
    assert sum(report.paralinguistics.interruptions.values()) == 0
    for prev, nxt in zip(report.turns, report.turns[1:], strict=False):
        assert nxt.start >= prev.end - 1e-6
