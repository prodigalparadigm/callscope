"""Diarization: the two-speaker constraint, correctness, and determinism."""

from __future__ import annotations

import numpy as np
import pytest

from callscope.audio import AudioBuffer
from callscope.diarize import (
    DiarizationConfig,
    diarize,
    pyannote_available,
)
from callscope.errors import DiarizationError
from callscope.fixtures import CallScript, Utterance, synthesize_call
from callscope.schema import SPEAKER_LABELS
from callscope.vad import VadConfig, detect_speech, split_long_segments


def _truth_at(script: CallScript, start: float, end: float) -> set[str]:
    return {u.speaker for u in script.utterances if u.start < end and u.end > start}


# --- the constraint itself -------------------------------------------------


def test_never_emits_more_than_two_labels(call_audio, script):
    result = diarize(call_audio)
    labels = {t.speaker for t in result.turns}
    assert labels <= set(SPEAKER_LABELS)
    assert len(labels) <= 2


def test_two_speaker_call_is_split_into_two(call_audio):
    result = diarize(call_audio)
    assert result.n_speakers == 2
    assert {t.speaker for t in result.turns} == set(SPEAKER_LABELS)


def test_unambiguous_segments_are_attributed_correctly(call_audio, script):
    """Every turn covering exactly one ground-truth speaker must name that speaker."""
    result = diarize(call_audio)
    checked = 0
    for turn in result.turns:
        truth = _truth_at(script, turn.start, turn.end)
        if len(truth) != 1:
            continue  # overlap regions are genuinely ambiguous; excluded on purpose
        checked += 1
        assert turn.speaker == truth.pop(), f"turn {turn.start:.2f}-{turn.end:.2f} mislabeled"
    assert checked >= 4, "fixture should yield several unambiguous turns"


def test_speaker_00_is_whoever_spoke_first(call_audio, script):
    result = diarize(call_audio)
    first = min(result.turns, key=lambda t: t.start)
    assert first.speaker == SPEAKER_LABELS[0]
    assert script.utterances[0].speaker == SPEAKER_LABELS[0]


# --- the single-speaker guard ---------------------------------------------


def test_monologue_is_not_forced_into_two_speakers(single_speaker_audio):
    """2-means will always produce two clusters; the guard must reject them."""
    audio, _ = single_speaker_audio
    result = diarize(audio)
    assert result.n_speakers == 1
    assert {t.speaker for t in result.turns} == {SPEAKER_LABELS[0]}
    assert result.warnings, "a folded-down split must be reported, not silent"


def test_long_monologue_falls_below_the_silhouette_threshold():
    script = CallScript(
        duration=20.0,
        utterances=tuple(
            Utterance(s, s + 2.5, "SPEAKER_00", f0_hz=115.0)
            for s in (0.5, 3.5, 6.5, 9.5, 12.5, 15.5)
        ),
    )
    result = diarize(synthesize_call(script))
    assert result.n_speakers == 1
    assert result.separation < DiarizationConfig().min_separation


def test_similar_pitched_speakers_still_separate():
    """The hard case: 120 Hz vs 145 Hz, a quarter of the fixture's spread."""
    script = CallScript(
        duration=16.0,
        utterances=(
            Utterance(0.5, 3.5, "SPEAKER_00", f0_hz=120.0),
            Utterance(4.0, 7.0, "SPEAKER_01", f0_hz=145.0),
            Utterance(7.5, 11.0, "SPEAKER_00", f0_hz=120.0),
            Utterance(11.5, 15.5, "SPEAKER_01", f0_hz=145.0),
        ),
    )
    result = diarize(synthesize_call(script))
    assert result.n_speakers == 2
    for turn in result.turns:
        truth = _truth_at(script, turn.start, turn.end)
        if len(truth) == 1:
            assert turn.speaker == truth.pop()


# --- degenerate input ------------------------------------------------------


def test_silence_yields_no_turns():
    silent = AudioBuffer(np.zeros(16_000 * 3, dtype=np.float32), 16_000)
    result = diarize(silent)
    assert result.turns == []
    assert result.n_speakers == 0
    assert "no speech detected" in " ".join(result.warnings)


def test_very_short_audio_does_not_crash():
    tiny = AudioBuffer(np.zeros(160, dtype=np.float32), 16_000)  # 10 ms
    result = diarize(tiny)
    assert {t.speaker for t in result.turns} <= set(SPEAKER_LABELS)


# --- reproducibility -------------------------------------------------------


def test_diarization_is_deterministic(call_audio):
    """Same audio, same turns. A QA score that drifts between runs is worthless."""
    first = diarize(call_audio).turns
    second = diarize(call_audio).turns
    assert first == second


# --- turn hygiene ----------------------------------------------------------


def test_turns_are_sorted_and_non_overlapping(call_audio):
    turns = diarize(call_audio).turns
    for prev, nxt in zip(turns, turns[1:], strict=False):
        assert nxt.start >= prev.start
        assert nxt.start >= prev.end - 1e-6, "built-in diarizer must not emit overlaps"


def test_turns_stay_within_the_recording(call_audio):
    for turn in diarize(call_audio).turns:
        assert turn.start >= 0.0
        assert turn.end <= call_audio.duration + 1e-6
        assert turn.duration > 0.0


# --- backend selection -----------------------------------------------------


def test_explicit_cluster_backend(call_audio):
    assert diarize(call_audio, backend="cluster").backend == "cluster"


def test_requesting_pyannote_without_it_fails_loudly(call_audio, call_wav):
    available, _ = pyannote_available()
    if available:
        pytest.skip("pyannote is installed in this environment")
    with pytest.raises(DiarizationError, match="unavailable"):
        diarize(call_audio, backend="pyannote", wav_path=call_wav)


def test_auto_backend_falls_back_silently(call_audio, call_wav):
    """auto must degrade to clustering, never raise, when pyannote is absent."""
    result = diarize(call_audio, backend="auto", wav_path=call_wav)
    assert result.backend in {"cluster", "pyannote"}


def test_unknown_backend_rejected(call_audio):
    with pytest.raises(ValueError, match="unknown diarization backend"):
        diarize(call_audio, backend="magic")


def test_pyannote_available_never_raises():
    ok, reason = pyannote_available()
    assert isinstance(ok, bool) and isinstance(reason, str)


# --- VAD building blocks ---------------------------------------------------


def test_vad_finds_the_scripted_utterances(call_audio, script):
    segments = detect_speech(call_audio.samples, call_audio.sample_rate)
    assert segments, "VAD found no speech in a call that is mostly speech"
    for utt in script.utterances:
        mid = (utt.start + utt.end) / 2.0
        assert any(s.start <= mid <= s.end for s in segments), f"missed {utt.start}-{utt.end}"


def test_vad_excludes_the_scripted_dead_air(call_audio):
    """13.0-16.5 s is silent by construction and must not be called speech."""
    segments = detect_speech(call_audio.samples, call_audio.sample_rate)
    for point in (13.6, 14.5, 15.5, 16.2):
        assert not any(s.start <= point <= s.end for s in segments), f"{point}s misdetected"


def test_vad_on_pure_noise_finds_nothing():
    rng = np.random.default_rng(7)
    noise = (rng.normal(0, 10 ** (-55 / 20), 16_000 * 4)).astype(np.float32)
    assert detect_speech(noise, 16_000, VadConfig()) == []


def test_split_long_segments_respects_the_cap(call_audio):
    raw = detect_speech(call_audio.samples, call_audio.sample_rate)
    split = split_long_segments(raw, max_seconds=1.0, target_seconds=0.75)
    assert all(s.duration <= 1.0 + 1e-6 for s in split)
    # Splitting must preserve total speech time, not discard any.
    assert sum(s.duration for s in split) == pytest.approx(sum(s.duration for s in raw))
