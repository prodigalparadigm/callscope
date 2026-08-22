"""Backend selection, transcript parsing, and speaker attribution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from callscope.errors import TranscriptionError
from callscope.schema import SpeakerTurn, Transcript, TranscriptSegment
from callscope.transcribe import (
    BACKEND_PREFERENCE,
    TranscriptionConfig,
    attribute_speakers,
    available_backends,
    select_backend,
    transcribe,
    transcript_from_dict,
)

# --- backend selection -----------------------------------------------------

def test_preference_order_is_documented_and_stable():
    """Changing this order changes which model a reviewer's machine picks."""
    assert BACKEND_PREFERENCE == ("mlx", "faster", "openai", "fixture", "null")


def test_fixture_and_null_are_always_available():
    available = available_backends()
    assert "fixture" in available and "null" in available


def test_available_backends_respect_the_preference_order():
    available = available_backends()
    positions = [BACKEND_PREFERENCE.index(name) for name in available]
    assert positions == sorted(positions)


def test_auto_never_raises_even_with_nothing_installed():
    assert select_backend("auto") in BACKEND_PREFERENCE


def test_auto_prefers_a_supplied_transcript_over_running_a_model(tmp_path: Path):
    cfg = TranscriptionConfig(fixture_path=tmp_path / "t.json")
    assert select_backend("auto", config=cfg) == "fixture"


def test_naming_an_uninstalled_backend_fails_with_an_install_hint():
    import importlib.util

    if importlib.util.find_spec("mlx_whisper") is not None:
        pytest.skip("mlx-whisper is installed in this environment")
    with pytest.raises(TranscriptionError, match=r"pip install 'callscope\[mlx\]'"):
        select_backend("mlx")


def test_unknown_backend_rejected():
    with pytest.raises(TranscriptionError, match="unknown transcription backend"):
        select_backend("telepathy")


# --- the null backend degrades rather than failing -------------------------

def test_null_backend_returns_an_empty_transcript_with_a_warning(call_wav: Path):
    transcript, warnings = transcribe(call_wav, TranscriptionConfig(backend="null"))
    assert transcript.segments == []
    assert transcript.backend == "null"
    assert any("No local Whisper backend" in w for w in warnings)


def test_missing_audio_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        transcribe(tmp_path / "absent.wav", TranscriptionConfig(backend="null"))


# --- the fixture backend ---------------------------------------------------

def test_fixture_backend_loads_a_transcript(call_wav: Path, transcript_json: Path):
    transcript, warnings = transcribe(
        call_wav, TranscriptionConfig(backend="fixture", fixture_path=transcript_json)
    )
    assert transcript.segments
    assert transcript.backend == "fixture"
    assert transcript.language == "en"
    assert warnings == []


def test_fixture_backend_without_a_path_explains_itself(call_wav: Path):
    with pytest.raises(TranscriptionError, match="--transcript"):
        transcribe(call_wav, TranscriptionConfig(backend="fixture"))


def test_fixture_backend_on_a_missing_file(call_wav: Path, tmp_path: Path):
    with pytest.raises(TranscriptionError, match="not found"):
        transcribe(call_wav, TranscriptionConfig(
            backend="fixture", fixture_path=tmp_path / "nope.json"))


def test_fixture_backend_on_invalid_json(call_wav: Path, tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(TranscriptionError, match="not valid JSON"):
        transcribe(call_wav, TranscriptionConfig(backend="fixture", fixture_path=bad))


# --- transcript parsing ----------------------------------------------------

def test_whisper_shaped_document_parses():
    transcript = transcript_from_dict({
        "language": "en",
        "segments": [{"start": 0.0, "end": 1.0, "text": " hello "}],
    })
    assert transcript.language == "en"
    assert transcript.segments[0].text == "hello"


def test_bare_segment_list_parses():
    transcript = transcript_from_dict([{"start": 0.0, "end": 1.0, "text": "hi"}])
    assert len(transcript.segments) == 1


def test_malformed_segments_are_skipped_not_fatal():
    """One bad timestamp in a 40-minute call must not lose the other 39 minutes."""
    transcript = transcript_from_dict({"segments": [
        {"start": 0.0, "end": 1.0, "text": "good"},
        {"start": "x", "end": 2.0, "text": "bad start"},
        {"end": 3.0, "text": "missing start"},
        {"start": 5.0, "end": 4.0, "text": "inverted"},
        {"start": 6.0, "end": 7.0, "text": "   "},
        "not even a mapping",
        {"start": 8.0, "end": 9.0, "text": "also good"},
    ]})
    assert [s.text for s in transcript.segments] == ["good", "also good"]


def test_segments_come_back_sorted():
    transcript = transcript_from_dict([
        {"start": 5.0, "end": 6.0, "text": "later"},
        {"start": 1.0, "end": 2.0, "text": "earlier"},
    ])
    assert [s.text for s in transcript.segments] == ["earlier", "later"]


def test_wrong_top_level_type_rejected():
    with pytest.raises(TranscriptionError, match="expected a transcript"):
        transcript_from_dict("just a string")


def test_non_list_segments_rejected():
    with pytest.raises(TranscriptionError, match="must be a list"):
        transcript_from_dict({"segments": {"start": 0}})


# --- speaker attribution ---------------------------------------------------

def test_segments_are_attributed_by_maximal_overlap():
    transcript = Transcript(segments=[
        TranscriptSegment(0.0, 2.0, "first"),
        TranscriptSegment(3.0, 5.0, "second"),
    ])
    turns = [
        SpeakerTurn(0.0, 2.5, "SPEAKER_00"),
        SpeakerTurn(2.5, 6.0, "SPEAKER_01"),
    ]
    attributed = attribute_speakers(transcript, turns)
    assert [s.speaker for s in attributed.segments] == ["SPEAKER_00", "SPEAKER_01"]


def test_straddling_segment_goes_to_the_dominant_speaker():
    transcript = Transcript(segments=[TranscriptSegment(0.0, 10.0, "long")])
    turns = [SpeakerTurn(0.0, 2.0, "SPEAKER_00"), SpeakerTurn(2.0, 10.0, "SPEAKER_01")]
    assert attribute_speakers(transcript, turns).segments[0].speaker == "SPEAKER_01"


def test_orphan_segment_falls_back_to_the_nearest_turn():
    """Whisper routinely pads past the end of speech; those words still count."""
    transcript = Transcript(segments=[TranscriptSegment(50.0, 52.0, "trailing")])
    turns = [SpeakerTurn(0.0, 2.0, "SPEAKER_00"), SpeakerTurn(40.0, 45.0, "SPEAKER_01")]
    assert attribute_speakers(transcript, turns).segments[0].speaker == "SPEAKER_01"


def test_attribution_without_turns_is_a_no_op():
    transcript = Transcript(segments=[TranscriptSegment(0.0, 1.0, "x")])
    assert attribute_speakers(transcript, []).segments[0].speaker is None


def test_attribution_preserves_provenance():
    transcript = Transcript(
        segments=[TranscriptSegment(0.0, 1.0, "x")], backend="mlx", model="turbo",
        language="en",
    )
    out = attribute_speakers(transcript, [SpeakerTurn(0.0, 1.0, "SPEAKER_00")])
    assert (out.backend, out.model, out.language) == ("mlx", "turbo", "en")


# --- transcript helpers ----------------------------------------------------

def test_word_counts_per_speaker():
    transcript = Transcript(segments=[
        TranscriptSegment(0.0, 1.0, "one two three", "SPEAKER_00"),
        TranscriptSegment(1.0, 2.0, "four five", "SPEAKER_01"),
    ])
    assert transcript.word_count() == 5
    assert transcript.word_count("SPEAKER_00") == 3
    assert transcript.word_count("SPEAKER_01") == 2


def test_transcript_text_joins_segments():
    transcript = Transcript(segments=[
        TranscriptSegment(0.0, 1.0, "hello"),
        TranscriptSegment(1.0, 2.0, "world"),
    ])
    assert transcript.text == "hello world"


def test_json_transcripts_round_trip(tmp_path: Path, script):
    from callscope.fixtures import script_to_transcript_dict

    path = tmp_path / "t.json"
    path.write_text(json.dumps(script_to_transcript_dict(script)), encoding="utf-8")
    transcript = transcript_from_dict(json.loads(path.read_text()))
    assert len(transcript.segments) == len([u for u in script.utterances if u.text])
    assert all(s.speaker for s in transcript.segments)
