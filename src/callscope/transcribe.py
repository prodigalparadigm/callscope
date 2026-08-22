"""Local Whisper transcription with runtime backend selection.

Preference order, best first:

1. ``mlx-whisper``   -- Apple Silicon. Runs on the GPU via MLX; fastest on a Mac.
2. ``faster-whisper``-- CTranslate2. Fast on CPU and CUDA, small memory footprint.
3. ``openai-whisper``-- the reference PyTorch implementation. Slowest, most portable.
4. ``fixture``       -- reads a sidecar JSON transcript. Used by the test suite and
   by anyone who already has a transcript and only wants the scoring stages.
5. ``null``          -- returns an empty transcript and a warning.

Detection happens at runtime rather than at install time, and an unavailable
backend degrades to the next one instead of hard-failing. A QA pipeline that
refuses to produce a paralinguistic profile because a transcription wheel is
missing is worse than one that produces the profile and says so.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from callscope.errors import TranscriptionError
from callscope.schema import Transcript, TranscriptSegment

logger = logging.getLogger(__name__)

#: Backends in descending order of preference. Documented in the module docstring
#: and in the README; keep the three in sync.
BACKEND_PREFERENCE: tuple[str, ...] = ("mlx", "faster", "openai", "fixture", "null")

_MODULE_FOR_BACKEND = {
    "mlx": "mlx_whisper",
    "faster": "faster_whisper",
    "openai": "whisper",
}

DEFAULT_MODELS = {
    "mlx": "mlx-community/whisper-large-v3-turbo",
    "faster": "base.en",
    "openai": "base.en",
}


@dataclass(frozen=True, slots=True)
class TranscriptionConfig:
    """Options passed through to whichever backend is selected."""

    backend: str = "auto"
    model: str | None = None
    language: str | None = None
    #: Path to a sidecar JSON transcript, used by the ``fixture`` backend.
    fixture_path: str | Path | None = None

    def resolved_model(self, backend: str) -> str | None:
        if self.model:
            return self.model
        env = os.environ.get("CALLSCOPE_WHISPER_MODEL")
        if env:
            return env
        return DEFAULT_MODELS.get(backend)


@runtime_checkable
class TranscriptionBackend(Protocol):
    """The contract every backend satisfies."""

    name: str

    def is_available(self) -> bool:
        """Whether this backend can run right now. Must never raise."""

    def transcribe(self, wav_path: Path, config: TranscriptionConfig) -> Transcript:
        """Transcribe a canonical 16 kHz mono WAV."""


def available_backends() -> list[str]:
    """Installed backends, best first. ``fixture`` and ``null`` are always present."""
    found = [
        name
        for name in BACKEND_PREFERENCE
        if name in _MODULE_FOR_BACKEND
        and importlib.util.find_spec(_MODULE_FOR_BACKEND[name]) is not None
    ]
    return found + ["fixture", "null"]


def select_backend(requested: str = "auto", *, config: TranscriptionConfig | None = None) -> str:
    """Resolve ``requested`` to a concrete backend name.

    Raises:
        TranscriptionError: a specific backend was named but is not installed.
    """
    cfg = config or TranscriptionConfig()
    if requested == "auto":
        if cfg.fixture_path:
            return "fixture"
        for name in BACKEND_PREFERENCE:
            if name in {"fixture", "null"}:
                continue
            module = _MODULE_FOR_BACKEND.get(name)
            if module and importlib.util.find_spec(module) is not None:
                return name
        return "null"

    if requested in {"fixture", "null"}:
        return requested
    module = _MODULE_FOR_BACKEND.get(requested)
    if module is None:
        raise TranscriptionError(
            f"unknown transcription backend {requested!r}; "
            f"choose from {', '.join(BACKEND_PREFERENCE)}"
        )
    if importlib.util.find_spec(module) is None:
        raise TranscriptionError(
            f"transcription backend {requested!r} requires the {module!r} package. "
            f"Install it with `pip install 'callscope[{requested}]'`."
        )
    return requested


def transcribe(
    wav_path: str | Path, config: TranscriptionConfig | None = None
) -> tuple[Transcript, list[str]]:
    """Transcribe a normalized WAV, returning the transcript and any warnings."""
    cfg = config or TranscriptionConfig()
    path = Path(wav_path)
    if not path.exists():
        raise FileNotFoundError(f"audio not found for transcription: {path}")

    name = select_backend(cfg.backend, config=cfg)
    warnings: list[str] = []
    if name == "null":
        warnings.append(
            "No local Whisper backend is installed, so the semantic track has no "
            "transcript to score. Install one with `pip install 'callscope[mlx]'` "
            "(Apple Silicon) or `pip install 'callscope[faster]'`."
        )

    backend = _build(name)
    try:
        transcript = backend.transcribe(path, cfg)
    except TranscriptionError:
        raise
    except Exception as exc:
        raise TranscriptionError(f"{name} backend failed on {path}: {exc}") from exc
    return transcript, warnings


def _build(name: str) -> TranscriptionBackend:
    match name:
        case "mlx":
            return MlxWhisperBackend()
        case "faster":
            return FasterWhisperBackend()
        case "openai":
            return OpenAiWhisperBackend()
        case "fixture":
            return FixtureBackend()
        case "null":
            return NullBackend()
    raise TranscriptionError(f"unknown backend {name!r}")


class NullBackend:
    """Produces an empty transcript so the paralinguistic track still runs."""

    name = "null"

    def is_available(self) -> bool:
        return True

    def transcribe(self, wav_path: Path, config: TranscriptionConfig) -> Transcript:
        return Transcript(segments=[], backend=self.name, language=config.language)


class FixtureBackend:
    """Loads a transcript from JSON instead of running a model.

    Accepts either a bare list of segment objects or a Whisper-shaped
    ``{"segments": [...], "language": "en"}`` document. This is what makes the
    scoring stages testable offline, and it doubles as a bring-your-own-transcript
    path for teams whose ASR already lives elsewhere.
    """

    name = "fixture"

    def is_available(self) -> bool:
        return True

    def transcribe(self, wav_path: Path, config: TranscriptionConfig) -> Transcript:
        if config.fixture_path is None:
            raise TranscriptionError(
                "the fixture backend needs a transcript: pass --transcript /path/to.json"
            )
        path = Path(config.fixture_path)
        if not path.exists():
            raise TranscriptionError(f"transcript fixture not found: {path}")
        try:
            raw: Any = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TranscriptionError(f"{path} is not valid JSON: {exc}") from exc
        return transcript_from_dict(raw, backend=self.name)


class MlxWhisperBackend:
    """Apple Silicon backend. Preferred when available."""

    name = "mlx"

    def is_available(self) -> bool:
        return importlib.util.find_spec("mlx_whisper") is not None

    def transcribe(self, wav_path: Path, config: TranscriptionConfig) -> Transcript:
        import mlx_whisper  # type: ignore[import-not-found]

        model = config.resolved_model(self.name)
        result = mlx_whisper.transcribe(
            str(wav_path),
            path_or_hf_repo=model,
            language=config.language,
            word_timestamps=False,
        )
        transcript = transcript_from_dict(result, backend=self.name)
        transcript.model = model
        return transcript


class FasterWhisperBackend:
    """CTranslate2 backend. Good CPU throughput, no Apple Silicon requirement."""

    name = "faster"

    def is_available(self) -> bool:
        return importlib.util.find_spec("faster_whisper") is not None

    def transcribe(self, wav_path: Path, config: TranscriptionConfig) -> Transcript:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]

        model_name = config.resolved_model(self.name) or "base.en"
        model = WhisperModel(model_name, device="auto", compute_type="int8")
        segments, info = model.transcribe(
            str(wav_path), language=config.language, vad_filter=False
        )
        out = [
            TranscriptSegment(
                start=float(s.start), end=float(s.end), text=str(s.text).strip()
            )
            for s in segments
            if float(s.end) > float(s.start)
        ]
        return Transcript(
            segments=out,
            language=getattr(info, "language", config.language),
            backend=self.name,
            model=model_name,
        )


class OpenAiWhisperBackend:
    """Reference PyTorch implementation. The portable fallback."""

    name = "openai"

    def is_available(self) -> bool:
        return importlib.util.find_spec("whisper") is not None

    def transcribe(self, wav_path: Path, config: TranscriptionConfig) -> Transcript:
        import whisper  # type: ignore[import-not-found]

        model_name = config.resolved_model(self.name) or "base.en"
        model = whisper.load_model(model_name)
        result = model.transcribe(str(wav_path), language=config.language, verbose=False)
        transcript = transcript_from_dict(result, backend=self.name)
        transcript.model = model_name
        return transcript


def transcript_from_dict(raw: Any, *, backend: str = "fixture") -> Transcript:
    """Normalize a Whisper-shaped dict (or a bare segment list) into a Transcript.

    Malformed segments are skipped rather than aborting the whole transcript --
    one bad timestamp in a 40-minute call should not lose the other 39 minutes.
    """
    if isinstance(raw, list):
        raw_segments: Any = raw
        language = None
    elif isinstance(raw, dict):
        raw_segments = raw.get("segments", [])
        language = raw.get("language")
    else:
        raise TranscriptionError(
            f"expected a transcript object or list, got {type(raw).__name__}"
        )

    if not isinstance(raw_segments, list):
        raise TranscriptionError("transcript 'segments' must be a list")

    segments: list[TranscriptSegment] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        text = str(item.get("text", "")).strip()
        if end <= start or not text:
            continue
        speaker = item.get("speaker")
        segments.append(
            TranscriptSegment(
                start=start,
                end=end,
                text=text,
                speaker=str(speaker) if speaker else None,
            )
        )

    segments.sort(key=lambda s: s.start)
    return Transcript(
        segments=segments,
        language=str(language) if language else None,
        backend=backend,
    )


def attribute_speakers(transcript: Transcript, turns: list) -> Transcript:
    """Attach a speaker label to each transcript segment by maximal overlap.

    Segments with no overlapping turn (Whisper padding past the end of speech,
    say) fall back to the nearest turn by midpoint distance, so every segment
    that has text ends up attributable.
    """
    if not turns:
        return transcript
    ordered = sorted(turns, key=lambda t: t.start)
    attributed: list[TranscriptSegment] = []

    for seg in transcript.segments:
        best_turn = None
        best_overlap = 0.0
        for turn in ordered:
            if turn.start >= seg.end:
                break
            overlap = min(seg.end, turn.end) - max(seg.start, turn.start)
            if overlap > best_overlap:
                best_overlap, best_turn = overlap, turn
        if best_turn is None:
            mid = (seg.start + seg.end) / 2.0
            best_turn = min(
                ordered, key=lambda t: abs(((t.start + t.end) / 2.0) - mid)
            )
        attributed.append(
            TranscriptSegment(
                start=seg.start, end=seg.end, text=seg.text, speaker=best_turn.speaker
            )
        )

    return Transcript(
        segments=attributed,
        language=transcript.language,
        backend=transcript.backend,
        model=transcript.model,
    )
