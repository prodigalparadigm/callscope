"""End-to-end orchestration: audio in, :class:`CallReport` out.

    normalize -> VAD -> diarize -> transcribe -> attribute
                                       |              |
                                       |              +--> SEMANTIC   (rubric judge)
                                       +-------------------> PARALINGUISTIC (signal)
                                                              |
                                                          combine -> CallReport

The two scoring tracks are genuinely independent: the paralinguistic track needs
no transcript, and the semantic track never sees the waveform. A call with no
usable transcription still produces a full paralinguistic profile, and the
report says why the semantic side is empty rather than failing the run.
"""

from __future__ import annotations

import logging
import math
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from callscope.audio import AudioBuffer, load_call_audio, read_wav
from callscope.diarize import DiarizationConfig, diarize
from callscope.errors import TranscriptionError
from callscope.paralinguistics import ParalinguisticConfig, analyze_paralinguistics
from callscope.rubric import Rubric, load_rubric
from callscope.schema import CallReport, Transcript
from callscope.scoring import JudgeContext, SemanticJudge, score_transcript
from callscope.transcribe import (
    TranscriptionConfig,
    attribute_speakers,
    transcribe,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineConfig:
    """Everything the pipeline needs that is not the audio itself."""

    rubric_path: str | Path | None = None
    rubric: Rubric | None = None
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    paralinguistics: ParalinguisticConfig = field(default_factory=ParalinguisticConfig)
    diarizer_backend: str = "auto"
    #: Skip ffmpeg when the input is already canonical 16 kHz mono PCM WAV.
    skip_normalization: bool = False
    call_id: str | None = None


def analyze_call(
    source: str | Path,
    config: PipelineConfig | None = None,
    *,
    judge: SemanticJudge | None = None,
) -> CallReport:
    """Run the full pipeline over one recording.

    Args:
        source: Any audio file ffmpeg can decode.
        config: Pipeline configuration. A rubric must be supplied via
            ``rubric`` or ``rubric_path``.
        judge: Override the judge selected by the rubric. Mainly for tests and
            for callers wiring in their own scorer.

    Returns:
        A fully populated :class:`~callscope.schema.CallReport`.

    Raises:
        ValueError: no rubric was supplied.
        FileNotFoundError, FfmpegError, AudioFormatError: bad or unreadable input.
    """
    cfg = config or PipelineConfig()
    src = Path(source)
    started = time.monotonic()
    warnings: list[str] = []

    rubric = _resolve_rubric(cfg)
    call_id = cfg.call_id or src.stem

    with tempfile.TemporaryDirectory(prefix="callscope-") as tmp:
        audio, wav_path = _load_audio(src, Path(tmp), cfg)

        diarization = diarize(
            audio,
            config=cfg.diarization,
            backend=cfg.diarizer_backend,
            wav_path=wav_path,
        )
        warnings.extend(diarization.warnings)

        transcript, transcribe_warnings = _transcribe(wav_path, cfg)
        warnings.extend(transcribe_warnings)
        transcript = attribute_speakers(transcript, diarization.turns)

    # --- Track 1: paralinguistic. Signal only. -----------------------------
    profile = analyze_paralinguistics(
        audio,
        diarization.turns,
        transcript=transcript if transcript.segments else None,
        config=cfg.paralinguistics,
    )

    # --- Track 2: semantic. Transcript only. -------------------------------
    context = JudgeContext(
        transcript=transcript,
        turns=diarization.turns,
        duration_seconds=audio.duration,
        call_id=call_id,
    )
    semantic = score_transcript(rubric, context, judge=judge)
    if not transcript.segments:
        warnings.append(
            "Semantic scores are all zero because the transcript is empty. "
            "The paralinguistic track below is unaffected."
        )

    return CallReport(
        call_id=call_id,
        source_path=str(src),
        duration_seconds=round(audio.duration, 3),
        semantic=semantic,
        paralinguistics=profile,
        transcript=transcript,
        turns=diarization.turns,
        metadata={
            "rubric_id": rubric.id,
            "rubric_version": rubric.version,
            "diarization_backend": diarization.backend,
            "diarization_speakers": diarization.n_speakers,
            # The pyannote backend reports NaN here: it has no equivalent
            # diagnostic, and NaN is not valid JSON.
            "diarization_separation": None
            if math.isnan(diarization.separation)
            else round(diarization.separation, 4),
            "transcription_backend": transcript.backend,
            "transcription_model": transcript.model,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
        warnings=warnings,
    )


def analyze_batch(
    sources: list[str | Path],
    config: PipelineConfig | None = None,
) -> tuple[list[CallReport], list[tuple[str, Exception]]]:
    """Run the pipeline over many recordings.

    Returns successful reports and a list of ``(path, exception)`` for the ones
    that failed. One corrupt file in a nightly batch must not take down the run.
    """
    reports: list[CallReport] = []
    failures: list[tuple[str, Exception]] = []
    for source in sources:
        try:
            reports.append(analyze_call(source, config))
        except Exception as exc:  # noqa: BLE001 - isolating one file is the point
            logger.warning("failed to analyze %s: %s", source, exc)
            failures.append((str(source), exc))
    return reports, failures


def _resolve_rubric(cfg: PipelineConfig) -> Rubric:
    if cfg.rubric is not None:
        return cfg.rubric
    if cfg.rubric_path is not None:
        return load_rubric(cfg.rubric_path)
    raise ValueError("a rubric is required: set PipelineConfig.rubric or .rubric_path")


def _load_audio(src: Path, workdir: Path, cfg: PipelineConfig) -> tuple[AudioBuffer, Path]:
    if cfg.skip_normalization:
        # Still validated as canonical by read_wav, so skipping ffmpeg cannot
        # smuggle a 44.1 kHz stereo file into the rest of the pipeline.
        return read_wav(src), src
    return load_call_audio(src, workdir)


def _transcribe(wav_path: Path, cfg: PipelineConfig) -> tuple[Transcript, list[str]]:
    try:
        return transcribe(wav_path, cfg.transcription)
    except TranscriptionError as exc:
        # A transcription failure degrades the semantic track; it does not end
        # the run, because the paralinguistic track is still fully computable.
        logger.warning("transcription failed: %s", exc)
        return Transcript(segments=[], backend="failed"), [f"transcription failed: {exc}"]
