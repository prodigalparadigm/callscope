"""Core data structures shared across the pipeline.

These are plain dataclasses rather than pydantic models on purpose: the whole
point of callscope is to have no surprising dependencies, and everything here is
either produced internally or validated at the boundary in ``rubric.py``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal

SpeakerLabel = Literal["SPEAKER_00", "SPEAKER_01"]

#: The only two speaker labels callscope will ever emit. The two-speaker
#: constraint is enforced at the diarization boundary, not by convention.
SPEAKER_LABELS: tuple[str, str] = ("SPEAKER_00", "SPEAKER_01")


@dataclass(frozen=True, slots=True)
class Interval:
    """A half-open time interval ``[start, end)`` in seconds."""

    start: float
    end: float

    def __post_init__(self) -> None:
        if not (self.end > self.start):
            raise ValueError(f"interval end must exceed start (got {self.start}, {self.end})")

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlap(self, other: Interval) -> float:
        """Seconds of overlap with ``other``; 0.0 when disjoint."""
        return max(0.0, min(self.end, other.end) - max(self.start, other.start))


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    """A contiguous stretch of speech attributed to one speaker."""

    start: float
    end: float
    speaker: str
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    def as_interval(self) -> Interval:
        return Interval(self.start, self.end)


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """One transcribed utterance, optionally attributed to a speaker."""

    start: float
    end: float
    text: str
    speaker: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(slots=True)
class Transcript:
    """An ordered set of transcript segments plus provenance."""

    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str | None = None
    backend: str = "none"
    model: str | None = None

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    def for_speaker(self, speaker: str) -> list[TranscriptSegment]:
        return [s for s in self.segments if s.speaker == speaker]

    def word_count(self, speaker: str | None = None) -> int:
        segs = self.segments if speaker is None else self.for_speaker(speaker)
        return sum(len(s.text.split()) for s in segs)


@dataclass(slots=True)
class SpeakerParalinguistics:
    """Per-speaker signal-derived metrics."""

    speaker: str
    talk_time_seconds: float
    talk_time_ratio: float
    turn_count: int
    mean_turn_seconds: float
    longest_turn_seconds: float
    #: Syllable-nucleus rate estimated from the amplitude envelope, in syllables
    #: per second of *speech* (not per second of wall clock).
    syllable_rate_hz: float
    #: Words per minute of speech. ``None`` when no transcript is available.
    words_per_minute: float | None
    f0_mean_hz: float | None
    f0_std_hz: float | None
    #: Pitch variability in semitones, which is comparable across speakers with
    #: different baseline pitch in a way that raw Hz standard deviation is not.
    f0_std_semitones: float | None
    voiced_fraction: float


@dataclass(slots=True)
class ParalinguisticProfile:
    """Whole-call signal-derived metrics, independent of what was said."""

    duration_seconds: float
    speech_seconds: float
    silence_seconds: float
    silence_ratio: float
    longest_silence_seconds: float
    silence_p50_seconds: float
    silence_p90_seconds: float
    dead_air_events: list[Interval]
    overlap_seconds: float
    overlap_events: list[Interval]
    interruptions: dict[str, int]
    mean_response_latency_seconds: float | None
    speakers: list[SpeakerParalinguistics]

    def speaker(self, label: str) -> SpeakerParalinguistics:
        for sp in self.speakers:
            if sp.speaker == label:
                return sp
        raise KeyError(label)


@dataclass(slots=True)
class Evidence:
    """A pointer back into the call supporting a criterion's score."""

    start: float
    end: float
    speaker: str | None
    quote: str
    matched: str | None = None


@dataclass(slots=True)
class CriterionResult:
    """The outcome of judging a single rubric criterion."""

    criterion_id: str
    name: str
    score: float
    max_score: float
    weight: float
    passed: bool
    rationale: str
    evidence: list[Evidence] = field(default_factory=list)
    judge: str = "unknown"

    @property
    def weighted_score(self) -> float:
        if self.max_score <= 0:
            return 0.0
        return (self.score / self.max_score) * self.weight


@dataclass(slots=True)
class SemanticResult:
    """All criterion outcomes for one call, plus the rolled-up score."""

    rubric_id: str
    rubric_name: str
    criteria: list[CriterionResult]
    judge_backend: str

    @property
    def total_weight(self) -> float:
        return sum(c.weight for c in self.criteria)

    @property
    def score(self) -> float:
        """Weighted score in ``[0, 1]``. Returns 0.0 for an empty rubric."""
        tw = self.total_weight
        if tw <= 0:
            return 0.0
        return sum(c.weighted_score for c in self.criteria) / tw

    @property
    def score_percent(self) -> float:
        return round(self.score * 100.0, 1)


@dataclass(slots=True)
class CallReport:
    """The combined per-call artifact: semantic track + paralinguistic track."""

    call_id: str
    source_path: str
    duration_seconds: float
    semantic: SemanticResult
    paralinguistics: ParalinguisticProfile
    transcript: Transcript
    turns: list[SpeakerTurn]
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Recursively convert to plain JSON-serializable Python."""
        return _asdict(self)


def _asdict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {}
        for f in dataclasses.fields(obj):
            out[f.name] = _asdict(getattr(obj, f.name))
        # Surface computed properties that a JSON consumer would otherwise miss.
        for prop in ("score", "score_percent", "weighted_score"):
            attr = getattr(type(obj), prop, None)
            if isinstance(attr, property):
                out[prop] = round(float(getattr(obj, prop)), 6)
        return out
    if isinstance(obj, dict):
        return {k: _asdict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_asdict(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj
