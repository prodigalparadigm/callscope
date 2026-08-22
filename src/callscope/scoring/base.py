"""The judge interface and the registry that selects one.

Everything a judge needs arrives in a :class:`JudgeContext`, and everything it
returns is a :class:`~callscope.schema.CriterionResult`. That narrow surface is
what makes the keyword-to-LLM migration a config change: a new judge implements
two methods and registers a name.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from callscope.errors import JudgeNotConfiguredError
from callscope.rubric import CriterionSpec, Rubric
from callscope.schema import (
    CriterionResult,
    SemanticResult,
    SpeakerTurn,
    Transcript,
    TranscriptSegment,
)


@dataclass(frozen=True, slots=True)
class JudgeContext:
    """Everything a judge is allowed to see about one call.

    Deliberately excludes the audio buffer. The semantic track scores *what was
    said*; letting it peek at the signal would collapse the two tracks into one
    and make disagreements between them impossible to interpret.
    """

    transcript: Transcript
    turns: list[SpeakerTurn]
    duration_seconds: float
    call_id: str

    def segments_in_scope(self, criterion: CriterionSpec) -> list[TranscriptSegment]:
        """Transcript segments a criterion is allowed to match against."""
        segments = self.transcript.segments
        match criterion.scope:
            case "speaker":
                if criterion.speaker is None:
                    return list(segments)
                return [s for s in segments if s.speaker == criterion.speaker]
            case "first_seconds":
                cutoff = criterion.window_seconds
                return [s for s in segments if s.start < cutoff]
            case "last_seconds":
                cutoff = max(0.0, self.duration_seconds - criterion.window_seconds)
                return [s for s in segments if s.end > cutoff]
            case _:
                return list(segments)


@runtime_checkable
class SemanticJudge(Protocol):
    """A pluggable scorer for rubric criteria."""

    name: str

    def judge(self, criterion: CriterionSpec, context: JudgeContext) -> CriterionResult:
        """Score one criterion against one call."""


JudgeFactory = Callable[[Rubric], SemanticJudge]

_REGISTRY: dict[str, JudgeFactory] = {}


def register_judge(name: str, factory: JudgeFactory) -> None:
    """Register a judge factory under ``name``.

    Third-party judges (a fine-tuned classifier, a rules engine already in the
    building) plug in here without touching callscope's own code.
    """
    _REGISTRY[name] = factory


def build_judge(rubric: Rubric) -> SemanticJudge:
    """Instantiate the judge named by ``rubric.judge.backend``.

    Raises:
        JudgeNotConfiguredError: the backend name is not registered.
    """
    factory = _REGISTRY.get(rubric.judge.backend)
    if factory is None:
        raise JudgeNotConfiguredError(
            f"no judge registered for backend {rubric.judge.backend!r}; "
            f"registered: {sorted(_REGISTRY) or '(none)'}"
        )
    return factory(rubric)


def score_transcript(
    rubric: Rubric,
    context: JudgeContext,
    *,
    judge: SemanticJudge | None = None,
) -> SemanticResult:
    """Run every criterion in ``rubric`` and roll the results up.

    A judge that raises on one criterion does not sink the whole call: that
    criterion is recorded as a zero with the failure in its rationale, and the
    rest of the rubric still runs. In batch QA, a partial score with a visible
    error beats an exception that loses 39 other criteria.
    """
    active = judge or build_judge(rubric)
    results: list[CriterionResult] = []

    for criterion in rubric.criteria:
        try:
            results.append(active.judge(criterion, context))
        except Exception as exc:  # noqa: BLE001 - isolating one criterion is the point
            results.append(
                CriterionResult(
                    criterion_id=criterion.id,
                    name=criterion.name,
                    score=0.0,
                    max_score=criterion.max_score,
                    weight=criterion.weight,
                    passed=False,
                    rationale=f"judge error ({type(exc).__name__}): {exc}",
                    evidence=[],
                    judge=getattr(active, "name", "unknown"),
                )
            )

    return SemanticResult(
        rubric_id=rubric.id,
        rubric_name=rubric.name,
        criteria=results,
        judge_backend=getattr(active, "name", rubric.judge.backend),
    )
