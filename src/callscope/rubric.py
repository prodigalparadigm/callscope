"""Rubric loading and validation.

Rubrics are user-supplied YAML or JSON. Nothing about the scoring criteria is
hardcoded -- the same binary scores a support call, a sales discovery call, or a
compliance disclosure review, and the only thing that changes is the file.

Validation is strict and the error messages name the offending path, because the
person editing a rubric is usually a QA lead, not the person who wrote this code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from callscope.errors import RubricError

Scope = Literal["call", "speaker", "first_seconds", "last_seconds"]

VALID_SCOPES: frozenset[str] = frozenset({"call", "speaker", "first_seconds", "last_seconds"})
VALID_MATCH_MODES: frozenset[str] = frozenset({"any", "all"})
VALID_BACKENDS: frozenset[str] = frozenset({"keyword", "llm"})


@dataclass(frozen=True, slots=True)
class CriterionSpec:
    """One scored line item in a rubric."""

    id: str
    name: str
    description: str
    weight: float
    max_score: float
    #: Where in the call this criterion may be satisfied.
    scope: str
    #: For ``scope: speaker``, whose speech is searched. ``None`` means either.
    speaker: str | None
    #: For the ``first_seconds`` / ``last_seconds`` scopes.
    window_seconds: float
    #: Regex patterns (case-insensitive) for the keyword judge.
    patterns: tuple[str, ...]
    #: ``any`` scores full marks on the first hit; ``all`` requires every pattern.
    match: str
    #: Patterns that, if matched, force the criterion to zero regardless of hits.
    disqualifiers: tuple[str, ...]
    #: Fraction of ``max_score`` at or above which the criterion counts as passed.
    pass_threshold: float
    #: Free-form guidance handed to the LLM judge. Ignored by the keyword judge.
    guidance: str | None = None


@dataclass(frozen=True, slots=True)
class JudgeSpec:
    """Which semantic judge to run, and how."""

    backend: str = "keyword"
    model: str | None = None
    #: Extra backend-specific options, passed through untouched.
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Rubric:
    """A validated, domain-agnostic scoring rubric."""

    id: str
    name: str
    description: str
    version: str
    criteria: tuple[CriterionSpec, ...]
    judge: JudgeSpec

    @property
    def total_weight(self) -> float:
        return sum(c.weight for c in self.criteria)


def load_rubric(path: str | Path) -> Rubric:
    """Load and validate a rubric from a ``.yaml``, ``.yml``, or ``.json`` file.

    Raises:
        RubricError: the file is missing, unparseable, or structurally invalid.
    """
    p = Path(path)
    if not p.exists():
        raise RubricError(f"rubric file not found: {p}")
    text = p.read_text(encoding="utf-8")

    try:
        if p.suffix.lower() == ".json":
            raw = json.loads(text)
        else:
            raw = yaml.safe_load(text)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise RubricError(f"{p} could not be parsed: {exc}") from exc

    if raw is None:
        raise RubricError(f"{p} is empty")
    return parse_rubric(raw, source=str(p))


def parse_rubric(raw: Any, *, source: str = "<memory>") -> Rubric:
    """Validate an already-parsed rubric mapping."""
    if not isinstance(raw, dict):
        raise RubricError(f"{source}: rubric must be a mapping, got {type(raw).__name__}")

    rubric_id = _require_str(raw, "id", source)
    name = str(raw.get("name") or rubric_id)
    description = str(raw.get("description") or "")
    version = str(raw.get("version") or "1")

    raw_criteria = raw.get("criteria")
    if not isinstance(raw_criteria, list) or not raw_criteria:
        raise RubricError(f"{source}: 'criteria' must be a non-empty list")

    criteria: list[CriterionSpec] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_criteria):
        spec = _parse_criterion(item, index=index, source=source)
        if spec.id in seen:
            raise RubricError(f"{source}: duplicate criterion id {spec.id!r}")
        seen.add(spec.id)
        criteria.append(spec)

    total = sum(c.weight for c in criteria)
    if total <= 0:
        raise RubricError(f"{source}: criterion weights sum to {total}; must be > 0")

    return Rubric(
        id=rubric_id,
        name=name,
        description=description,
        version=version,
        criteria=tuple(criteria),
        judge=_parse_judge(raw.get("judge"), source=source),
    )


def _parse_criterion(item: Any, *, index: int, source: str) -> CriterionSpec:
    where = f"{source}: criteria[{index}]"
    if not isinstance(item, dict):
        raise RubricError(f"{where} must be a mapping, got {type(item).__name__}")

    cid = _require_str(item, "id", where)
    scope = str(item.get("scope", "call")).strip()
    if scope not in VALID_SCOPES:
        raise RubricError(f"{where}: scope {scope!r} is not one of {sorted(VALID_SCOPES)}")

    match = str(item.get("match", "any")).strip().lower()
    if match not in VALID_MATCH_MODES:
        raise RubricError(f"{where}: match must be 'any' or 'all', got {match!r}")

    patterns = _string_tuple(item.get("patterns"), where=f"{where}.patterns")
    disqualifiers = _string_tuple(item.get("disqualifiers"), where=f"{where}.disqualifiers")

    weight = _positive_float(item.get("weight", 1.0), where=f"{where}.weight")
    max_score = _positive_float(item.get("max_score", 1.0), where=f"{where}.max_score")

    window = float(item.get("window_seconds", 60.0))
    if scope in {"first_seconds", "last_seconds"} and window <= 0:
        raise RubricError(f"{where}: window_seconds must be > 0 for scope {scope!r}")

    threshold = float(item.get("pass_threshold", 1.0))
    if not 0.0 <= threshold <= 1.0:
        raise RubricError(f"{where}: pass_threshold must be in [0, 1], got {threshold}")

    speaker = item.get("speaker")
    if speaker is not None:
        speaker = str(speaker)

    # A keyword-judged criterion with no patterns can never score; that is
    # almost always a typo rather than an intentional always-zero line item.
    if not patterns:
        raise RubricError(
            f"{where}: at least one pattern is required. If this criterion is meant "
            "for an LLM judge only, give it a pattern that can never match "
            "(e.g. '(?!)') and set `judge.backend: llm` at the rubric level."
        )
    _compile_all(patterns, where=f"{where}.patterns")
    _compile_all(disqualifiers, where=f"{where}.disqualifiers")

    return CriterionSpec(
        id=cid,
        name=str(item.get("name") or cid),
        description=str(item.get("description") or ""),
        weight=weight,
        max_score=max_score,
        scope=scope,
        speaker=speaker,
        window_seconds=window,
        patterns=patterns,
        match=match,
        disqualifiers=disqualifiers,
        pass_threshold=threshold,
        guidance=str(item["guidance"]) if item.get("guidance") else None,
    )


def _parse_judge(raw: Any, *, source: str) -> JudgeSpec:
    if raw is None:
        return JudgeSpec()
    if not isinstance(raw, dict):
        raise RubricError(f"{source}: 'judge' must be a mapping")
    backend = str(raw.get("backend", "keyword")).strip().lower()
    if backend not in VALID_BACKENDS:
        raise RubricError(
            f"{source}: judge.backend {backend!r} is not one of {sorted(VALID_BACKENDS)}"
        )
    options = raw.get("options") or {}
    if not isinstance(options, dict):
        raise RubricError(f"{source}: judge.options must be a mapping")
    model = raw.get("model")
    return JudgeSpec(
        backend=backend,
        model=str(model) if model else None,
        options=dict(options),
    )


def _require_str(mapping: dict[str, Any], key: str, where: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RubricError(f"{where}: '{key}' is required and must be a non-empty string")
    return value.strip()


def _string_tuple(value: Any, *, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise RubricError(f"{where} must be a string or a list of strings")
    return tuple(v for v in value if v.strip())


def _positive_float(value: Any, *, where: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise RubricError(f"{where} must be a number, got {value!r}") from exc
    if out <= 0:
        raise RubricError(f"{where} must be > 0, got {out}")
    return out


def _compile_all(patterns: tuple[str, ...], *, where: str) -> None:
    """Fail at load time on a bad regex, not halfway through a batch of calls."""
    import re

    for pattern in patterns:
        try:
            re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise RubricError(f"{where}: invalid regex {pattern!r}: {exc}") from exc
