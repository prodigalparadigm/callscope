"""The default semantic judge: deterministic regex matching over the transcript.

Deliberately unsophisticated. It is fast, free, fully offline, and -- most
importantly for QA -- reproducible: the same call scores the same way every time,
which is a precondition for scores that affect anyone's performance review.

Its ceiling is real and stated in the README: it matches phrasings, not meaning.
The LLM judge exists for criteria where that ceiling is binding.
"""

from __future__ import annotations

import re

from callscope.rubric import CriterionSpec, Rubric
from callscope.schema import CriterionResult, Evidence
from callscope.scoring.base import JudgeContext, register_judge

#: Longest evidence quote retained per match. Keeps reports readable and avoids
#: a report file that is just the transcript again.
MAX_QUOTE_CHARS = 240


class KeywordJudge:
    """Scores criteria by regex match against in-scope transcript segments."""

    name = "keyword"

    def __init__(self, rubric: Rubric | None = None) -> None:
        self.rubric = rubric
        self._cache: dict[str, re.Pattern[str]] = {}

    def judge(self, criterion: CriterionSpec, context: JudgeContext) -> CriterionResult:
        """Score one criterion. Never raises on ordinary input."""
        segments = context.segments_in_scope(criterion)

        disqualified: list[Evidence] = []
        for pattern in criterion.disqualifiers:
            disqualified.extend(self._find(pattern, segments))

        hits: dict[str, list[Evidence]] = {}
        for pattern in criterion.patterns:
            found = self._find(pattern, segments)
            if found:
                hits[pattern] = found

        matched_count = len(hits)
        total_patterns = len(criterion.patterns)

        if disqualified:
            score = 0.0
            rationale = (
                f"disqualified: matched {len(disqualified)} disqualifying "
                f"pattern(s) in scope '{criterion.scope}'"
            )
            evidence = disqualified[:5]
        elif not segments:
            score = 0.0
            rationale = (
                f"no transcript content in scope '{criterion.scope}'"
                + (f" for {criterion.speaker}" if criterion.speaker else "")
            )
            evidence = []
        elif criterion.match == "all":
            # Partial credit is proportional: 3 of 4 required phrases scores 0.75.
            score = criterion.max_score * (matched_count / total_patterns)
            missing = [p for p in criterion.patterns if p not in hits]
            rationale = (
                f"matched {matched_count}/{total_patterns} required patterns"
                + (f"; missing: {', '.join(missing[:3])}" if missing else "")
            )
            evidence = _dedupe(hits)
        else:
            score = criterion.max_score if matched_count else 0.0
            rationale = (
                f"matched {matched_count}/{total_patterns} patterns (any-of)"
                if matched_count
                else f"no pattern matched in scope '{criterion.scope}'"
            )
            evidence = _dedupe(hits)

        ratio = score / criterion.max_score if criterion.max_score > 0 else 0.0
        return CriterionResult(
            criterion_id=criterion.id,
            name=criterion.name,
            score=round(score, 4),
            max_score=criterion.max_score,
            weight=criterion.weight,
            passed=ratio >= criterion.pass_threshold and not disqualified,
            rationale=rationale,
            evidence=evidence,
            judge=self.name,
        )

    def _compile(self, pattern: str) -> re.Pattern[str]:
        compiled = self._cache.get(pattern)
        if compiled is None:
            compiled = re.compile(pattern, re.IGNORECASE)
            self._cache[pattern] = compiled
        return compiled

    def _find(self, pattern: str, segments: list) -> list[Evidence]:
        """All segment-level matches for ``pattern``, with timestamps."""
        try:
            regex = self._compile(pattern)
        except re.error:
            # load_rubric compiles every pattern up front, so reaching here means
            # a hand-built spec. Treat it as a non-match rather than exploding.
            return []

        out: list[Evidence] = []
        for seg in segments:
            match = regex.search(seg.text)
            if match is None:
                continue
            quote = seg.text.strip()
            if len(quote) > MAX_QUOTE_CHARS:
                quote = quote[: MAX_QUOTE_CHARS - 1].rstrip() + "…"
            out.append(
                Evidence(
                    start=round(seg.start, 3),
                    end=round(seg.end, 3),
                    speaker=seg.speaker,
                    quote=quote,
                    matched=match.group(0),
                )
            )
        return out



def _dedupe(hits: dict[str, list[Evidence]], *, limit: int = 5) -> list[Evidence]:
    """Flatten per-pattern hits, dropping repeats of the same transcript segment.

    Two patterns matching the same sentence is one piece of evidence, not two;
    without this the report shows the same quote several times and looks broken.
    """
    seen: set[tuple[float, float]] = set()
    out: list[Evidence] = []
    for evidence in sorted(
        (e for group in hits.values() for e in group), key=lambda e: e.start
    ):
        key = (evidence.start, evidence.end)
        if key in seen:
            continue
        seen.add(key)
        out.append(evidence)
        if len(out) >= limit:
            break
    return out


register_judge("keyword", lambda rubric: KeywordJudge(rubric))
