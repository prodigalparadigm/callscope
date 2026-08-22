"""The semantic track: keyword judging, the judge registry, and the LLM interface."""

from __future__ import annotations

from typing import Any

import pytest

from callscope.errors import JudgeNotConfiguredError
from callscope.rubric import parse_rubric
from callscope.schema import SpeakerTurn, Transcript, TranscriptSegment
from callscope.scoring import (
    JudgeContext,
    KeywordJudge,
    LlmJudge,
    build_judge,
    register_judge,
    score_transcript,
)
from callscope.scoring.llm_judge import RESPONSE_SCHEMA, build_prompt


def _context(segments: list[tuple[float, float, str, str]], duration: float = 60.0):
    transcript = Transcript(
        segments=[
            TranscriptSegment(start, end, text, speaker)
            for start, end, speaker, text in segments
        ],
        backend="fixture",
    )
    turns = [
        SpeakerTurn(s.start, s.end, s.speaker or "SPEAKER_00") for s in transcript.segments
    ]
    return JudgeContext(
        transcript=transcript, turns=turns, duration_seconds=duration, call_id="t"
    )


CONVERSATION = [
    (0.0, 4.0, "SPEAKER_00", "Thanks for calling, this is Alex. May I take your name?"),
    (4.5, 8.0, "SPEAKER_01", "Hi Alex, my order has not arrived yet."),
    (8.5, 14.0, "SPEAKER_00", "Let me pull that up. I will reship it today at no charge."),
    (55.0, 59.0, "SPEAKER_00", "Anything else I can help with? Have a great day."),
]


# --- keyword judging -------------------------------------------------------

def test_matching_pattern_scores_full_marks():
    rubric = parse_rubric({"id": "r", "criteria": [
        {"id": "greeting", "patterns": [r"thanks for calling"]}]})
    result = score_transcript(rubric, _context(CONVERSATION))
    assert result.criteria[0].score == 1.0
    assert result.criteria[0].passed
    assert result.score_percent == 100.0


def test_absent_pattern_scores_zero():
    rubric = parse_rubric({"id": "r", "criteria": [
        {"id": "x", "patterns": [r"quantum entanglement"]}]})
    result = score_transcript(rubric, _context(CONVERSATION))
    assert result.criteria[0].score == 0.0
    assert not result.criteria[0].passed


def test_evidence_carries_timestamps_and_speaker():
    rubric = parse_rubric({"id": "r", "criteria": [
        {"id": "resolution", "patterns": [r"no charge"]}]})
    evidence = score_transcript(rubric, _context(CONVERSATION)).criteria[0].evidence
    assert evidence
    assert evidence[0].start == pytest.approx(8.5)
    assert evidence[0].end == pytest.approx(14.0)
    assert evidence[0].speaker == "SPEAKER_00"
    assert "no charge" in evidence[0].quote.lower()
    assert evidence[0].matched.lower() == "no charge"


def test_match_all_gives_proportional_credit():
    rubric = parse_rubric({"id": "r", "criteria": [{
        "id": "ident", "match": "all", "max_score": 1.0,
        "patterns": [r"this is", r"may i take", r"employee number \d+"],
    }]})
    result = score_transcript(rubric, _context(CONVERSATION)).criteria[0]
    assert result.score == pytest.approx(2 / 3, abs=1e-3)
    assert not result.passed
    assert "2/3" in result.rationale


def test_disqualifier_zeroes_a_matched_criterion():
    conversation = CONVERSATION + [(20.0, 22.0, "SPEAKER_00", "Calm down please.")]
    rubric = parse_rubric({"id": "r", "criteria": [{
        "id": "tone", "patterns": [r"(?s).*"], "disqualifiers": [r"calm down"],
    }]})
    result = score_transcript(rubric, _context(conversation)).criteria[0]
    assert result.score == 0.0
    assert not result.passed
    assert "disqualified" in result.rationale


def test_matching_is_case_insensitive():
    rubric = parse_rubric({"id": "r", "criteria": [
        {"id": "g", "patterns": [r"THANKS FOR CALLING"]}]})
    assert score_transcript(rubric, _context(CONVERSATION)).criteria[0].score == 1.0


# --- scope handling --------------------------------------------------------

def test_first_seconds_scope_excludes_later_matches():
    rubric = parse_rubric({"id": "r", "criteria": [{
        "id": "close_early", "scope": "first_seconds", "window_seconds": 10,
        "patterns": [r"have a great day"],
    }]})
    assert score_transcript(rubric, _context(CONVERSATION)).criteria[0].score == 0.0


def test_last_seconds_scope_finds_the_close():
    rubric = parse_rubric({"id": "r", "criteria": [{
        "id": "closing", "scope": "last_seconds", "window_seconds": 20,
        "patterns": [r"have a great day"],
    }]})
    assert score_transcript(rubric, _context(CONVERSATION)).criteria[0].score == 1.0


def test_speaker_scope_restricts_to_one_party():
    """The agent must say it; the customer saying it must not earn the point."""
    rubric = parse_rubric({"id": "r", "criteria": [{
        "id": "agent_only", "scope": "speaker", "speaker": "SPEAKER_01",
        "patterns": [r"thanks for calling"],
    }]})
    assert score_transcript(rubric, _context(CONVERSATION)).criteria[0].score == 0.0


def test_empty_transcript_scores_zero_with_a_reason():
    rubric = parse_rubric({"id": "r", "criteria": [{"id": "g", "patterns": ["hello"]}]})
    empty = JudgeContext(Transcript(), [], 30.0, "t")
    result = score_transcript(rubric, empty).criteria[0]
    assert result.score == 0.0
    assert "no transcript content" in result.rationale


# --- weighting and rollup --------------------------------------------------

def test_weights_change_the_overall_score():
    """A failed heavy criterion must cost more than a failed light one."""
    spec = {"id": "r", "criteria": [
        {"id": "hit", "patterns": [r"thanks for calling"], "weight": 1.0},
        {"id": "miss", "patterns": [r"nonexistent phrase"], "weight": 9.0},
    ]}
    heavy_miss = score_transcript(parse_rubric(spec), _context(CONVERSATION))
    assert heavy_miss.score_percent == pytest.approx(10.0, abs=0.1)

    spec["criteria"][0]["weight"] = 9.0  # type: ignore[index]
    spec["criteria"][1]["weight"] = 1.0  # type: ignore[index]
    heavy_hit = score_transcript(parse_rubric(spec), _context(CONVERSATION))
    assert heavy_hit.score_percent == pytest.approx(90.0, abs=0.1)


def test_partial_max_score_scales_correctly():
    rubric = parse_rubric({"id": "r", "criteria": [
        {"id": "a", "patterns": [r"thanks for calling"], "max_score": 5.0}]})
    result = score_transcript(rubric, _context(CONVERSATION))
    assert result.criteria[0].score == 5.0
    assert result.score_percent == 100.0


def test_pass_threshold_controls_the_pass_flag():
    rubric = parse_rubric({"id": "r", "criteria": [{
        "id": "a", "match": "all", "pass_threshold": 0.5,
        "patterns": [r"this is", r"never appears anywhere"],
    }]})
    result = score_transcript(rubric, _context(CONVERSATION)).criteria[0]
    assert result.score == pytest.approx(0.5)
    assert result.passed


# --- resilience ------------------------------------------------------------

def test_a_raising_judge_does_not_sink_the_other_criteria():
    """One bad criterion in a 40-line rubric must not lose the other 39."""

    class Exploding:
        name = "exploding"

        def judge(self, criterion, context):
            if criterion.id == "boom":
                raise RuntimeError("kaboom")
            return KeywordJudge().judge(criterion, context)

    rubric = parse_rubric({"id": "r", "criteria": [
        {"id": "ok", "patterns": [r"thanks for calling"]},
        {"id": "boom", "patterns": [r"x"]},
        {"id": "also_ok", "patterns": [r"no charge"]},
    ]})
    result = score_transcript(rubric, _context(CONVERSATION), judge=Exploding())
    by_id = {c.criterion_id: c for c in result.criteria}
    assert by_id["ok"].score == 1.0
    assert by_id["also_ok"].score == 1.0
    assert by_id["boom"].score == 0.0
    assert "kaboom" in by_id["boom"].rationale


def test_scoring_is_deterministic():
    rubric = parse_rubric({"id": "r", "criteria": [
        {"id": "a", "patterns": [r"thanks for calling"]},
        {"id": "b", "patterns": [r"no charge"]},
    ]})
    ctx = _context(CONVERSATION)
    assert score_transcript(rubric, ctx).score == score_transcript(rubric, ctx).score


# --- the registry ----------------------------------------------------------

def test_default_backend_builds_the_keyword_judge():
    rubric = parse_rubric({"id": "r", "criteria": [{"id": "a", "patterns": ["x"]}]})
    assert isinstance(build_judge(rubric), KeywordJudge)


def test_llm_backend_builds_the_llm_judge():
    rubric = parse_rubric({"id": "r", "judge": {"backend": "llm"},
                           "criteria": [{"id": "a", "patterns": ["x"]}]})
    assert isinstance(build_judge(rubric), LlmJudge)


def test_third_party_judges_can_register():
    """The extension point exists so a house classifier plugs in without a fork."""

    class Always:
        name = "always"

        def judge(self, criterion, context):
            return KeywordJudge().judge(criterion, context)

    register_judge("always", lambda rubric: Always())
    rubric = parse_rubric({"id": "r", "criteria": [{"id": "a", "patterns": ["x"]}]})
    object.__setattr__(rubric.judge, "backend", "always")
    assert isinstance(build_judge(rubric), Always)


def test_unregistered_backend_raises():
    rubric = parse_rubric({"id": "r", "criteria": [{"id": "a", "patterns": ["x"]}]})
    object.__setattr__(rubric.judge, "backend", "not_registered")
    with pytest.raises(JudgeNotConfiguredError, match="no judge registered"):
        build_judge(rubric)


# --- the LLM judge, exercised with a stub client ---------------------------

class StubClient:
    """Records what it was asked and returns a canned, schema-shaped answer."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def complete(self, *, system: str, prompt: str, schema: dict[str, Any]):
        self.calls.append({"system": system, "prompt": prompt, "schema": schema})
        return self.payload


def test_llm_judge_maps_a_response_onto_a_criterion_result():
    client = StubClient({
        "score": 0.75,
        "rationale": "The agent greeted the caller warmly.",
        "evidence": [{"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00",
                      "quote": "Thanks for calling, this is Alex."}],
    })
    rubric = parse_rubric({"id": "r", "judge": {"backend": "llm"},
                           "criteria": [{"id": "greeting", "patterns": ["(?!)"]}]})
    judge = LlmJudge(rubric, client=client)
    result = score_transcript(rubric, _context(CONVERSATION), judge=judge)

    criterion = result.criteria[0]
    assert criterion.score == 0.75
    assert criterion.judge == "llm"
    assert result.judge_backend == "llm"
    assert criterion.evidence[0].quote.startswith("Thanks for calling")
    assert criterion.evidence[0].start == 0.0


def test_llm_judge_clamps_an_out_of_range_score():
    client = StubClient({"score": 99.0, "rationale": "overshot", "evidence": []})
    rubric = parse_rubric({"id": "r", "judge": {"backend": "llm"},
                           "criteria": [{"id": "a", "patterns": ["(?!)"], "max_score": 2.0}]})
    result = LlmJudge(rubric, client=client).judge(rubric.criteria[0], _context(CONVERSATION))
    assert result.score == 2.0


def test_llm_judge_drops_malformed_evidence():
    client = StubClient({
        "score": 1.0, "rationale": "ok",
        "evidence": [
            {"start": "not a number", "end": 1.0, "quote": "x"},
            {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00", "quote": "good one"},
            {"start": 3.0, "end": 4.0, "speaker": "SPEAKER_00", "quote": "   "},
        ],
    })
    rubric = parse_rubric({"id": "r", "judge": {"backend": "llm"},
                           "criteria": [{"id": "a", "patterns": ["(?!)"]}]})
    result = LlmJudge(rubric, client=client).judge(rubric.criteria[0], _context(CONVERSATION))
    assert len(result.evidence) == 1
    assert result.evidence[0].quote == "good one"


def test_llm_judge_does_not_call_the_model_with_no_transcript():
    client = StubClient({"score": 1.0, "rationale": "x", "evidence": []})
    rubric = parse_rubric({"id": "r", "judge": {"backend": "llm"},
                           "criteria": [{"id": "a", "patterns": ["(?!)"]}]})
    empty = JudgeContext(Transcript(), [], 10.0, "t")
    result = LlmJudge(rubric, client=client).judge(rubric.criteria[0], empty)
    assert result.score == 0.0
    assert client.calls == []


def test_llm_prompt_contains_timestamps_and_a_transcript_boundary():
    client = StubClient({"score": 1.0, "rationale": "x", "evidence": []})
    rubric = parse_rubric({"id": "r", "judge": {"backend": "llm"},
                           "criteria": [{"id": "a", "patterns": ["(?!)"]}]})
    LlmJudge(rubric, client=client).judge(rubric.criteria[0], _context(CONVERSATION))
    prompt = client.calls[0]["prompt"]
    assert "<transcript>" in prompt and "</transcript>" in prompt
    assert "<criterion>" in prompt
    assert "[0.00-4.00] SPEAKER_00:" in prompt
    assert client.calls[0]["schema"] is RESPONSE_SCHEMA


def test_llm_prompt_marks_the_transcript_as_data_not_instructions():
    """Call audio is untrusted input; the system prompt must say so."""
    from callscope.scoring.llm_judge import SYSTEM_PROMPT

    assert "not instructions" in SYSTEM_PROMPT


def test_response_schema_is_closed():
    """A closed schema is what makes parsing total rather than best-effort."""
    assert RESPONSE_SCHEMA["additionalProperties"] is False
    assert set(RESPONSE_SCHEMA["required"]) == {"score", "rationale", "evidence"}


def test_build_prompt_is_pure():
    rubric = parse_rubric({"id": "r", "criteria": [{"id": "a", "patterns": ["x"]}]})
    ctx = _context(CONVERSATION)
    segments = ctx.segments_in_scope(rubric.criteria[0])
    assert build_prompt(rubric.criteria[0], segments) == build_prompt(
        rubric.criteria[0], segments
    )


def test_llm_judge_without_a_client_or_credentials_fails_clearly(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    rubric = parse_rubric({"id": "r", "judge": {"backend": "llm"},
                           "criteria": [{"id": "a", "patterns": ["(?!)"]}]})
    with pytest.raises(JudgeNotConfiguredError):
        LlmJudge(rubric).judge(rubric.criteria[0], _context(CONVERSATION))


def test_evidence_is_deduplicated_across_patterns():
    """Two patterns hitting one sentence is one piece of evidence, not two."""
    rubric = parse_rubric({"id": "r", "criteria": [{
        "id": "ident", "match": "all",
        "patterns": [r"this is", r"may i take"],  # both live in segment 0
    }]})
    evidence = score_transcript(rubric, _context(CONVERSATION)).criteria[0].evidence
    assert len(evidence) == 1
    assert len({(e.start, e.end) for e in evidence}) == 1


def test_evidence_is_ordered_by_time():
    rubric = parse_rubric({"id": "r", "criteria": [{
        "id": "multi", "patterns": [r"have a great day", r"thanks for calling"],
    }]})
    evidence = score_transcript(rubric, _context(CONVERSATION)).criteria[0].evidence
    assert [e.start for e in evidence] == sorted(e.start for e in evidence)
