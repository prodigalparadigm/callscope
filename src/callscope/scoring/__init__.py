"""The semantic scoring track.

The judge modules are named ``keyword_judge`` / ``llm_judge`` rather than
``keyword`` / ``llm`` so that they cannot shadow the standard library
``keyword`` module for anyone whose working directory happens to be this one.

Judges are interchangeable behind :class:`~callscope.scoring.base.SemanticJudge`.
Moving from deterministic keyword scoring to LLM judging is a rubric config
change (``judge: {backend: llm}``), not a code change -- see ``docs/llm-judge.md``.
"""

from callscope.scoring.base import (
    JudgeContext,
    SemanticJudge,
    build_judge,
    register_judge,
    score_transcript,
)
from callscope.scoring.keyword_judge import KeywordJudge
from callscope.scoring.llm_judge import LlmJudge

__all__ = [
    "JudgeContext",
    "KeywordJudge",
    "LlmJudge",
    "SemanticJudge",
    "build_judge",
    "register_judge",
    "score_transcript",
]
