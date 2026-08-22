# Migrating the semantic track to LLM judging

The semantic track ships with a deterministic keyword judge. This document is the
migration path to model-based judging — the interface is already in place, so
this is a configuration change rather than a rewrite.

## When to migrate

Migrate a criterion when its ceiling is *recall of phrasing*, not recall of
meaning. Concretely: you have written the fifth regex alternative for the same
criterion, you are still finding false negatives in review, and the false
negatives do not share surface form.

Do **not** migrate a criterion that a regex answers correctly. The keyword judge
is free, offline, instantaneous, and byte-for-byte reproducible; those are real
properties and worth keeping wherever they suffice. A rubric where four
criteria are keyword-judged and two are LLM-judged is a normal end state — the
backend is currently rubric-level, so splitting that way means two rubric files
over the same call, or extending `JudgeSpec` to be per-criterion (see
[Per-criterion backends](#per-criterion-backends) below).

## Making the switch

Install the extra:

```bash
uv pip install "callscope[llm]"
export ANTHROPIC_API_KEY=...      # or run `ant auth login`
```

Change the rubric's `judge` block:

```yaml
judge:
  backend: llm
  model: claude-opus-5
  options:
    effort: medium        # low | medium | high | xhigh | max
    max_tokens: 4096
```

Nothing else changes. `analyze_call` returns the same `CallReport`, criteria
return the same `CriterionResult` with the same evidence timestamps, and the
JSON, text, and HTML reports are byte-identical in shape.

Fill in the `guidance` field on each criterion while you are there. The keyword
judge ignores it; the LLM judge puts it in the prompt, and it is the highest-
leverage thing you can write.

```yaml
- id: resolution
  name: Offered a concrete resolution
  description: A specific next action or remedy was offered.
  guidance: >
    A concrete resolution names an action and, where relevant, a timeframe.
    "Someone will get back to you" is not a resolution. Neither is agreeing
    that the problem is real.
  patterns: ["(?!)"]     # never matches; the LLM judge does the work
```

## What the model is actually asked

One request per criterion. The system prompt establishes the assessor role and
three rules that matter: quote verbatim, cite timestamps, and score 0 rather
than inventing support. The user message carries the criterion in a
`<criterion>` block and the in-scope transcript in a `<transcript>` block, each
line prefixed with `[start-end] SPEAKER_NN:`.

The response is constrained by `output_config.format` to a closed JSON schema —
`score`, `rationale`, and an `evidence` array of `{start, end, speaker, quote}`.
Constraining the output means parsing is total: a malformed score is prevented
at the request rather than handled at runtime. Scores outside `[0, max_score]`
are clamped, and evidence entries that fail to parse are dropped rather than
failing the criterion.

### One request per criterion

This is a deliberate cost-for-clarity trade. The model is never asked to hold
ten rubric lines in working memory at once; a failure on one criterion cannot
corrupt the others; and each score has its own auditable prompt you can replay.

Batching all criteria into a single request is the obvious optimization when
call volume makes the cost matter. The place to do it is a new judge class —
implement `SemanticJudge`, return a `CriterionResult` per criterion, register it
under a new name. Nothing else in the pipeline needs to know.

## The privacy consequence

This is the one configuration in which call content leaves the machine. Only
the in-scope transcript segments for LLM-judged criteria are sent — never audio,
never the paralinguistic profile — but that is still customer speech crossing a
network boundary, and in a regulated environment it is a decision that needs
signing off rather than defaulting.

The generated HTML report's footer changes to say so when this path is active.

There is currently **no PII redaction hook**. If you need one, the honest place
to put it is a wrapper around `LlmClient.complete` that scrubs the prompt, since
that is the single chokepoint every outbound request passes through.

## Pointing at something other than the Anthropic API

`LlmClient` is a one-method protocol:

```python
class LlmClient(Protocol):
    def complete(self, *, system: str, prompt: str,
                 schema: dict[str, Any]) -> dict[str, Any]: ...
```

Anything that satisfies it works — a self-hosted model behind vLLM, an internal
gateway, a fine-tuned classifier that ignores the prompt entirely and returns a
score from a feature vector. Wire it in directly:

```python
from callscope.scoring import LlmJudge, register_judge

register_judge("internal", lambda rubric: LlmJudge(rubric, client=MyGateway()))
```

Then set `judge: {backend: internal}` in the rubric. For a fully local judge,
this is also how you keep the "no data leaves the machine" property while still
getting semantic rather than lexical scoring.

## Per-criterion backends

`JudgeSpec` is currently rubric-level. Making it per-criterion is a small,
contained change:

1. Add an optional `judge` block to `CriterionSpec` in `rubric.py`, parsed by
   the existing `_parse_judge`.
2. In `score_transcript`, resolve the judge per criterion instead of once,
   caching instances by backend name so one client is built per backend.

It was left out of the initial version because a rubric that mixes backends
also mixes cost models and reproducibility guarantees, and that is worth an
explicit decision rather than a default.

## Validating the migration

Do not trust the switch on inspection. The procedure that actually works:

1. Score a held-out set of calls with both backends.
2. Have a human score the same set blind.
3. Compute agreement per criterion — Cohen's kappa if your scores are binary,
   mean absolute error if they are graded.
4. Migrate only the criteria where the LLM judge beats the keyword judge against
   the human baseline. Some criteria will not; "did the agent state their name"
   is a regex problem and always will be.

Keep the held-out set. It is also how you detect drift after a model change.
