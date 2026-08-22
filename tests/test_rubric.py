"""Rubric loading, validation, and the domain-agnostic contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from callscope.errors import RubricError
from callscope.rubric import Rubric, load_rubric, parse_rubric

RUBRIC_DIR = Path(__file__).resolve().parents[1] / "src" / "callscope" / "rubrics"

MINIMAL = {
    "id": "minimal_v1",
    "criteria": [{"id": "greeting", "patterns": ["hello"]}],
}


def _write(tmp_path: Path, data: dict, name: str = "r.yaml") -> Path:
    import yaml

    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# --- the bundled examples --------------------------------------------------


@pytest.mark.parametrize("name", ["support_call.yaml", "sales_discovery.yaml"])
def test_bundled_rubrics_load(name):
    rubric = load_rubric(RUBRIC_DIR / name)
    assert isinstance(rubric, Rubric)
    assert rubric.criteria
    assert rubric.total_weight > 0


def test_two_bundled_rubrics_share_no_criteria():
    """The engine must carry no domain assumptions -- proven by two disjoint rubrics."""
    support = {c.id for c in load_rubric(RUBRIC_DIR / "support_call.yaml").criteria}
    sales = {c.id for c in load_rubric(RUBRIC_DIR / "sales_discovery.yaml").criteria}
    assert support and sales
    assert not (support & sales)


def test_defaults_are_applied(tmp_path: Path):
    rubric = load_rubric(_write(tmp_path, MINIMAL))
    criterion = rubric.criteria[0]
    assert criterion.weight == 1.0
    assert criterion.max_score == 1.0
    assert criterion.scope == "call"
    assert criterion.match == "any"
    assert criterion.pass_threshold == 1.0
    assert rubric.judge.backend == "keyword"
    assert rubric.name == "minimal_v1"  # falls back to the id


# --- format support --------------------------------------------------------


def test_json_rubrics_load(tmp_path: Path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(MINIMAL), encoding="utf-8")
    assert load_rubric(path).id == "minimal_v1"


def test_yaml_and_json_produce_identical_rubrics(tmp_path: Path):
    yaml_path = _write(tmp_path, MINIMAL)
    json_path = tmp_path / "r.json"
    json_path.write_text(json.dumps(MINIMAL), encoding="utf-8")
    assert load_rubric(yaml_path).criteria == load_rubric(json_path).criteria


# --- validation ------------------------------------------------------------


def test_missing_file(tmp_path: Path):
    with pytest.raises(RubricError, match="not found"):
        load_rubric(tmp_path / "absent.yaml")


def test_empty_file(tmp_path: Path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(RubricError, match="empty"):
        load_rubric(path)


def test_malformed_yaml(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("id: x\ncriteria: [unclosed", encoding="utf-8")
    with pytest.raises(RubricError, match="could not be parsed"):
        load_rubric(path)


def test_missing_id():
    with pytest.raises(RubricError, match="'id' is required"):
        parse_rubric({"criteria": [{"id": "a", "patterns": ["x"]}]})


def test_no_criteria():
    with pytest.raises(RubricError, match="non-empty list"):
        parse_rubric({"id": "x", "criteria": []})


def test_duplicate_criterion_ids():
    with pytest.raises(RubricError, match="duplicate criterion id"):
        parse_rubric(
            {
                "id": "x",
                "criteria": [
                    {"id": "a", "patterns": ["x"]},
                    {"id": "a", "patterns": ["y"]},
                ],
            }
        )


def test_invalid_scope():
    with pytest.raises(RubricError, match="scope 'sometime' is not one of"):
        parse_rubric({"id": "x", "criteria": [{"id": "a", "patterns": ["x"], "scope": "sometime"}]})


def test_invalid_match_mode():
    with pytest.raises(RubricError, match="match must be"):
        parse_rubric({"id": "x", "criteria": [{"id": "a", "patterns": ["x"], "match": "some"}]})


def test_criterion_without_patterns_is_rejected():
    """A patternless keyword criterion can only ever score zero: reject the typo."""
    with pytest.raises(RubricError, match="at least one pattern"):
        parse_rubric({"id": "x", "criteria": [{"id": "a"}]})


def test_invalid_regex_fails_at_load_not_at_scoring_time():
    """Catching this at load keeps a batch of 400 calls from dying on call 200."""
    with pytest.raises(RubricError, match="invalid regex"):
        parse_rubric({"id": "x", "criteria": [{"id": "a", "patterns": ["(unclosed"]}]})


def test_negative_weight_rejected():
    with pytest.raises(RubricError, match="must be > 0"):
        parse_rubric({"id": "x", "criteria": [{"id": "a", "patterns": ["x"], "weight": -1}]})


def test_non_numeric_weight_rejected():
    with pytest.raises(RubricError, match="must be a number"):
        parse_rubric({"id": "x", "criteria": [{"id": "a", "patterns": ["x"], "weight": "heavy"}]})


def test_pass_threshold_out_of_range():
    with pytest.raises(RubricError, match=r"pass_threshold must be in \[0, 1\]"):
        parse_rubric(
            {"id": "x", "criteria": [{"id": "a", "patterns": ["x"], "pass_threshold": 1.5}]}
        )


def test_zero_window_for_windowed_scope_rejected():
    with pytest.raises(RubricError, match="window_seconds must be > 0"):
        parse_rubric(
            {
                "id": "x",
                "criteria": [
                    {"id": "a", "patterns": ["x"], "scope": "first_seconds", "window_seconds": 0}
                ],
            }
        )


def test_error_messages_name_the_offending_criterion():
    """A QA lead editing a 40-line rubric needs the index, not just 'invalid'."""
    with pytest.raises(RubricError, match=r"criteria\[1\]"):
        parse_rubric(
            {
                "id": "x",
                "criteria": [
                    {"id": "a", "patterns": ["x"]},
                    {"id": "b", "patterns": ["x"], "scope": "nonsense"},
                ],
            }
        )


# --- judge configuration ---------------------------------------------------


def test_judge_backend_is_configurable():
    rubric = parse_rubric(
        {
            "id": "x",
            "judge": {"backend": "llm", "model": "claude-opus-5", "options": {"effort": "low"}},
            "criteria": [{"id": "a", "patterns": ["x"]}],
        }
    )
    assert rubric.judge.backend == "llm"
    assert rubric.judge.model == "claude-opus-5"
    assert rubric.judge.options == {"effort": "low"}


def test_unknown_judge_backend_rejected():
    with pytest.raises(RubricError, match="judge.backend"):
        parse_rubric(
            {"id": "x", "judge": {"backend": "vibes"}, "criteria": [{"id": "a", "patterns": ["x"]}]}
        )


# --- normalization ---------------------------------------------------------


def test_single_pattern_string_is_accepted():
    rubric = parse_rubric({"id": "x", "criteria": [{"id": "a", "patterns": "hello"}]})
    assert rubric.criteria[0].patterns == ("hello",)


def test_disqualifiers_are_optional():
    rubric = parse_rubric({"id": "x", "criteria": [{"id": "a", "patterns": ["x"]}]})
    assert rubric.criteria[0].disqualifiers == ()


def test_rubric_is_immutable():
    """Frozen specs: a judge cannot quietly rewrite the rubric it is scoring against."""
    rubric = parse_rubric(MINIMAL)
    with pytest.raises((AttributeError, TypeError)):
        rubric.criteria[0].weight = 99  # type: ignore[misc]
