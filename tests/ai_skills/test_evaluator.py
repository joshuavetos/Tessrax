"""Tests for the deterministic evaluator utilities."""

from __future__ import annotations

from ai_skills.prompting.evaluator import Evaluator


def test_exact_match_reports_full_score() -> None:
    """Exact matches should report ``match=True`` and full similarity."""

    evaluator = Evaluator()
    result = evaluator.score("Answer", "answer")
    assert result.match is True
    assert result.similarity == 1.0
    assert "match" in result.explanation.lower()


def test_partial_overlap_reports_similarity() -> None:
    """Partial overlap should produce a similarity below one and an advisory message."""

    evaluator = Evaluator()
    result = evaluator.score("abc", "abz")
    assert result.match is False
    assert 0 < result.similarity < 1
    assert (
        "close" in result.explanation.lower() or "overlap" in result.explanation.lower()
    )


def test_dict_conversion_preserves_fields() -> None:
    """The dictionary representation must include the main result fields."""

    evaluator = Evaluator()
    payload = evaluator.score_as_dict("x", "y")
    assert set(payload) == {"match", "similarity", "explanation"}
