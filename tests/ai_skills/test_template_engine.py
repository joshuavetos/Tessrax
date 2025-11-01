"""Tests for the deterministic template rendering utilities."""

from __future__ import annotations

import pytest

from ai_skills.prompting.template_engine import TemplateEngine


def test_render_known_template() -> None:
    """Rendering the stock template should substitute the provided keys."""

    engine = TemplateEngine()
    rendered = engine.render("socratic_debugger", task="Add", context="1 and 2")
    assert "TASK: Add" in rendered
    assert "CONTEXT: 1 and 2" in rendered


def test_missing_template_key_raises_keyerror() -> None:
    """Missing required keys must raise a ``KeyError`` with details."""

    engine = TemplateEngine()
    with pytest.raises(KeyError):
        engine.render("socratic_debugger", task="Add")


def test_extra_template_key_raises_keyerror() -> None:
    """Providing extra keys should raise a ``KeyError`` for clarity."""

    engine = TemplateEngine()
    with pytest.raises(KeyError):
        engine.render("socratic_debugger", task="Add", context="1 and 2", extra="boom")


def test_empty_template_raises_value_error(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Empty template files must be rejected with a ``ValueError``."""

    templates_dir = tmp_path_factory.mktemp("templates")
    (templates_dir / "blank.txt").write_text("   \n", encoding="utf-8")
    engine = TemplateEngine(templates_dir)
    with pytest.raises(ValueError):
        engine.render("blank")


def test_path_traversal_raises_permission_error() -> None:
    """Path traversal attempts are blocked with ``PermissionError``."""

    engine = TemplateEngine()
    with pytest.raises(PermissionError):
        engine.render("../secrets", task="x", context="y")


def test_describe_returns_placeholder_metadata() -> None:
    """Descriptions should expose template metadata for tooling."""

    engine = TemplateEngine()
    details = engine.describe("chain_of_verification")
    assert details.name == "chain_of_verification"
    assert "task" in details.placeholders and "context" in details.placeholders
    assert details.path.exists()
