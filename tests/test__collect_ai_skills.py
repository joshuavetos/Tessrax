"""Integration smoke-tests for the ``ai_skills`` namespace."""

from __future__ import annotations

from ai_skills.prompting import TemplateEngine


def test_package_exposes_templates() -> None:
    """The package should expose the bundled templates via ``TemplateEngine``."""

    engine = TemplateEngine()
    templates = engine.list_templates()
    assert "socratic_debugger" in templates
    assert "chain_of_verification" in templates
