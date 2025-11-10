"""Utilities for loading and rendering hermetic text templates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from string import Formatter


@dataclass(frozen=True)
class TemplateDetails:
    """Small value object describing a template on disk."""

    name: str
    path: Path
    placeholders: list[str]


class TemplateEngine:
    """Render named templates stored on disk with explicit key validation."""

    def __init__(self, templates_dir: Path | None = None) -> None:
        """Initialise the engine with the directory where templates live."""

        default_dir = Path(__file__).resolve().parent / "templates"
        self._templates_dir = (
            Path(templates_dir).resolve() if templates_dir else default_dir
        )
        if not self._templates_dir.exists():
            raise FileNotFoundError(
                f"Template directory '{self._templates_dir}' is missing. Create it before use."
            )

    def list_templates(self) -> list[str]:
        """Return the available template names without file extensions."""

        names: list[str] = []
        for file in sorted(self._templates_dir.glob("*.txt")):
            names.append(file.stem)
        return names

    def describe(self, name: str) -> TemplateDetails:
        """Return metadata for a template, including the required placeholders."""

        template_path = self._safe_path_for(name)
        raw_template = template_path.read_text(encoding="utf-8")
        if raw_template.strip() == "":
            raise ValueError(
                f"Template '{name}' is empty. Populate it with content before use."
            )
        placeholders = self._extract_placeholders(raw_template)
        return TemplateDetails(name=name, path=template_path, placeholders=placeholders)

    def render(self, name: str, **values: str) -> str:
        """Render a template by name, validating placeholder coverage and shape."""

        template = self._load_template(name)
        placeholders = set(self._extract_placeholders(template))
        missing = placeholders.difference(values)
        if missing:
            joined = ", ".join(sorted(missing))
            raise KeyError(
                f"Template '{name}' expected keys {{{joined}}}, but they were not provided."
            )
        extras = set(values).difference(placeholders)
        if extras:
            joined = ", ".join(sorted(extras))
            raise KeyError(
                f"Template '{name}' does not define keys {{{joined}}}. Remove the unexpected data."
            )
        return template.format(**values)

    def _load_template(self, name: str) -> str:
        """Read the raw template text from disk with safety checks."""

        template_path = self._safe_path_for(name)
        content = template_path.read_text(encoding="utf-8")
        if content.strip() == "":
            raise ValueError(
                f"Template '{name}' is empty. Populate it with content before use."
            )
        return content

    def _safe_path_for(self, name: str) -> Path:
        """Resolve the file path for ``name`` while preventing path traversal attacks."""

        candidate = Path(name)
        if candidate.name != name or any(part == ".." for part in candidate.parts):
            raise PermissionError(
                "Template resolution blocked: only bare template names are permitted."
            )
        filename = f"{name}.txt" if not name.endswith(".txt") else name
        template_path = (self._templates_dir / filename).resolve()
        if not str(template_path).startswith(str(self._templates_dir)):
            raise PermissionError(
                "Template resolution blocked: attempted access outside the templates directory."
            )
        if not template_path.exists():
            raise KeyError(
                f"Template '{name}' was not found. Choose from: {', '.join(self.list_templates())}."
            )
        return template_path

    @staticmethod
    def _extract_placeholders(template: str) -> list[str]:
        """List the placeholder names defined in ``template`` in order of appearance."""

        formatter = Formatter()
        placeholders: list[str] = []
        for _, field_name, _, _ in formatter.parse(template):
            if field_name and field_name not in placeholders:
                placeholders.append(field_name)
        return placeholders

    def render_from_mapping(self, name: str, mapping: dict[str, str]) -> str:
        """Render a template using a mapping of values, mirroring ``dict`` semantics."""

        return self.render(name, **mapping)

    def ensure_ready(self) -> None:
        """Eagerly validate all templates to surface configuration issues early."""

        for template_name in self.list_templates():
            template = self._load_template(template_name)
            placeholders = self._extract_placeholders(template)
            self.render(template_name, **{key: "sample" for key in placeholders})

    def iter_templates(self) -> Iterable[TemplateDetails]:
        """Yield metadata for every template known to this engine."""

        for name in self.list_templates():
            yield self.describe(name)
