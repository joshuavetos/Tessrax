"""Top-level package for the `ai_skills` library."""

from __future__ import annotations

__all__ = ["__version__", "AISkillsLabClient", "LiveReceiptStream", "ReceiptEnvelope"]

__version__ = "0.1.0"

from ai_skills.lab_integration import AISkillsLabClient, LiveReceiptStream, ReceiptEnvelope
