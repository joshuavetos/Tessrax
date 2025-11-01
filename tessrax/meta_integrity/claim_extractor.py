"""Lightweight text claim extraction tuned for epistemic audits."""

from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass

_HEDGE_PATTERN = re.compile(
    r"\b(might|possibly|maybe|could|unclear|unknown)\b", re.IGNORECASE
)


@dataclass(slots=True)
class ExtractedClaim:
    """Structured representation of a textual claim."""

    claim_id: str
    prompt_id: str
    statement: str
    object_text: str
    certainty: float
    metadata: Mapping[str, object]

    def as_dict(self) -> MutableMapping[str, object]:
        """Return a mutable mapping for downstream compatibility."""

        return {
            "claim_id": self.claim_id,
            "prompt_id": self.prompt_id,
            "statement": self.statement,
            "object": self.object_text,
            "certainty": self.certainty,
            "metadata": dict(self.metadata),
        }


class ClaimExtractor:
    """Extract coarse claims from natural-language answers."""

    _OBJECT_PATTERNS: Sequence[re.Pattern[str]] = (
        re.compile(r"\buses?\b(?P<object>.+)", re.IGNORECASE),
        re.compile(r"\brelies on\b(?P<object>.+)", re.IGNORECASE),
        re.compile(r"\bmight rely on\b(?P<object>.+)", re.IGNORECASE),
        re.compile(r"\bis\b(?P<object>.+)", re.IGNORECASE),
        re.compile(r"\bmight be\b(?P<object>.+)", re.IGNORECASE),
        re.compile(r"\bmirrored\b(?P<object>.+)", re.IGNORECASE),
    )

    def __init__(self, *, default_certainty: float = 0.9) -> None:
        self._default_certainty = max(0.0, min(default_certainty, 1.0))

    def extract(
        self,
        response: str,
        *,
        prompt_id: str,
        metadata: Mapping[str, object] | None = None,
    ) -> list[MutableMapping[str, object]]:
        """Return structured claims derived from ``response``.

        Parameters
        ----------
        response:
            Full model output or natural-language answer.
        prompt_id:
            Stable identifier for the originating prompt.
        metadata:
            Optional contextual metadata added to each claim.
        """

        clean_response = (response or "").strip()
        if not clean_response:
            return []

        sentences = self._segment(clean_response)
        claims: list[MutableMapping[str, object]] = []
        for index, sentence in enumerate(sentences, start=1):
            object_fragment = self._infer_object(sentence)
            certainty = self._estimate_certainty(sentence)
            claim = ExtractedClaim(
                claim_id=f"{prompt_id}-{index}",
                prompt_id=prompt_id,
                statement=sentence,
                object_text=object_fragment,
                certainty=certainty,
                metadata=metadata or {},
            )
            claims.append(claim.as_dict())
        return claims

    def _segment(self, text: str) -> list[str]:
        """Split text into sentences while preserving semantic cues."""

        parts = re.split(r"(?<=[.!?])\s+", text)
        sentences = [part.strip().rstrip("\"'") for part in parts if part.strip()]
        return sentences

    def _infer_object(self, sentence: str) -> str:
        """Infer the object or tail phrase of the claim."""

        for pattern in self._OBJECT_PATTERNS:
            match = pattern.search(sentence)
            if match:
                fragment = match.group("object").strip()
                fragment = fragment.strip(" .")
                if fragment:
                    return fragment
        return sentence

    def _estimate_certainty(self, sentence: str) -> float:
        """Derive a certainty score based on hedge cues."""

        certainty = self._default_certainty
        if not sentence:
            return 0.0
        hedges = len(_HEDGE_PATTERN.findall(sentence))
        certainty -= 0.25 * hedges
        if sentence.strip().endswith("?"):
            certainty -= 0.2
        return max(0.0, min(certainty, 1.0))


__all__ = ["ClaimExtractor", "ExtractedClaim"]
