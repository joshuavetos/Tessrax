"""
Tessrax Detachment Engine v1.0
------------------------------
Behavioral catalyst between contradiction detection and action.
Transforms recognition of failure into measurable detachment, readiness, and fuel events.

Dependencies:
    • re (regex parsing)
    • collections (Counter)
    • datetime (for ledger timestamps)
    • core.ledger_core.append_entry (for event logging)

Author: Tessrax LLC
"""

import re
from typing import Dict, List, Tuple, Any
from collections import Counter
from datetime import datetime
from core.ledger_core import append_entry

# === RECOGNITION PATTERNS =====================================================

RECOGNITION_PATTERNS = [
    r"\bthis isn'?t working\b",
    r"\bi (was|am) wrong\b",
    r"\bthe data (shows|proves|indicates)\b",
    r"\bwe need to (stop|change|pivot)\b"
]

# === ATTACHMENT TYPES =========================================================

ATTACHMENT_TAGS = {
    "time": [r"\bspent\b", r"\bwasted\b", r"\bmonths\b", r"\byears\b"],
    "identity": [r"\bi('?m| am) the\b", r"\bthat'?s who i am\b"],
    "ego": [r"\b(can'?t|won'?t) be wrong\b", r"\bprove\b"],
    "certainty": [r"\bcan'?t risk\b", r"\bunknown\b", r"\bdoubt\b"],
    "social": [r"\beveryone (thinks|does|expects)\b", r"\blook foolish\b"],
    "investment": [r"\bmoney\b", r"\binvested\b", r"\bcost\b", r"\bresources\b"]
}

# === CORE FUNCTIONS ===========================================================

def detect_recognition(text: str) -> bool:
    """Detect if the text includes an explicit recognition of failure or contradiction."""
    return any(re.search(p, text.lower()) for p in RECOGNITION_PATTERNS)

def parse_attachment(text: str) -> Tuple[List[str], float]:
    """Identify attachment categories and calculate total weight."""
    matches = []
    for tag, patterns in ATTACHMENT_TAGS.items():
        if any(re.search(p, text.lower()) for p in patterns):
            matches.append(tag)
    weight = min(1.0, len(matches) * 0.2)
    return matches, weight

def calc_detachment(
    recognition_event: bool,
    attachment_weight: float,
    threshold: float = 0.7
) -> Tuple[float, str]:
    """Compute detachment score and status."""
    score = (1 if recognition_event else 0) * (1 - attachment_weight)
    status = "READY" if score >= threshold else "PARALYZED"
    return score, status

def track_action(event_log: List[Dict], action_taken: bool) -> str:
    """Track outcome and create FUEL or PARALYSIS event."""
    if action_taken:
        event = {"event": "fuel_event", "stability_delta": +1.0}
        event_log.append(event)
        return "FUEL"
    else:
        event = {"event": "paralysis_alert", "stability_delta": -0.5}
        event_log.append(event)
        return "PARALYSIS"

def analyze_patterns(history: List[Dict]) -> Dict[str, Any]:
    """Aggregate detachment outcomes and attachment frequencies over time."""
    recognitions = sum(1 for h in history if h.get("type") == "Recognition")
    actions = sum(1 for h in history if h.get("event") == "fuel_event")
    dsr = actions / recognitions if recognitions else 0

    top_attachments = Counter(
        tag for h in history for tag in h.get("attachments", [])
    )

    return {
        "DSR": round(dsr, 2),
        "attachment_profile": top_attachments.most_common()
    }

# === PIPELINE ENTRY ===========================================================

def analyze_and_log(text: str, event_log: List[Dict]) -> Dict[str, Any]:
    """Run full detachment analysis and log results to detachment ledger."""
    recognition = detect_recognition(text)
    attachments, weight = parse_attachment(text)
    score, status = calc_detachment(recognition, weight)

    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "text": text,
        "recognition": recognition,
        "attachments": attachments,
        "attachment_weight": weight,
        "detachment_score": score,
        "status": status,
    }

    # Append to ledger for full audit trail
    append_entry(
        {"event": "DETACHMENT_ANALYSIS", "payload": result},
        ledger_path="data/detachment_ledger.jsonl"
    )

    event_log.append(result)
    return result
    
from core.detachment_engine import detect_recognition, parse_attachment, calc_detachment

def analyze_with_detachment(self, text: str) -> Dict[str, Any]:
    """Extended analysis pipeline: contradiction + detachment."""
    base = self.analyze_for_contradictions(text)
    recognition = detect_recognition(text)

    if recognition:
        attachments, weight = parse_attachment(text)
        score, status = calc_detachment(recognition, weight)
        base["detachment"] = {
            "recognition": recognition,
            "attachments": attachments,
            "attachment_weight": weight,
            "detachment_score": score,
            "status": status
        }

    return base