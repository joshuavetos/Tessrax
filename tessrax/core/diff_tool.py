# diff_tool.py
# Enhanced Conversation Diff & Contradiction Detector

import argparse
import difflib
import os
from datetime import datetime
from typing import List, Tuple, Dict


# ------------------------------------------------------------
# File Processing
# ------------------------------------------------------------

def parse_conversation(filepath: str) -> List[str]:
    """Reads a text file and returns a list of stripped, non-empty lines."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")


# ------------------------------------------------------------
# Comparison Logic
# ------------------------------------------------------------

def compare_conversations(
    conv1: List[str],
    conv2: List[str],
    threshold: float = 0.85
) -> Dict[str, List]:
    """
    Compare two lists of utterances:
    - Match similar lines
    - Identify contradictions
    - Return unique and overlapping segments
    """
    unique1 = set(conv1)
    unique2 = set(conv2)
    common_matches = []

    for s1 in conv1:
        for s2 in conv2:
            ratio = difflib.SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
            if ratio >= threshold:
                common_matches.append((s1, s2))
                unique1.discard(s1)
                unique2.discard(s2)
                break

    contradiction_pairs = [
        ("is", "is not"),
        ("will", "will not"),
        ("can", "cannot"),
        ("agree", "disagree"),
        ("robust", "fragile"),
        ("always", "never")
    ]

    contradictions = []

    for s1 in list(unique1):
        for s2 in list(unique2):
            ratio = difflib.SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
            if 0.6 < ratio < 0.95:
                for w1, w2 in contradiction_pairs:
                    if (w1 in s1.lower() and w2 in s2.lower()) or (w2 in s1.lower() and w1 in s2.lower()):
                        contradictions.append((s1, s2))
                        unique1.discard(s1)
                        unique2.discard(s2)
                        break

    return {
        "common": common_matches,
        "contradictions": contradictions,
        "unique1": list(unique1),
        "unique2": list(unique2),
    }


# ------------------------------------------------------------
# Report Generation
# ------------------------------------------------------------

def format_html_block(title: str, entries: List[Tuple[str, str]] | List[str]) -> str:
    """Format a block of HTML from list data."""
    if not entries:
        return f"<h2>{title}</h2><p><em>None</em></p>"

    if isinstance(entries[0], tuple):
        items = "".join(f"<li>{a} <strong>|</strong> {b}</li>" for a, b in entries)
    else:
        items = "".join(f"<li>{entry}</li>" for entry in entries)

    return f"<h2>{title}</h2><ul>{items}</ul>"


def generate_html_report(results: Dict[str, List], file1: str, file2: str) -> str:
    """Assemble the full HTML report."""
    report_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Conversation Diff Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 2em; }}
            h1 {{ color: #333; }}
            ul {{ padding-left: 1.5em; }}
            li {{ margin-bottom: 0.5em; }}
            strong {{ color: #a00; }}
        </style>
    </head>
    <body>
        <h1>Conversation Diff Report</h1>
        <p><em>Generated: {report_time}</em></p>
        {format_html_block("Contradictions", results["contradictions"])}
        {format_html_block("Common Lines", results["common"])}
        {format_html_block(f"Unique to {os.path.basename(file1)}", results["unique1"])}
        {format_html_block(f"Unique to {os.path.basename(file2)}", results["unique2"])}
    </body>
    </html>
    """


# ------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare two conversation logs.")
    parser.add_argument("file1", help="Path to first conversation file")
    parser.add_argument("file2", help="Path to second conversation file")
    parser.add_argument("-o", "--output", default="report.html", help="HTML report output file")
    parser.add_argument("-t", "--threshold", type=float, default=0.85, help="Similarity threshold (default: 0.85)")
    args = parser.parse_args()

    conv1 = parse_conversation(args.file1)
    conv2 = parse_conversation(args.file2)

    results = compare_conversations(conv1, conv2, threshold=args.threshold)
    html = generate_html_report(results, args.file1, args.file2)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[✓] Report saved to: {args.output}")


if __name__ == "__main__":
    main()