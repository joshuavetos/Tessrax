# diff_tool.py

import difflib
import argparse
import os
from datetime import datetime

def parse_conversation(filepath: str) -> list[str]:
    """Reads a text file and returns a list of clean, non-empty lines."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def compare_conversations(conv1: list[str], conv2: list[str], threshold: float = 0.85):
    unique1 = set(conv1)
    unique2 = set(conv2)
    common = []

    for s1 in conv1:
        for s2 in conv2:
            matcher = difflib.SequenceMatcher(None, s1.lower(), s2.lower())
            if matcher.ratio() >= threshold:
                common.append((s1, s2))
                unique1.discard(s1)
                unique2.discard(s2)
                break

    contradictions = []
    contradiction_pairs = [("is", "is not"), ("will", "will not"), ("can", "cannot"), ("robust", "fragile"), ("agree", "disagree")]
    for s1 in list(unique1):
        for s2 in list(unique2):
            matcher = difflib.SequenceMatcher(None, s1.lower(), s2.lower())
            if 0.6 < matcher.ratio() < 0.95:
                for word1, word2 in contradiction_pairs:
                    if (word1 in s1.lower() and word2 in s2.lower()) or (word2 in s1.lower() and word1 in s2.lower()):
                        contradictions.append((s1, s2))
                        unique1.discard(s1)
                        unique2.discard(s2)
                        break

    return {
        "common": common,
        "unique1": list(unique1),
        "unique2": list(unique2),
        "contradictions": contradictions,
    }

def generate_html_report(results: dict, file1_name: str, file2_name: str) -> str:
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head><meta charset="UTF-8"><title>Conversation Diff Report</title></head>
    <body>
        <h1>Conversation Diff Report</h1>
        <h2>Contradictions</h2>{contradictions}
        <h2>Common</h2>{common}
        <h2>Unique from {f1}</h2>{u1}
        <h2>Unique from {f2}</h2>{u2}
    </body>
    </html>
    """

    def format_pairs(pairs):
        return "".join([f"<p>{a} | {b}</p>" for a, b in pairs]) if pairs else "<p>None</p>"

    def format_list(items):
        return "".join([f"<p>{i}</p>" for i in items]) if items else "<p>None</p>"

    return html_template.format(
        contradictions=format_pairs(results["contradictions"]),
        common=format_pairs(results["common"]),
        u1=format_list(results["unique1"]),
        u2=format_list(results["unique2"]),
        f1=os.path.basename(file1_name),
        f2=os.path.basename(file2_name)
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file1")
    parser.add_argument("file2")
    parser.add_argument("-o", "--output", default="report.html")
    args = parser.parse_args()

    conv1 = parse_conversation(args.file1)
    conv2 = parse_conversation(args.file2)
    results = compare_conversations(conv1, conv2)
    report = generate_html_report(results, args.file1, args.file2)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report written to {args.output}")

if __name__ == "__main__":
    main()
