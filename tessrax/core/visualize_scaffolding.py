"""
Tessrax Visualization Module v1.0
Displays scaffolding + governance history as a timeline or HTML dashboard.

Author: Joshua Vetos / Tessrax LLC
License: CC BY 4.0
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from textwrap import shorten
import webbrowser
import html

# ============================================================
# Paths and Constants
# ============================================================

SCAFFOLD_PATH = Path("data/scaffolding_log.jsonl")
LEDGER_PATH = Path("data/ledger.jsonl")
HTML_PATH = Path("data/visualization.html")

# ============================================================
# Loaders
# ============================================================

def _load_jsonl(path):
    """Safely load JSONL file into a list."""
    if not path.exists(): 
        return []
    data = []
    with open(path) as f:
        for line in f:
            try: data.append(json.loads(line))
            except Exception: pass
    return data

def _sha256(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

# ============================================================
# Console View
# ============================================================

def render_console(scaffolding, ledger):
    """Render merged history to console as a timeline."""
    combined = []

    for s in scaffolding:
        combined.append({
            "time": s.get("timestamp"),
            "type": "DESIGN_DECISION",
            "file": s.get("file_changed"),
            "tags": s.get("tags", []),
            "summary": shorten(s.get("prompt", ""), 60)
        })

    for e in ledger:
        combined.append({
            "time": e.get("timestamp"),
            "type": e.get("event"),
            "file": e.get("file_changed", ""),
            "tags": e.get("tags", []),
            "summary": shorten(json.dumps(e, sort_keys=True), 60)
        })

    # sort chronologically
    combined = sorted(combined, key=lambda x: x.get("time", ""))

    print("\n📜 Tessrax Governance Timeline\n" + "="*60)
    for entry in combined:
        time = entry["time"]
        typ = entry["type"]
        file = entry["file"]
        tags = ",".join(entry["tags"]) if entry["tags"] else "-"
        summary = entry["summary"]
        print(f"[{time}] {typ:<25} | {file:<35} | {tags:<20} | {summary}")
    print("="*60)
    print(f"Entries: {len(combined)}  |  Scaffolding: {len(scaffolding)}  |  Ledger: {len(ledger)}")

# ============================================================
# HTML View
# ============================================================

def render_html(scaffolding, ledger):
    """Render timeline as interactive HTML file."""
    combined = []
    for s in scaffolding:
        combined.append({
            "time": s.get("timestamp"),
            "type": "DESIGN_DECISION",
            "file": s.get("file_changed"),
            "tags": s.get("tags", []),
            "prompt": s.get("prompt"),
            "response": s.get("response"),
        })

    for e in ledger:
        combined.append({
            "time": e.get("timestamp"),
            "type": e.get("event"),
            "file": e.get("file_changed", ""),
            "tags": e.get("tags", []),
            "details": e
        })

    combined = sorted(combined, key=lambda x: x.get("time", ""))

    rows = []
    for c in combined:
        color = {
            "DESIGN_DECISION": "#9FE2BF",
            "DESIGN_DECISION_ACK": "#77DD77",
            "POLICY_VIOLATION": "#FF6961",
            "POLICY_VIOLATION_REVIEWED": "#FDFD96",
            "QUORUM_VOTE_RESULT": "#AEC6CF",
        }.get(c["type"], "#FFFFFF")

        tags = ", ".join(c.get("tags", []))
        content = html.escape(json.dumps(c, indent=2, sort_keys=True))
        rows.append(
            f"<tr style='background:{color}'><td>{c['time']}</td>"
            f"<td>{html.escape(c['type'])}</td>"
            f"<td>{html.escape(c.get('file',''))}</td>"
            f"<td>{html.escape(tags)}</td>"
            f"<td><details><summary>View</summary><pre>{content}</pre></details></td></tr>"
        )

    html_doc = f"""
    <html>
    <head>
      <meta charset="utf-8">
      <title>Tessrax Governance Timeline</title>
      <style>
        body {{ font-family: monospace; background:#202020; color:#DDD; }}
        table {{ width:100%; border-collapse:collapse; }}
        th,td {{ padding:6px; border:1px solid #444; vertical-align:top; }}
        th {{ background:#333; color:#FFD700; }}
        summary {{ cursor:pointer; color:#1E90FF; }}
        pre {{ white-space: pre-wrap; word-wrap: break-word; color:#EEE; }}
      </style>
    </head>
    <body>
      <h1>🧭 Tessrax Design + Governance History</h1>
      <table>
        <tr><th>Timestamp</th><th>Type</th><th>File</th><th>Tags</th><th>Details</th></tr>
        {''.join(rows)}
      </table>
      <p>Entries: {len(combined)}</p>
    </body>
    </html>
    """

    HTML_PATH.write_text(html_doc, encoding="utf-8")
    webbrowser.open(HTML_PATH.as_uri())
    print(f"✅ Visualization written to {HTML_PATH}")

# ============================================================
# Main Entrypoint
# ============================================================

if __name__ == "__main__":
    scaffolding = _load_jsonl(SCAFFOLD_PATH)
    ledger = _load_jsonl(LEDGER_PATH)

    print(f"Loaded {len(scaffolding)} scaffolding entries, {len(ledger)} ledger events.")
    render_console(scaffolding, ledger)

    choice = input("\nGenerate HTML dashboard? (y/n): ").strip().lower()
    if choice == "y":
        render_html(scaffolding, ledger)