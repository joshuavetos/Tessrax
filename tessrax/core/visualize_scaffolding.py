"""
Tessrax Visualization Module v1.0
Displays scaffolding + governance events as a unified timeline.
"""

import json, html, webbrowser
from pathlib import Path
from textwrap import shorten
from datetime import datetime

SCAFFOLD_PATH = Path("data/scaffolding_log.jsonl")
LEDGER_PATH = Path("data/ledger.jsonl")
HTML_PATH = Path("data/visualization.html")

def _load_jsonl(path):
    if not path.exists(): return []
    data = []
    with open(path) as f:
        for line in f:
            try: data.append(json.loads(line))
            except: pass
    return data

def render_console(scaffolding, ledger):
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
    combined = sorted(combined, key=lambda x: x.get("time", ""))

    print("\n📜 Tessrax Governance Timeline\n" + "="*60)
    for entry in combined:
        print(f"[{entry['time']}] {entry['type']:<25} | {entry['file']:<35} | "
              f"{','.join(entry['tags']) or '-':<20} | {entry['summary']}")
    print("="*60)
    print(f"Entries: {len(combined)}  |  Scaffolding: {len(scaffolding)}  |  Ledger: {len(ledger)}")

def render_html(scaffolding, ledger):
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

    color_map = {
        "DESIGN_DECISION": "#9FE2BF",
        "DESIGN_DECISION_ACK": "#77DD77",
        "POLICY_VIOLATION": "#FF6961",
        "POLICY_VIOLATION_REVIEWED": "#FDFD96",
        "QUORUM_VOTE_RESULT": "#AEC6CF",
    }

    rows = []
    for c in combined:
        color = color_map.get(c["type"], "#FFFFFF")
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
    <html><head><meta charset='utf-8'><title>Tessrax Governance Timeline</title>
    <style>
        body {{ font-family: monospace; background:#202020; color:#DDD; }}
        table {{ width:100%; border-collapse:collapse; }}
        th,td {{ padding:6px; border:1px solid #444; vertical-align:top; }}
        th {{ background:#333; color:#FFD700; }}
        summary {{ cursor:pointer; color:#1E90FF; }}
        pre {{ white-space: pre-wrap; word-wrap: break-word; color:#EEE; }}
    </style></head><body>
    <h1>🧭 Tessrax Design + Governance History</h1>
    <table>
        <tr><th>Timestamp</th><th>Type</th><th>File</th><th>Tags</th><th>Details</th></tr>
        {''.join(rows)}
    </table>
    <p>Entries: {len(combined)}</p></body></html>
    """
    HTML_PATH.write_text(html_doc, encoding="utf-8")
    webbrowser.open(HTML_PATH.as_uri())
    print(f"✅ Visualization written to {HTML_PATH}")

if __name__ == "__main__":
    scaffolding = _load_jsonl(SCAFFOLD_PATH)
    ledger = _load_jsonl(LEDGER_PATH)
    print(f"Loaded {len(scaffolding)} scaffolding entries, {len(ledger)} ledger events.")
    render_console(scaffolding, ledger)
    if input("\nGenerate HTML dashboard? (y/n): ").strip().lower() == "y":
        render_html(scaffolding, ledger)