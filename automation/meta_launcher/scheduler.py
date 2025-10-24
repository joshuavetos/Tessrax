"""Scheduler CLI for Tessrax meta launch campaign."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from poster import PostPayload, dispatch_post

CONFIG_PATH = Path(__file__).resolve().with_name("config.json")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_template(content: str) -> Tuple[str, str]:
    lines = content.splitlines()
    title = ""
    body_lines: List[str] = []
    capture_body = False
    for line in lines:
        stripped = line.strip()
        if stripped == "## Title":
            capture_body = False
            continue
        if stripped == "## Body":
            capture_body = True
            body_lines = []
            continue
        if stripped.startswith("## "):
            capture_body = False
            continue
        if title == "" and not stripped.startswith("#") and stripped:
            title = stripped
            continue
        if capture_body:
            body_lines.append(line)
    body = "\n".join(body_lines).strip()
    return title, body


def iter_channels(config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    return sorted(config["channels"], key=lambda item: item["window_start"])


def governance_hash(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def ensure_logs_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def preview_plan(config: Dict[str, Any]) -> str:
    rows = []
    for entry in iter_channels(config):
        template_path = Path(entry["post_template"])
        title, _ = parse_template(template_path.read_text(encoding="utf-8"))
        rows.append(
            f"{entry['window_start']} | {entry['channel']} | {title}"
        )
    header = "Launch Plan (chronological)"
    return "\n".join([header, "-" * len(header), *rows])


def run_campaign(config: Dict[str, Any], *, live: bool = False) -> Path:
    channels = list(iter_channels(config))
    logs_dir = Path(config["campaign"]["logs_directory"])
    ensure_logs_directory(logs_dir)
    timestamp_obj = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    timestamp = timestamp_obj.isoformat().replace("+00:00", "Z")
    safe_stamp = timestamp_obj.strftime("%Y%m%dT%H%M%SZ")
    log_path = logs_dir / f"run_{safe_stamp}.log"
    governance_path = Path(config["campaign"]["governance_receipt_path"])
    receipt_hash = governance_hash(governance_path)
    mode = "LIVE" if live else "DRY-RUN"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# Tessrax Meta Launcher Log - {timestamp}\n")
        handle.write(f"# Governance Receipt SHA256: {receipt_hash}\n")
        handle.write(f"# Mode: {mode}\n")
        handle.write(f"# Total Channels: {len(channels)}\n\n")
        for entry in channels:
            template_path = Path(entry["post_template"])
            template_content = template_path.read_text(encoding="utf-8")
            title, body = parse_template(template_content)
            payload = PostPayload(
                channel=entry["channel"],
                title=title,
                body=body,
                metadata={
                    "client": entry["client"],
                    "window_start": entry["window_start"],
                    "window_end": entry["window_end"],
                    "platform": entry["platform_context"]["platform"],
                },
            )
            result = dispatch_post(payload, live=live)
            event_time = (
                dt.datetime.now(dt.timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
            handle.write(
                f"[{event_time}] {payload.channel} -> {payload.title} :: {result}\n"
            )
    ledger_path = Path(config["campaign"]["ledger_path"])
    ledger_entry = {
        "timestamp": timestamp,
        "mode": mode,
        "log": log_path.name,
        "total_channels": len(channels),
    }
    with ledger_path.open("a", encoding="utf-8") as ledger:
        ledger.write(json.dumps(ledger_entry) + "\n")
    return log_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tessrax launch scheduler")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preview_parser = subparsers.add_parser("preview", help="show launch plan")
    preview_parser.set_defaults(func=handle_preview)

    run_parser = subparsers.add_parser("run", help="execute schedule queue")
    run_parser.add_argument(
        "--live",
        action="store_true",
        help="perform live posting (requires prior approval)",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="force dry-run mode even if --live is supplied",
    )
    run_parser.set_defaults(func=handle_run)
    return parser


def handle_preview(_: argparse.Namespace) -> None:
    config = load_config(CONFIG_PATH)
    print(preview_plan(config))


def handle_run(args: argparse.Namespace) -> None:
    config = load_config(CONFIG_PATH)
    live = bool(args.live and not args.dry_run)
    log_path = run_campaign(config, live=live)
    print(f"Log written to {log_path}")


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
