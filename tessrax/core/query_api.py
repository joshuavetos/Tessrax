# query_api.py
# Tessrax Engine v2.3 - Query API Layer
# Provides CLI and REST interfaces for querying scars, claims, and handoff chain integrity.

import argparse
import json
import os
import hashlib
from typing import List, Dict, Optional
from fastapi import FastAPI
import uvicorn

# ------------------------------
# Utility Functions
# ------------------------------

def _read_jsonl(path: str) -> List[dict]:
    """Read JSONL file and return list of entries."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def _sha256(obj: dict) -> str:
    """Deterministic hash of a dictionary with sorted keys."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

# ------------------------------
# Query API Class
# ------------------------------

class TessraxQueryAPI:
    def __init__(self, scars_file="scars.jsonl", claims_file="claims.jsonl", handoff_file="handoffs.jsonl"):
        self.scars_file = scars_file
        self.claims_file = claims_file
        self.handoff_file = handoff_file

    # ---- Scar Queries ----
    def get_scars(self, status: Optional[str] = None) -> List[dict]:
        scars = _read_jsonl(self.scars_file)
        if status:
            return [s for s in scars if s.get("status") == status]
        return scars

    # ---- Claim Queries ----
    def get_claims(self, unresolved_only: bool = False) -> List[dict]:
        claims = _read_jsonl(self.claims_file)
        if unresolved_only:
            return [c for c in claims if not c.get("resolved", False)]
        return claims

    # ---- Handoff Chain Verification ----
    def verify_chain(self, depth: Optional[int] = None) -> bool:
        handoffs = _read_jsonl(self.handoff_file)
        if depth:
            handoffs = handoffs[-depth:]

        prev_hash = None
        for entry in handoffs:
            chained = {"parent_hash": entry.get("parent_hash"), "state": entry.get("state")}
            if _sha256(chained) != entry["state_hash"]:
                return False
            if prev_hash and entry.get("parent_hash") != prev_hash:
                return False
            prev_hash = entry["state_hash"]
        return True

# ------------------------------
# CLI Interface
# ------------------------------

def cli():
    parser = argparse.ArgumentParser(description="Tessrax Query API CLI")
    subparsers = parser.add_subparsers(dest="command")

    scars_parser = subparsers.add_parser("scars", help="Query scars")
    scars_parser.add_argument("--status", choices=["open", "resolved"], help="Filter by status")

    claims_parser = subparsers.add_parser("claims", help="Query claims")
    claims_parser.add_argument("--unresolved", action="store_true", help="Only unresolved claims")

    handoff_parser = subparsers.add_parser("verify", help="Verify handoff chain")
    handoff_parser.add_argument("--depth", type=int, help="Verify last N handoffs only")

    rest_parser = subparsers.add_parser("serve", help="Run REST API server")
    rest_parser.add_argument("--port", type=int, default=8000, help="Port to serve on")

    args = parser.parse_args()
    api = TessraxQueryAPI()

    if args.command == "scars":
        print(json.dumps(api.get_scars(status=args.status), indent=2))
    elif args.command == "claims":
        print(json.dumps(api.get_claims(unresolved_only=args.unresolved), indent=2))
    elif args.command == "verify":
        print("Chain valid:", api.verify_chain(depth=args.depth))
    elif args.command == "serve":
        app = build_rest_api(api)
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        parser.print_help()

# ------------------------------
# REST API Interface
# ------------------------------

def build_rest_api(api: TessraxQueryAPI) -> FastAPI:
    app = FastAPI(title="Tessrax Query API", version="2.3")

    @app.get("/scars/")
    def scars(status: Optional[str] = None):
        return api.get_scars(status=status)

    @app.get("/claims/")
    def claims(unresolved_only: bool = False):
        return api.get_claims(unresolved_only=unresolved_only)

    @app.get("/handoffs/verify")
    def verify(depth: Optional[int] = None):
        return {"valid": api.verify_chain(depth=depth)}

    return app

# ------------------------------
# Entry Point
# ------------------------------

if __name__ == "__main__":
    cli()
# query_api.py
# Query API for Tessrax Engine v2.3

import argparse, json
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from src.tessrax_engine.engine import TessraxEngine

engine = TessraxEngine()

class QueryAPI:
    def __init__(self, engine):
        self.engine = engine

    def scars(self, status=None):
        scars = []
        with open(self.engine.handoff_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                scars.extend(entry.get("state", {}).get("active_scars", []))
        if status:
            return [s for s in scars if s.get("status") == status]
        return scars

    def claims(self, unresolved=False):
        claims = []
        with open(self.engine.claims_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                claims.append(entry)
        if unresolved:
            return [c for c in claims if c.get("status") == "unresolved"]
        return claims

    def verify_chain(self, depth=None):
        return self.engine.verify_handoff_chain()

def build_rest_api(api: QueryAPI):
    app = FastAPI(title="Tessrax Query API")

    @app.get("/scars")
    def get_scars(status: str | None = None):
        return api.scars(status=status)

    @app.get("/claims")
    def get_claims(unresolved: bool = False):
        return api.claims(unresolved=unresolved)

    @app.get("/verify")
    def verify_chain(depth: int | None = None):
        return {"valid": api.verify_chain(depth)}

    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query API for Tessrax Engine")
    parser.add_argument("mode", choices=["serve", "scars", "claims", "verify"])
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--status", type=str)
    parser.add_argument("--unresolved", action="store_true")
    args = parser.parse_args()

    api = QueryAPI(engine)

    if args.mode == "serve":
        app = build_rest_api(api)
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    elif args.mode == "scars":
        print(json.dumps(api.scars(status=args.status), indent=2))
    elif args.mode == "claims":
        print(json.dumps(api.claims(unresolved=args.unresolved), indent=2))
    elif args.mode == "verify":
        print(json.dumps({"valid": api.verify_chain()}, indent=2))
