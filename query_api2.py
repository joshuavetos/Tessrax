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
