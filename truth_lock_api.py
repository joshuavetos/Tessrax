"""Truth-Lock Prototype FastAPI service.

This module implements a lightweight in-memory question answering service
that favours deterministic responses for a very small curated knowledge
base.  The goal is not to build a fully fledged AI system, but rather a
 demonstrator that highlights the governance surfaces of a hypothetical
"Truth-Lock" system:

* Every query is evaluated using an explicit verification policy and the
  result is persisted to an append-only ledger.
* A minimal red-team suite is exposed to validate falsifiability controls.
* A FastAPI interface is provided so that the service can be inspected with
  standard tooling (curl, httpie, browsers, etc.).

The implementation below intentionally keeps business logic extremely simple
so that the accompanying tests can reason about it deterministically.  The
knowledge base is a dictionary of fact -> answer pairs and only a handful of
questions are supported.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ledger_persistence import LEDGER_PATH, append_entry, load_ledger


class VerificationPolicy(BaseModel):
    """Policy describing how a query should be evaluated.

    The policy is intentionally small: in a real system this might track
    multiple independent verifiers or confidence thresholds.  Here we only
    expose a ``require_source`` flag that ensures at least one knowledge
    source is listed.
    """

    name: str = Field(default="default")
    require_source: bool = Field(
        default=True,
        description=(
            "If true, the query must reference a registered knowledge source."
        ),
    )


class KnowledgeSource(BaseModel):
    """Represents a curated source used to answer queries."""

    name: str
    url: Optional[str] = None


class QueryRequest(BaseModel):
    """Incoming request payload for the ``/query`` endpoint."""

    question: str = Field(..., description="User provided natural language question")
    policy: VerificationPolicy = Field(default_factory=VerificationPolicy)


class QueryResult(BaseModel):
    """Individual piece of evidence returned from the verifier."""

    source: KnowledgeSource
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    """Response payload returned from the ``/query`` endpoint."""

    question: str
    status: str
    results: List[QueryResult]
    evaluated_at: datetime


class RedTeamResult(BaseModel):
    """Outcome of a single red-team test."""

    name: str
    passed: bool
    details: str


class RedTeamSummary(BaseModel):
    """Collection of red-team test results."""

    all_passed: bool
    results: List[RedTeamResult]


class RedTeamRegistry:
    """Container for red-team scenarios that probe the service."""

    def __init__(self) -> None:
        self._tests = [
            ("accurate_fact", self._test_accurate_fact),
            ("unknown_fact", self._test_unknown_fact),
        ]

    def register(self, name: str, handler) -> None:
        """Register an additional test handler."""

        self._tests.append((name, handler))

    def run_falsifiability_suite(self, service: "TruthLockService") -> RedTeamSummary:
        """Execute all tests against the provided service instance."""

        results: List[RedTeamResult] = []
        all_passed = True
        for name, handler in self._tests:
            try:
                passed, details = handler(service)
            except Exception as exc:  # pragma: no cover - defensive logging
                passed = False
                details = f"Unhandled exception: {exc!r}"
            results.append(RedTeamResult(name=name, passed=passed, details=details))
            all_passed = all_passed and passed
        return RedTeamSummary(all_passed=all_passed, results=results)

    @staticmethod
    def _test_accurate_fact(service: "TruthLockService") -> (bool, str):
        """Ensure a canonical fact returns a verified status."""

        response = service.answer_query("What is the capital of France?")
        passed = response.status == "verified" and response.results[0].answer == "Paris"
        details = "Capital of France verified" if passed else "Unexpected verification"
        return passed, details

    @staticmethod
    def _test_unknown_fact(service: "TruthLockService") -> (bool, str):
        """Ensure unrecognised facts are treated as unknown."""

        response = service.answer_query("Who is the president of Atlantis?")
        passed = response.status == "unknown"
        details = "Unknown facts rejected" if passed else "Unexpected acceptance"
        return passed, details


class TruthLockService:
    """Core service responsible for verifying questions."""

    def __init__(self) -> None:
        self.knowledge_base: Dict[str, str] = {
            "what is the capital of france?": "Paris",
            "who wrote pride and prejudice?": "Jane Austen",
        }
        self.knowledge_sources: Dict[str, KnowledgeSource] = {
            "world_facts": KnowledgeSource(
                name="World Fact Compendium", url="https://example.com/world-facts"
            ),
            "literature": KnowledgeSource(
                name="Literature Reference", url="https://example.com/literature"
            ),
        }
        self.red_team_registry = RedTeamRegistry()

    def _normalise_question(self, question: str) -> str:
        return question.strip().lower()

    def answer_query(self, question: str, policy: Optional[VerificationPolicy] = None) -> QueryResponse:
        """Answer a question and persist the interaction to the ledger."""

        policy = policy or VerificationPolicy()
        if policy.require_source and not self.knowledge_sources:
            raise HTTPException(status_code=400, detail="No knowledge sources registered")

        normalised = self._normalise_question(question)
        evaluated_at = datetime.now(timezone.utc)
        results: List[QueryResult] = []
        status = "unknown"

        if normalised in self.knowledge_base:
            answer = self.knowledge_base[normalised]
            source = self.knowledge_sources["world_facts"]
            results.append(QueryResult(source=source, answer=answer, confidence=1.0))
            status = "verified"

        entry = {
            "question": question,
            "normalised": normalised,
            "status": status,
            "results": [r.model_dump() for r in results],
            "evaluated_at": evaluated_at.isoformat(),
            "ledger_path": str(LEDGER_PATH),
        }
        append_entry(entry)

        return QueryResponse(
            question=question,
            status=status,
            results=results,
            evaluated_at=evaluated_at,
        )


app = FastAPI(title="Truth-Lock Prototype", version="1.0.0")
service = TruthLockService()


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Answer the supplied question using the Truth-Lock service."""

    return service.answer_query(request.question, request.policy)


@app.get("/red-team/test", response_model=RedTeamSummary)
def red_team() -> RedTeamSummary:
    """Run the falsifiability suite and return structured results."""

    return service.red_team_registry.run_falsifiability_suite(service)


@app.get("/ledger/entries")
def ledger_entries(limit: int = 50) -> List[dict]:
    """Return the most recent ledger entries up to the specified limit."""

    return load_ledger(limit=limit)


@app.get("/health")
def health() -> Dict[str, str]:
    """Liveness endpoint for operational monitoring."""

    return {"status": "ok", "evaluated_at": datetime.now(timezone.utc).isoformat()}


if __name__ == "__main__":
    # Running the module directly will start the FastAPI server using uvicorn.
    import uvicorn

    uvicorn.run("truth_lock_api:app", host="0.0.0.0", port=8000, reload=False)
