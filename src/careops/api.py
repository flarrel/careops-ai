"""Local REST API for policy answers, synthetic review and run summaries."""

import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Path as PathArg, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from careops.agent_tools import ClaimTask, StrictInput, SyntheticClaim
from careops.agent_trace import JsonlTrace, start_event
from careops.answer import MAX_QUERY_CHARS, load_answerer
from careops.corpus import DEV_CLAIMS
from careops.run_history import HISTORY_PATH, RunHistory

APPROVAL_TTL = 600
MAX_PENDING = 32


class AnswerRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)


class Citation(BaseModel):
    source_id: str
    chunk_id: str
    location: str
    title: str
    url: str


class AnswerResponse(BaseModel):
    status: Literal["answer", "abstain"]
    answer: str
    citations: list[Citation]
    reason: Literal[
        "scope_required", "no_evidence", "insufficient_evidence", "invalid_citation"
    ] | None


class ReviewRequest(ClaimTask):
    kind: Literal["claim"] = "claim"


class Approval(StrictInput):
    approved: StrictBool


class PendingReview(BaseModel):
    run_id: str
    status: Literal["awaiting_approval"]


class ReviewResult(BaseModel):
    status: Literal["review", "abstain", "escalate"]
    summary: str
    reason: str
    citations: list[str]
    tool_calls: list[str]


class CompletedReview(BaseModel):
    run_id: str
    status: Literal["complete"]
    result: ReviewResult
    lookup: SyntheticClaim | None


class RunSummary(BaseModel):
    run_id: str
    kind: Literal["policy", "claim"]
    started_at: str
    status: Literal["answer", "review", "abstain", "escalate", "error"]
    reason: str | None
    duration_ms: float
    citation_count: int
    tool_call_count: int


def create_app(load_service=load_answerer, *, history_path: Path = HISTORY_PATH,
               claims_path: Path = DEV_CLAIMS, clock=monotonic):
    history = RunHistory(history_path)
    pending = {}
    pending_lock = Lock()
    service_lock = Lock()
    service = None
    agent = None

    def policy_answer(query):
        nonlocal service
        with service_lock:
            if service is None:
                service = load_service()
            answerer = service
        return answerer.answer(query)

    @asynccontextmanager
    async def lifespan(_app):
        history.initialize()
        yield
        pending.clear()

    app = FastAPI(title="CareOps AI", lifespan=lifespan)

    @app.middleware("http")
    async def no_cache(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request, exc):
        # Pydantic's default error body can echo user-supplied values.
        fields = {"query", "claim_id", "max_amount_idr", "policy_query", "approved", "limit",
                  "run_id", "kind"}
        invalid = sorted({item for error in exc.errors() for item in error["loc"]
                          if isinstance(item, str) and item in fields})
        detail = "Invalid request."
        if invalid:
            detail += " Check " + ", ".join(invalid) + "."
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.exception_handler(sqlite3.Error)
    async def history_error(_request, _exc):
        return JSONResponse(status_code=503, content={"detail": "Run history unavailable."})

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/answer", response_model=AnswerResponse)
    def answer(body: AnswerRequest):
        started = start_event()
        try:
            result = AnswerResponse.model_validate(policy_answer(body.query))
        except (OSError, RuntimeError, ValueError) as exc:
            history.record("policy", {"status": "error", "reason": "policy_unavailable"}, started)
            raise HTTPException(503, "Policy answer service unavailable") from exc
        history.record("policy", result.model_dump(), started)
        return result

    def expire(now):
        for run_id in [key for key, (expires, _) in pending.items() if expires <= now]:
            del pending[run_id]

    @app.post("/reviews", status_code=202, response_model=PendingReview)
    def start_review(body: ReviewRequest):
        nonlocal agent
        started = start_event()
        with pending_lock:
            expire(clock())
            if len(pending) >= MAX_PENDING:
                raise HTTPException(429, "Too many pending reviews. Complete one or try again later.")
            try:
                if agent is None:
                    from careops.agent import load_agent

                    trace = JsonlTrace(history_path.with_name("api-agent-traces.jsonl"))
                    agent = load_agent(policy_answer, trace, claims_path=claims_path)
                run = agent.start(body.model_dump())
            except (OSError, RuntimeError, ValueError) as exc:
                history.record("claim", {"status": "error", "reason": "review_unavailable"},
                               started)
                raise HTTPException(503, "Synthetic review unavailable.") from exc
            pending[run.run_id] = (clock() + APPROVAL_TTL, run)
        return run.response

    @app.post("/reviews/{run_id}/decision", response_model=CompletedReview)
    def decide(run_id: Annotated[str, PathArg(pattern=r"^[0-9a-f]{32}$")], body: Approval):
        with pending_lock:
            expire(clock())
            entry = pending.pop(run_id, None)
        if entry is None:
            raise HTTPException(404, "Pending review expired or already decided. Start a new review.")
        started = start_event()
        try:
            result = CompletedReview.model_validate(entry[1].resume(body.approved))
        except (OSError, RuntimeError, ValueError) as exc:
            history.record("claim", {"status": "error", "reason": "review_unavailable"},
                           started, run_id=run_id)
            raise HTTPException(503, "Synthetic review unavailable.") from exc
        history.record("claim", result.result.model_dump(), started, run_id=run_id)
        return result

    @app.get("/runs", response_model=list[RunSummary])
    def runs(limit: Annotated[int, Query(ge=1, le=100)] = 20):
        return history.recent(limit)

    return app


app = create_app(
    history_path=Path(os.environ.get("CAREOPS_HISTORY_PATH", HISTORY_PATH)),
    claims_path=Path(os.environ.get("CAREOPS_CLAIMS_PATH", DEV_CLAIMS)),
)
