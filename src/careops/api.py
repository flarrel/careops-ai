"""Local FastAPI endpoint for cited policy passages."""

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from careops.answer import MAX_QUERY_CHARS, load_answerer


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


def create_app(load_service=load_answerer):
    @asynccontextmanager
    async def lifespan(app):
        app.state.answerer = load_service()
        yield

    app = FastAPI(title="CareOps AI", lifespan=lifespan)

    @app.post("/answer", response_model=AnswerResponse)
    def answer(body: AnswerRequest, request: Request):
        try:
            return request.app.state.answerer.answer(body.query)
        except (OSError, RuntimeError) as exc:
            raise HTTPException(503, "Policy answer service unavailable") from exc

    return app


app = create_app()
