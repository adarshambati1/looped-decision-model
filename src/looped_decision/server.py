"""Jev-compatible app factory. No fallback or random answers when no model exists."""

from dataclasses import dataclass
from typing import Protocol

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .api_contract import SystemOneRequest, SystemOneResponse


@dataclass(frozen=True)
class InferenceResult:
    response: SystemOneResponse
    passes: int
    elapsed_ms: float


class Backend(Protocol):
    model_id: str

    def infer(self, request: SystemOneRequest) -> InferenceResult: ...


def create_app(backend: Backend | None = None):
    app = FastAPI(title="Looped decision model", version="0.1.0")
    app.state.backend = backend

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Match Jev's error envelope, not FastAPI's default detail array.
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "message": "Invalid SystemOne request",
                    "field": ".".join(str(x) for x in exc.errors()[0]["loc"] if x != "body"),
                }
            },
        )

    @app.get("/health")
    def health():
        return {
            "ready": app.state.backend is not None,
            "model": getattr(app.state.backend, "model_id", None),
        }

    @app.post("/v1/systemone")
    async def systemone(payload: SystemOneRequest):
        engine = app.state.backend
        if engine is None:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {"message": "No calibrated checkpoint/backend loaded", "field": None}
                },
            )
        if payload.model != engine.model_id:
            return JSONResponse(
                status_code=422, content={"error": {"message": "Unknown model", "field": "model"}}
            )
        result = await run_in_threadpool(engine.infer, payload)
        if set(result.response.answers) != set(payload.questions):
            return JSONResponse(
                status_code=500,
                content={
                    "error": {"message": "Backend returned mismatched question IDs", "field": None}
                },
            )
        return JSONResponse(
            content=result.response.model_dump(mode="json"),
            headers={
                "X-Looped-Passes": str(result.passes),
                "X-Looped-Milliseconds": f"{result.elapsed_ms:.3f}",
            },
        )

    return app


app = create_app()
