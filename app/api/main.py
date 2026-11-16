"""FastAPI application factory and cross-cutting middleware."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.data_quality import router as data_quality_router
from app.api.routes import router as api_router
from app.api.repository import DataStoreUnavailable
from app.core.config import settings
from app.core.logging import configure_logging
from app.ml.inference import DelayPredictor

configure_logging(settings.log_level)
logger = logging.getLogger("supply_chain.api")


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load once at startup. Fail fast rather than serving an API whose
        # required Part 3.1 prediction artifact is absent or corrupt.
        app.state.delay_predictor = DelayPredictor.load(settings.model_artifact_dir)
        yield
        app.state.delay_predictor = None

    app = FastAPI(
        title="Supply Chain Intelligence API",
        description="Read APIs for the curated supply-chain data product.",
        version=settings.api_version,
        lifespan=lifespan,
    )
    app.include_router(api_router)
    app.include_router(data_quality_router)

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.info(
                "http_request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "latency_ms": latency_ms,
                    "request_id": request_id,
                },
            )

    @app.exception_handler(DataStoreUnavailable)
    async def data_store_exception_handler(request: Request, exc: DataStoreUnavailable):
        logger.exception("data_store_unavailable")
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "DATA_STORE_QUERY_FAILED",
                    "message": "The curated data product is temporarily unavailable.",
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        detail = exc.detail
        if isinstance(detail, dict):
            code = detail.get("code", "HTTP_ERROR")
            message = detail.get("message", "The request could not be completed.")
            extra = {key: value for key, value in detail.items() if key not in {"code", "message"}}
        else:
            code = "HTTP_ERROR"
            message = str(detail)
            extra = {}
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": code, "message": message, **extra}},
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request parameters failed validation.",
                    "details": jsonable_encoder(exc.errors()),
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("unhandled_request_exception")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred.",
                }
            },
        )

    return app


app = create_app()
