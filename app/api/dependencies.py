"""FastAPI dependencies for SQLite warehouse access."""
from __future__ import annotations

from collections.abc import Generator
import sqlite3

from fastapi import HTTPException, Request

from app.core.config import settings
from app.data.database import connect


def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """Yield a read-only SQLite connection for one request."""
    try:
        connection = connect(settings.database_path, read_only=True)
    except Exception as exc:
        raise HTTPException(status_code=503, detail={
            "code": "DATABASE_UNAVAILABLE",
            "message": "The data store is unavailable. Run the data pipeline and verify DATABASE_PATH.",
        }) from exc
    try:
        yield connection
    finally:
        connection.close()


def get_delay_predictor(request: Request):
    predictor = getattr(request.app.state, "delay_predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail={"code": "MODEL_NOT_READY", "message": "The saved delay model is not loaded."})
    return predictor
