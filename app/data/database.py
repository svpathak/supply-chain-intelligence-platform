"""SQLite connection and naming helpers for the local warehouse."""
from __future__ import annotations

from pathlib import Path
import sqlite3


def table_name(layer: str, name: str) -> str:
    if layer not in {"raw", "staging", "integration", "serving"}:
        raise ValueError(f"Unknown data layer: {layer}")
    return f"{layer}_{name}"


def connect(db_path: str | Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Open SQLite warehouse; read-only callers cannot create or mutate it."""
    path = Path(db_path)
    if not read_only:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(path), timeout=30.0)
    else:
        if not path.is_file():
            raise FileNotFoundError(f"SQLite warehouse does not exist: {path}")
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=10.0)
    connection.row_factory = None
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 10000")
    if not read_only:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
    return connection
