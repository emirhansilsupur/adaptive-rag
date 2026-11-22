"""Lightweight SQLite prompt cache to eliminate duplicate LLM spend during dev & eval."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class CachedResponse:
    """Entry retrieved from the local SQLite LLM cache."""

    response_text: str
    prompt_tokens: int
    completion_tokens: int


class SQLiteLLMCache:
    """Deterministic key-value cache backed by SQLite."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, timeout=10.0)

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    schema_name TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    @staticmethod
    def build_key(
        *,
        model_name: str,
        temperature: float,
        schema_name: str,
        messages: list[dict[str, Any]],
    ) -> str:
        payload = json.dumps(
            {
                "model_name": model_name,
                "temperature": round(temperature, 4),
                "schema_name": schema_name,
                "messages": messages,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> CachedResponse | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT response_text, prompt_tokens, completion_tokens
                FROM llm_cache
                WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return CachedResponse(
            response_text=str(row[0]),
            prompt_tokens=int(row[1]),
            completion_tokens=int(row[2]),
        )

    def set(
        self,
        *,
        cache_key: str,
        model_name: str,
        schema_name: str,
        response_text: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO llm_cache (
                    cache_key, model_name, schema_name, response_text, prompt_tokens, completion_tokens
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    model_name,
                    schema_name,
                    response_text,
                    prompt_tokens,
                    completion_tokens,
                ),
            )
            conn.commit()
