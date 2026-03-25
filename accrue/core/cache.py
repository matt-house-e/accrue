"""CacheManager — SQLite-backed input-hash cache for pipeline steps.

Per-step-per-row caching with SHA-256 keys. Uses WAL mode for
concurrent access. Single `.accrue/cache.db` file, zero new deps.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class CacheManager:
    """SQLite-backed cache for step results.

    Each entry is keyed by a SHA-256 hash of the step's input
    (row data, prior results, field specs, model, temperature, etc.).
    TTL expiry is lazy — checked on ``get()``, bulk-cleaned via
    ``cleanup_expired()``.

    Args:
        cache_dir: Directory for ``cache.db``. Created if absent.
        ttl: Time-to-live in seconds. 0 = no expiry.
    """

    def __init__(self, cache_dir: str = ".accrue", ttl: int = 3600) -> None:
        self._cache_dir = Path(cache_dir)
        self._ttl = ttl
        self._conn: sqlite3.Connection | None = None

    def _ensure_connection(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        db_path = self._cache_dir / "cache.db"

        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-8000")  # 8 MB

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key        TEXT PRIMARY KEY,
                step_name  TEXT NOT NULL,
                value      TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_step ON cache(step_name)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires_at)")
        self._conn.commit()
        return self._conn

    def get(self, key: str) -> dict | None:
        """Look up a cached value by key. Returns None on miss or expiry."""
        conn = self._ensure_connection()
        row = conn.execute("SELECT value, expires_at FROM cache WHERE key = ?", (key,)).fetchone()

        if row is None:
            return None

        value_json, expires_at = row
        if expires_at is not None and time.time() > expires_at:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()
            return None

        return json.loads(value_json)

    def set(self, key: str, step_name: str, value: dict) -> None:
        """Store a value in the cache."""
        conn = self._ensure_connection()
        now = time.time()
        expires_at = (now + self._ttl) if self._ttl > 0 else None

        conn.execute(
            "INSERT OR REPLACE INTO cache (key, step_name, value, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (key, step_name, json.dumps(value, default=str), now, expires_at),
        )
        conn.commit()

    def delete_step(self, step_name: str) -> int:
        """Delete all cache entries for a step. Returns count deleted."""
        conn = self._ensure_connection()
        cursor = conn.execute("DELETE FROM cache WHERE step_name = ?", (step_name,))
        conn.commit()
        return cursor.rowcount

    def delete_all(self) -> int:
        """Delete all cache entries. Returns count deleted."""
        conn = self._ensure_connection()
        cursor = conn.execute("DELETE FROM cache")
        conn.commit()
        return cursor.rowcount

    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count deleted."""
        conn = self._ensure_connection()
        cursor = conn.execute(
            "DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at < ?",
            (time.time(),),
        )
        conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# Cache key computation
# ---------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, compact separators, str fallback."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def compute_cache_key(**components: Any) -> str:
    """SHA-256 hex digest of canonical JSON of *components*."""
    payload = canonical_json(components)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _compute_step_cache_key(
    step: Any,
    row: dict[str, Any],
    prior_results: dict[str, Any],
    step_fields: dict[str, Any],
) -> str:
    """Build the cache key for a step + row, duck-typing the step.

    LLMStep has ``model`` and ``temperature``; FunctionStep has ``cache_version``.
    """
    if getattr(step, "model", None) is not None:
        # LLMStep path
        system_prompt = getattr(step, "_custom_system_prompt", None) or ""
        system_prompt_header = getattr(step, "_system_prompt_header", None) or ""
        # Include grounding config in cache key when present
        grounding_cfg = getattr(step, "_grounding_config", None)
        grounding_hash = ""
        if grounding_cfg is not None:
            grounding_hash = hashlib.sha256(
                grounding_cfg.model_dump_json().encode("utf-8")
            ).hexdigest()
        return compute_cache_key(
            step_name=step.name,
            row=row,
            prior_results=prior_results,
            field_specs=step_fields,
            model=step.model,
            temperature=getattr(step, "temperature", None),
            system_prompt_hash=hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            system_prompt_header_hash=hashlib.sha256(
                system_prompt_header.encode("utf-8")
            ).hexdigest(),
            grounding_hash=grounding_hash,
        )
    else:
        # FunctionStep path
        return compute_cache_key(
            step_name=step.name,
            row=row,
            prior_results=prior_results,
            cache_version=getattr(step, "cache_version", None),
        )
