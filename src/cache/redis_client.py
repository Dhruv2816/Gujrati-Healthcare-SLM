"""src/cache/redis_client.py — Redis-backed query cache."""

from __future__ import annotations
import hashlib
import json
from typing import Optional
import redis
from src.config import Config


def _make_key(query: str) -> str:
    return "gu_health:" + hashlib.sha256(query.strip().lower().encode()).hexdigest()


class RedisClient:
    """Simple Redis client for caching LLM answers."""

    def __init__(
        self,
        config: Config | None = None,
    ):
        self.config = config or Config()
        self.host: str = self.config.REDIS_HOST
        self.port: int = self.config.REDIS_PORT
        self.password: str = self.config.REDIS_PASSWORD

        self._r = redis.Redis(
            host=self.host,
            port=self.port,
            password=self.password or None,
            decode_responses=True,
        )

    def ping(self) -> bool:
        try:
            return self._r.ping()
        except Exception:
            return False

    def get_cached(self, query: str) -> Optional[dict]:
        """Return cached result dict or None."""
        try:
            raw = self._r.get(_make_key(query))
            if raw:
                return json.loads(raw)
        except Exception:
            pass
        return None

    def set_cache(self, query: str, result: dict, ttl: int | None = None) -> None:
        """Cache a result dict with TTL (seconds)."""
        if ttl is None:
            ttl = self.config.REDIS_CACHE_TTL
        try:
            self._r.setex(_make_key(query), ttl, json.dumps(result, ensure_ascii=False))
        except Exception:
            pass  # Cache failure should never break the pipeline

    def invalidate(self, query: str) -> None:
        try:
            self._r.delete(_make_key(query))
        except Exception:
            pass

    def flush_all(self) -> None:
        """⚠️ Clears entire cache — use cautiously."""
        try:
            self._r.flushdb()
        except Exception:
            pass

    def get_stats(self) -> dict:
        try:
            info = self._r.info("keyspace")
            return info
        except Exception:
            return {}
