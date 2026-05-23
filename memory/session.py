"""Session management with Redis (optional) and in-memory fallback."""
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from config import config


@dataclass
class Session:
    id: str
    topic: str
    status: str = "running"
    messages: list[dict[str, Any]] = field(default_factory=list)
    search_results: list[dict[str, Any]] = field(default_factory=list)
    report: str = ""
    created_at: float = field(default_factory=time.time)


class SessionStore:
    """Session persistence. Uses Redis if configured, otherwise in-memory dict."""

    def __init__(self) -> None:
        self._redis = None
        if config.redis_enabled:
            from redis import Redis
            self._redis = Redis.from_url(config.redis_url, decode_responses=True)
        self._sessions: dict[str, Session] = {}

    def _key(self, session_id: str) -> str:
        return f"newsbot:session:{session_id}"

    def save(self, session: Session) -> None:
        if self._redis:
            self._redis.setex(
                self._key(session.id), config.session_ttl,
                json.dumps(asdict(session), ensure_ascii=False),
            )
        else:
            self._sessions[session.id] = session

    def load(self, session_id: str) -> Session | None:
        if self._redis and (raw := self._redis.get(self._key(session_id))):
            return Session(**json.loads(raw))
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        if self._redis:
            self._redis.delete(self._key(session_id))
        else:
            self._sessions.pop(session_id, None)


session_store = SessionStore()
