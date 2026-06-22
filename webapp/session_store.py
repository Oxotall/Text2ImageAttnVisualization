"""Hold per-generation results and their visualizers in server memory."""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass

from attnviz import (
    CrossAttentionVisualizer,
    GenerationResult,
    ImageToTextVisualizer,
    SelfAttentionVisualizer,
)


@dataclass
class Session:
    """One generation plus the visualizers built over it."""

    result: GenerationResult
    cross: CrossAttentionVisualizer
    self_attn: SelfAttentionVisualizer
    image2text: ImageToTextVisualizer


class SessionStore:
    """Thread-safe, bounded LRU store of active sessions.

    Each browser generation gets a UUID; the heavy attention tensors live here
    so follow-up token/region requests are cheap lookups instead of reruns.
    """

    def __init__(self, max_sessions: int = 8):
        self._max = max_sessions
        self._lock = threading.Lock()
        self._sessions: "OrderedDict[str, Session]" = OrderedDict()

    def add(self, session: Session) -> str:
        session_id = uuid.uuid4().hex
        with self._lock:
            self._sessions[session_id] = session
            self._sessions.move_to_end(session_id)
            self._evict_if_needed()
        return session_id

    def get(self, session_id: str) -> Session:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(session_id)
            self._sessions.move_to_end(session_id)
            return self._sessions[session_id]

    def _evict_if_needed(self) -> None:
        while len(self._sessions) > self._max:
            self._sessions.popitem(last=False)
