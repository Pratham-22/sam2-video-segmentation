"""In-pod hot state for the active chunk session (no Redis)."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from app.schemas import PointPrompt


@dataclass
class ChunkSession:
    session_id: str
    upload_id: str
    chunk_index: int
    process_start: int
    process_end: int
    save_start: int
    save_end: int
    mode: Literal["points", "text"] | None = None
    text_prompt: str | None = None
    clicks: dict[int, dict[int, list[PointPrompt]]] = field(default_factory=dict)
    # frame_idx -> obj_id -> points
    inference_handle: Any | None = None
    engine_kind: str | None = None


class SessionManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, ChunkSession] = {}

    def create(
        self,
        upload_id: str,
        chunk_index: int,
        process_start: int,
        process_end: int,
        save_start: int,
        save_end: int,
    ) -> ChunkSession:
        session = ChunkSession(
            session_id=str(uuid.uuid4()),
            upload_id=upload_id,
            chunk_index=chunk_index,
            process_start=process_start,
            process_end=process_end,
            save_start=save_start,
            save_end=save_end,
        )
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> ChunkSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def delete(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session and session.inference_handle is not None:
            try:
                from app.sam3_engine import release_session

                release_session(session)
            except Exception:
                pass

    def add_points(
        self,
        session_id: str,
        frame_idx: int,
        obj_id: int,
        points: list[PointPrompt],
        replace: bool,
    ) -> None:
        session = self.get(session_id)
        session.mode = "points"
        by_frame = session.clicks.setdefault(frame_idx, {})
        if replace or obj_id not in by_frame:
            by_frame[obj_id] = list(points)
        else:
            by_frame[obj_id].extend(points)

    def set_text(self, session_id: str, text: str, frame_idx: int) -> None:
        session = self.get(session_id)
        session.mode = "text"
        session.text_prompt = text.strip()
        session.clicks.setdefault(frame_idx, {})


session_manager = SessionManager()
