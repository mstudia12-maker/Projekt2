from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from app.config import settings
from app.schemas import ComparisonResult, ProgressInfo

logger = logging.getLogger(__name__)

_lock = threading.Lock()


def _read_session_csv(path: Path) -> Optional[pd.DataFrame]:
    for enc in ("utf-8-sig", "utf-8", "cp1250", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            logger.warning("Odczyt sesji %s (%s): %s", path, enc, exc)
            return None
    try:
        return pd.read_csv(path, encoding="utf-8", encoding_errors="replace", low_memory=False)
    except Exception as exc:
        logger.warning("Odczyt sesji %s: %s", path, exc)
        return None


@dataclass
class SessionData:
    session_id: str
    raw_path: Path
    upload_df: Optional[pd.DataFrame] = None
    processed_df: Optional[pd.DataFrame] = None
    progress: ProgressInfo = field(
        default_factory=lambda: ProgressInfo(status="idle", message="", percent=0.0)
    )
    comparison_result: Optional[ComparisonResult] = None
    last_error: Optional[str] = None


class SessionStore:
    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base = Path(base_dir or settings.data_dir).resolve()
        self.base.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, SessionData] = {}

    def _ensure_upload_df(self, data: SessionData) -> None:
        if data.upload_df is not None:
            return
        df = _read_session_csv(data.raw_path)
        if df is not None:
            data.upload_df = df
        else:
            logger.error("Brak DataFrame dla sesji %s (plik: %s)", data.session_id, data.raw_path)

    def create_session(self, file_bytes: bytes, filename: str) -> SessionData:
        sid = str(uuid.uuid4())
        path = self.base / f"{sid}_{filename}"
        path.write_bytes(file_bytes)
        data = SessionData(session_id=sid, raw_path=path)
        with _lock:
            self._sessions[sid] = data
        return data

    def get(self, session_id: str) -> Optional[SessionData]:
        with _lock:
            if session_id in self._sessions:
                data = self._sessions[session_id]
                self._ensure_upload_df(data)
                return data
            for p in self.base.glob(f"{session_id}_*"):
                if p.is_file():
                    data = SessionData(session_id=session_id, raw_path=p)
                    self._ensure_upload_df(data)
                    self._sessions[session_id] = data
                    return data
        return None

    def update_progress(self, session_id: str, progress: ProgressInfo) -> None:
        s = self.get(session_id)
        if s:
            s.progress = progress

    def set_processed(self, session_id: str, df: pd.DataFrame) -> None:
        s = self.get(session_id)
        if s:
            s.processed_df = df

    def set_result(self, session_id: str, result: ComparisonResult | None, error: str | None) -> None:
        s = self.get(session_id)
        if s:
            s.comparison_result = result
            s.last_error = error


store = SessionStore()
