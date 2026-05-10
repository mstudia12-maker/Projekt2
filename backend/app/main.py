from __future__ import annotations

import io
import logging
from typing import Any, Optional

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import settings
from app.schemas import ProgressInfo, RunComparisonRequest, RunComparisonResponse, UploadResponse
from app.services.comparison_runner import run_full_comparison
from app.services.afg_merge import load_merged_afg_from_disk
from app.services.preprocessing import infer_frequency, preview_df
from app.session_store import store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _read_upload_csv_bytes(raw: bytes) -> pd.DataFrame:
    """CSV z uploadu: UTF-8 z BOM, potem Windows-1250, na końcu tolerant read."""
    for encoding in ("utf-8-sig", "utf-8", "cp1250", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(raw), encoding="utf-8", encoding_errors="replace", low_memory=False)


app = FastAPI(title="Demand Forecast Comparison API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _job_compare(session_id: str, payload: RunComparisonRequest) -> None:
    sess = store.get(session_id)
    if not sess or sess.upload_df is None:
        store.set_result(session_id, None, "Brak danych sesji.")
        store.update_progress(session_id, ProgressInfo(status="error", message="Brak danych", percent=0))
        return
    store.update_progress(session_id, ProgressInfo(status="running", message="Start", percent=0.01))
    try:
        res = run_full_comparison(
            sess.upload_df.copy(),
            payload.mapping,
            payload.plan_forecast,
            payload.rolling_step_months,
            session_id,
            progress_cb=lambda f, m: store.update_progress(
                session_id,
                ProgressInfo(status="running", message=m, percent=float(min(0.99, 0.05 + 0.9 * f))),
            ),
        )
        store.set_result(session_id, res, None)
        store.update_progress(session_id, ProgressInfo(status="completed", message="Gotowe", percent=1.0))
    except Exception as exc:
        logger.exception("Comparison failed")
        store.set_result(session_id, None, str(exc))
        store.update_progress(session_id, ProgressInfo(status="error", message=str(exc), percent=1.0))


@app.post("/upload-csv", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)) -> UploadResponse:
    raw = await file.read()
    try:
        df = _read_upload_csv_bytes(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Nie można odczytać CSV: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="Plik CSV jest pusty.")

    sess = store.create_session(raw, file.filename or "upload.csv")
    sess.upload_df = df
    cols = list(df.columns)
    has_orders = "OrdersIn" in cols
    warn: list[str] = []
    if not has_orders:
        warn.append("Nie znaleziono kolumny 'OrdersIn' — wskaż właściwą kolumnę docelową przy uruchomieniu.")

    freq = "monthly"
    for c in df.columns:
        parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().sum() >= max(5, int(0.5 * len(df))):
            freq = infer_frequency(parsed)
            break

    preview_rows = preview_df(df, 15)
    return UploadResponse(
        session_id=sess.session_id,
        columns=cols,
        row_count=int(len(df)),
        preview=preview_rows,
        warnings=warn,
        detected_frequency=freq,
        has_orders_column=has_orders,
    )


@app.post("/load-afg-bundled", response_model=UploadResponse)
def load_afg_bundled() -> UploadResponse:
    """Ładuje i scala oba pliki AFG z katalogu projektu (feature-only + train)."""
    try:
        merged, merge_warns = load_merged_afg_from_disk()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    cols = list(merged.columns)
    if "EOM" not in merged.columns:
        raise HTTPException(status_code=400, detail="Scalone dane AFG nie zawierają kolumny EOM.")
    has_orders = "OrdersIn" in cols or "KPI_OrdersIn_Qty" in cols
    warn: list[str] = list(merge_warns)
    if not has_orders:
        warn.append("Brak kolumny OrdersIn ani KPI_OrdersIn_Qty — wskaż kolumnę docelową.")

    freq = infer_frequency(pd.to_datetime(merged["EOM"], errors="coerce"))

    buf = io.StringIO()
    merged.to_csv(buf, index=False)
    raw = buf.getvalue().encode("utf-8")
    sess = store.create_session(raw, "AFG_merged.csv")
    sess.upload_df = merged

    preview_rows = preview_df(merged, 15)
    return UploadResponse(
        session_id=sess.session_id,
        columns=cols,
        row_count=int(len(merged)),
        preview=preview_rows,
        warnings=warn,
        detected_frequency=freq,
        has_orders_column=has_orders,
    )


@app.get("/preview-data")
def get_preview(session_id: str = Query(...), rows: int = 30) -> dict[str, Any]:
    sess = store.get(session_id)
    if not sess or sess.upload_df is None:
        raise HTTPException(status_code=404, detail="Nie znaleziono sesji lub danych.")
    df = sess.upload_df
    return {
        "session_id": session_id,
        "columns": list(df.columns),
        "row_count": len(df),
        "preview": preview_df(df.head(max(rows, 1)), min(rows, 200)),
    }


@app.post("/run-comparison", response_model=RunComparisonResponse)
def run_comparison(body: RunComparisonRequest, background_tasks: BackgroundTasks) -> RunComparisonResponse:
    sess = store.get(body.session_id)
    if not sess or sess.upload_df is None:
        raise HTTPException(status_code=404, detail="Nie znaleziono sesji lub danych.")

    if body.mapping.orders_column not in sess.upload_df.columns:
        raise HTTPException(status_code=400, detail=f"Brak kolumny {body.mapping.orders_column} w danych.")
    if body.mapping.date_column not in sess.upload_df.columns:
        raise HTTPException(status_code=400, detail=f"Brak kolumny daty {body.mapping.date_column}.")

    store.update_progress(body.session_id, ProgressInfo(status="running", message="Kolejka joba", percent=0.0))
    background_tasks.add_task(_job_compare, body.session_id, body)
    return RunComparisonResponse(session_id=body.session_id, job_status="accepted", result=None, error=None)


@app.get("/model-results", response_model=RunComparisonResponse)
def get_results(session_id: str = Query(...)) -> RunComparisonResponse:
    sess = store.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Nie znaleziono sesji.")
    if sess.progress.status == "completed" and sess.comparison_result is not None:
        return RunComparisonResponse(session_id=session_id, job_status="completed", result=sess.comparison_result)
    if sess.progress.status == "error":
        return RunComparisonResponse(
            session_id=session_id,
            job_status="error",
            result=None,
            error=sess.last_error or "Nieznany błąd joba",
        )
    if sess.comparison_result is not None:
        return RunComparisonResponse(session_id=session_id, job_status="completed", result=sess.comparison_result)
    if sess.progress.status == "running":
        return RunComparisonResponse(session_id=session_id, job_status="running", result=None, error=None)
    return RunComparisonResponse(session_id=session_id, job_status="idle", result=None, error=None)


@app.get("/export-results")
def export_results(session_id: str = Query(...), fmt: str = Query("csv")):
    sess = store.get(session_id)
    if not sess or not sess.comparison_result:
        raise HTTPException(status_code=404, detail="Brak wyników do eksportu.")
    if fmt not in ("csv", "xlsx"):
        raise HTTPException(status_code=400, detail="fmt musi być csv lub xlsx")
    rows = [m.model_dump() for m in sess.comparison_result.metrics]
    df = pd.DataFrame(rows)
    if fmt == "csv":
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="model_results_{session_id}.csv"'},
        )
    bio = io.BytesIO()
    df.to_excel(bio, index=False)
    bio.seek(0)
    return StreamingResponse(
        bio,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="model_results_{session_id}.xlsx"'},
    )


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/run-progress")
def run_progress(session_id: str = Query(...)) -> ProgressInfo:
    sess = store.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Nie znaleziono sesji.")
    return sess.progress
