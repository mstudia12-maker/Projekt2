"""
Demand Forecast Lab — Streamlit UI.

Uruchom z katalogu głównego projektu:
  py -m pip install -r requirements.txt
  py -m streamlit run streamlit_app/app.py
"""

from __future__ import annotations

import io
import re
import sys
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

STREAMLIT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STREAMLIT_DIR.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.schemas import ColumnMapping  # noqa: E402
from app.services.afg_merge import load_merged_afg_from_disk  # noqa: E402
from app.services.comparison_runner import run_full_comparison  # noqa: E402

st.set_page_config(page_title="Demand Forecast Lab", layout="wide")


def _read_upload_csv_bytes(raw: bytes) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp1250", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(raw), encoding="utf-8", encoding_errors="replace", low_memory=False)


def _guess_date_col(cols: list[str]) -> str:
    for c in cols:
        if re.search(r"date|month|data|time|eom", c, re.I):
            return c
    return cols[0] if cols else ""


def _guess_orders_col(cols: list[str]) -> str:
    for c in cols:
        if re.search(r"order|demand|qty|volume|sales|popyt", c, re.I):
            return c
    return cols[0] if cols else ""


def _init_state() -> None:
    defaults = {
        "df": None,
        "upload_warnings": [],
        "comparison_result": None,
        "run_error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main() -> None:
    _init_state()

    st.title("Porównanie prognoz kwantylowych OrdersIn (6 mies.)")
    st.caption(
        "CatBoost, LightGBM, sklearn GBR (quantile) + baseline — rolling backtest. "
        "Dane i logika jak w backendzie FastAPI (`backend/app/services/`)."
    )

    with st.sidebar:
        st.header("Parametry porównania")
        plan_forecast = st.selectbox(
            "Prognoza planistyczna (metryki PLAN)",
            options=["p50", "p90_raw", "p90_cal"],
            index=0,
            format_func=lambda x: {"p50": "P50", "p90_raw": "P90 surowe", "p90_cal": "P90 skalibrowane"}[x],
        )
        rolling_step = st.number_input("Krok rolling (miesiące)", min_value=1, max_value=12, value=6, step=1)
        lead_raw = st.text_input("Lead time (miesiące, opcjonalnie)", value="3")
        lead_t = lead_raw.strip()
        lead_time_months = int(lead_t) if lead_t.isdigit() else None

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        uploaded = st.file_uploader("Wgraj CSV", type=["csv"])
        if uploaded is not None:
            try:
                raw = uploaded.getvalue()
                st.session_state.df = _read_upload_csv_bytes(raw)
                st.session_state.upload_warnings = []
                st.session_state.comparison_result = None
                st.session_state.run_error = None
                cols = list(st.session_state.df.columns)
                st.success(f"Wczytano {len(st.session_state.df):,} wierszy.")
            except Exception as e:
                st.error(f"Nie można odczytać CSV: {e}")

    with col_up2:
        if st.button("Załaduj oba pliki AFG z katalogu projektu"):
            try:
                merged, warns = load_merged_afg_from_disk()
                st.session_state.df = merged
                st.session_state.upload_warnings = list(warns)
                st.session_state.comparison_result = None
                st.session_state.run_error = None
                st.success(f"AFG: {len(merged):,} wierszy po scaleniu.")
            except FileNotFoundError as e:
                st.error(str(e))
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Błąd AFG: {e}")

    df: pd.DataFrame | None = st.session_state.df
    if df is None or df.empty:
        st.info("Wgraj CSV lub załaduj AFG, żeby kontynuować.")
        return

    if st.session_state.upload_warnings:
        with st.expander("Komunikaty importu", expanded=False):
            for w in st.session_state.upload_warnings:
                st.write(w)

    cols = list(df.columns)
    st.subheader("Mapowanie kolumn")
    c1, c2, c3 = st.columns(3)
    with c1:
        date_col = st.selectbox("Data (miesiąc)", cols, index=max(0, cols.index(_guess_date_col(cols)) if _guess_date_col(cols) in cols else 0))
        orders_col = st.selectbox("OrdersIn / popyt", cols, index=max(0, cols.index(_guess_orders_col(cols)) if _guess_orders_col(cols) in cols else 0))
    with c2:
        sku_opts = ["— brak —"] + cols
        sku_def = "ProductID" if "ProductID" in cols else "— brak —"
        sku_ix = sku_opts.index(sku_def) if sku_def in sku_opts else 0
        sku_choice = st.selectbox("SKU / produkt", sku_opts, index=sku_ix)
        loc_def = "ProductLine" if "ProductLine" in cols else "— brak —"
        loc_opts = ["— brak —"] + cols
        loc_ix = loc_opts.index(loc_def) if loc_def in loc_opts else 0
        loc_choice = st.selectbox("Lokalizacja", loc_opts, index=loc_ix)
    with c3:
        ful_opts = ["— brak —"] + cols
        ful_def = "KPI_SalesQty" if "KPI_SalesQty" in cols else "FulfilledQty" if "FulfilledQty" in cols else "— brak —"
        ful_ix = ful_opts.index(ful_def) if ful_def in ful_opts else 0
        fulfilled_choice = st.selectbox("Zrealizowana ilość", ful_opts, index=ful_ix)

    sku_col = None if sku_choice == "— brak —" else sku_choice
    loc_col = None if loc_choice == "— brak —" else loc_choice
    fulfilled_col = None if fulfilled_choice == "— brak —" else fulfilled_choice

    with st.expander("Podgląd danych (15 pierwszych wierszy)"):
        st.dataframe(df.head(15), use_container_width=True)

    if st.button("Uruchom porównanie modeli", type="primary"):
        if orders_col not in df.columns or date_col not in df.columns:
            st.error("Nieprawidłowe mapowanie kolumn.")
        else:
            st.session_state.run_error = None
            st.session_state.comparison_result = None
            mapping = ColumnMapping(
                orders_column=orders_col,
                date_column=date_col,
                sku_column=sku_col,
                location_column=loc_col,
                fulfilled_column=fulfilled_col,
                lead_time_months=lead_time_months,
            )
            progress = st.progress(0.0, text="Start…")
            status = st.empty()

            def progress_cb(frac: float, msg: str) -> None:
                try:
                    progress.progress(min(1.0, max(0.0, 0.05 + 0.9 * frac)), text=msg)
                except TypeError:
                    progress.progress(min(1.0, max(0.0, 0.05 + 0.9 * frac)))
                status.caption(msg)

            try:
                result = run_full_comparison(
                    df.copy(),
                    mapping,
                    plan_forecast,
                    int(rolling_step),
                    str(uuid.uuid4()),
                    progress_cb=progress_cb,
                )
                st.session_state.comparison_result = result
                progress.progress(1.0, text="Gotowe.")
                status.caption("")
            except Exception as e:
                st.session_state.run_error = str(e)
                progress.empty()
                status.empty()

    if st.session_state.run_error:
        st.error(st.session_state.run_error)

    res = st.session_state.comparison_result
    if res is not None:
        st.divider()
        st.subheader("Wyniki")
        b1, b2, b3 = st.columns(3)
        best = next((m for m in res.metrics if m.rank == 1), res.metrics[0] if res.metrics else None)
        with b1:
            st.metric("Rekomendacja", res.recommended_model or "—")
        with b2:
            st.metric("Najlepszy score", f"{best.final_score:.3f}" if best and best.final_score is not None else "—")
        with b3:
            st.metric("Punkty backtestu (N common)", str(best.n_backtest_common) if best else "—")

        if res.missing_metric_explanations:
            with st.expander("Ograniczenia metryk", expanded=False):
                for line in res.missing_metric_explanations:
                    st.write(line)

        rows = [m.model_dump() for m in res.metrics]
        metrics_df = pd.DataFrame(rows)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        st.subheader("Ranking")
        for i, name in enumerate(res.ranking_order, start=1):
            st.write(f"{i}. {name}")

        st.subheader("Interpretacja")
        st.markdown(res.interpretation)

        csv_buf = metrics_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Pob CSV — metryki",
            data=csv_buf,
            file_name="model_metrics.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
