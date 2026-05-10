from __future__ import annotations

import logging
import warnings
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi.encoders import jsonable_encoder

logger = logging.getLogger(__name__)


def _parse_dates(series: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(series, errors="coerce", utc=False)


def validate_and_clean(
    df: pd.DataFrame,
    date_col: str,
    orders_col: str,
    sku_col: Optional[str] = None,
    loc_col: Optional[str] = None,
    extra_numeric_cols: Optional[list[str]] = None,
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Clean dataframe: parse dates, coerce numeric orders, drop bad rows, dedupe, aggregate to monthly.
    """
    warnings_out: list[str] = []
    if orders_col not in df.columns:
        raise ValueError(f"Brak kolumny docelowej '{orders_col}' w pliku.")
    if date_col not in df.columns:
        raise ValueError(f"Brak kolumny daty '{date_col}'.")

    work = df.copy()
    work[date_col] = _parse_dates(work[date_col])
    invalid_dates = work[date_col].isna().sum()
    if invalid_dates:
        warnings_out.append(f"Usunięto {invalid_dates} wierszy z niepoprawną datą.")
    work = work.dropna(subset=[date_col])

    work[orders_col] = pd.to_numeric(work[orders_col], errors="coerce")
    nan_orders = work[orders_col].isna().sum()
    if nan_orders:
        warnings_out.append(f"Brakujące wartości Orders: wypełniono 0 dla {nan_orders} wierszy.")
        work[orders_col] = work[orders_col].fillna(0.0)

    neg = (work[orders_col] < 0).sum()
    if neg:
        warnings_out.append(f"Wykryto {neg} ujemnych wartości popytu — ustawiono na 0.")
        work.loc[work[orders_col] < 0, orders_col] = 0.0

    group_keys: list[str] = []
    if sku_col and sku_col in work.columns:
        group_keys.append(sku_col)
    if loc_col and loc_col in work.columns:
        group_keys.append(loc_col)

    numeric_extras = [c for c in (extra_numeric_cols or []) if c in work.columns and c != orders_col]
    for c in numeric_extras:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    work = work.sort_values([date_col] + group_keys)
    before = len(work)
    subset = [date_col] + group_keys
    work = work.drop_duplicates(subset=subset, keep="last")
    dup = before - len(work)
    if dup:
        warnings_out.append(f"Usunięto {dup} duplikatów (ta sama data + identyfikatory).")

    freq = infer_frequency(work[date_col])
    if freq in ("daily", "weekly"):
        warnings_out.append(f"Wykryto częstotliwość {freq} — agregacja sumą do miesięcy.")
        work["period"] = work[date_col].dt.to_period("M").dt.to_timestamp()
        agg_dict: dict[str, str] = {orders_col: "sum"}
        for c in numeric_extras:
            agg_dict[c] = "sum"
        gb_cols = ["period"] + group_keys
        work = work.groupby(gb_cols, dropna=False).agg(agg_dict).reset_index()
        work = work.rename(columns={"period": date_col})
    else:
        work[date_col] = work[date_col].dt.to_period("M").dt.to_timestamp()

    work = work.sort_values([date_col] + group_keys).reset_index(drop=True)

    for ck in group_keys:
        if ck in work.columns:
            work[ck] = work[ck].fillna("missing").astype(str).replace("", "missing")

    return work, warnings_out


def infer_frequency(date_series: pd.Series) -> str:
    s = date_series.sort_values().drop_duplicates().diff().dropna()
    if s.empty:
        return "monthly"
    med = s.median()
    if med <= pd.Timedelta(days=1.5):
        return "daily"
    if med <= pd.Timedelta(days=8):
        return "weekly"
    return "monthly"


def preview_df(df: pd.DataFrame, n: int = 15) -> list[dict[str, Any]]:
    view = df.head(n).copy()
    for c in view.columns:
        if pd.api.types.is_datetime64_any_dtype(view[c]):
            view[c] = view[c].dt.strftime("%Y-%m-%d")
    # numpy / pd.NA / mieszane typy w object — bezpieczna serializacja JSON
    return jsonable_encoder(view.to_dict(orient="records"))
