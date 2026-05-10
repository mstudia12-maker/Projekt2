from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def pinball_mean(y: np.ndarray, f: np.ndarray, q: float = 0.9) -> float:
    e = y - f
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def wape(y: np.ndarray, f: np.ndarray) -> float:
    den = float(np.sum(np.abs(y)))
    if den <= 0:
        return float("nan")
    return float(np.sum(np.abs(y - f)) / den)


def mae(y: np.ndarray, f: np.ndarray) -> float:
    return float(np.mean(np.abs(y - f)))


def bias_pct(y: np.ndarray, f: np.ndarray) -> float:
    den = float(np.sum(y))
    if den <= 0:
        return float("nan")
    return float(np.sum(f - y) / den)


def coverage(y: np.ndarray, f: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    return float(np.mean(y <= f))


def plan_array(plan: str, p50: np.ndarray, p90_raw: np.ndarray, p90_cal: np.ndarray) -> np.ndarray:
    if plan == "p50":
        return p50
    if plan == "p90_raw":
        return p90_raw
    return p90_cal


def expand_series_key(df: pd.DataFrame, series_col: str, keys: list[str]) -> pd.DataFrame:
    out = df.copy()
    if not keys:
        return out

    def _parts(sk: str) -> list[str]:
        p = str(sk).split("|", maxsplit=len(keys) - 1)
        while len(p) < len(keys):
            p.append("")
        return p[: len(keys)]

    mat = np.array([_parts(x) for x in out[series_col].astype(str).values])
    for i, k in enumerate(keys):
        out[k] = mat[:, i]
    return out


def compute_service_metrics(
    model_df: pd.DataFrame,
    raw: pd.DataFrame,
    date_col: str,
    orders_col: str,
    series_col: str,
    sku_col: Optional[str],
    loc_col: Optional[str],
    fulfilled_col: Optional[str],
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float], int, int]:
    """
    Point-wise proxy on backtest dates:
    - actual lost sales proxy: max(0, orders - fulfilled)
    - predicted lost sales proxy: max(0, p50 - fulfilled)  (stan niezależny od prognozy — uproszczenie)
    - fill rate: fulfilled/orders vs fulfilled/p50
    """
    mismatch = 0
    if not fulfilled_col or fulfilled_col not in raw.columns:
        return None, None, None, None, 0, 0

    keys = [k for k in [sku_col, loc_col] if k and k in raw.columns]
    left = expand_series_key(model_df, series_col, keys)
    right = raw.copy()
    right[date_col] = pd.to_datetime(right[date_col])
    left["forecast_date"] = pd.to_datetime(left["forecast_date"])
    rcols = [date_col, fulfilled_col, orders_col] + keys
    rsub = right[rcols].copy()
    if keys:
        merge_left = ["forecast_date"] + keys
        merge_right = [date_col] + keys
        rsub = rsub.drop_duplicates(merge_right)
        merged = left.merge(rsub, left_on=merge_left, right_on=merge_right, how="left")
    else:
        rsub = rsub.drop_duplicates([date_col])
        merged = left.merge(rsub, left_on=["forecast_date"], right_on=[date_col], how="left")
    if merged[fulfilled_col].isna().any():
        mismatch += int(merged[fulfilled_col].isna().sum())

    ff_m = pd.to_numeric(merged[fulfilled_col], errors="coerce").fillna(0.0).values
    ord_m = pd.to_numeric(merged[orders_col], errors="coerce").fillna(0.0).values
    mask = np.isfinite(ff_m) & np.isfinite(ord_m)
    if not np.any(mask):
        return None, None, None, None, 0, mismatch

    act_lost = np.maximum(0.0, ord_m - ff_m)
    pred_lost = np.maximum(0.0, merged["p50"].values - ff_m)
    act_fill = np.where(ord_m > 1e-9, np.clip(ff_m / ord_m, 0.0, 1.0), np.nan)
    p50 = merged["p50"].values
    pred_fill = np.where(p50 > 1e-9, np.clip(ff_m / p50, 0.0, 1.0), np.nan)
    m2 = mask & np.isfinite(act_fill) & np.isfinite(pred_fill)
    n_service = int(np.sum(m2))
    if n_service == 0:
        return None, None, None, None, 0, mismatch
    lost_mae = float(np.mean(np.abs(act_lost[m2] - pred_lost[m2])))
    lost_bias = float(np.mean(pred_lost[m2] - act_lost[m2]))
    fill_mae = float(np.mean(np.abs(act_fill[m2] - pred_fill[m2])))
    fill_bias = float(np.mean(pred_fill[m2] - act_fill[m2]))
    return lost_mae, lost_bias, fill_mae, fill_bias, n_service, mismatch
