from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

LAG_MONTHS = [1, 2, 3, 6, 12]
ROLL_WINDOWS = [3, 6, 12]


def add_time_features(
    df: pd.DataFrame,
    date_col: str,
    orders_col: str,
    sku_col: Optional[str] = None,
    loc_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Per time-series group (SKU+loc), sorted by date: lags, rolling stats, calendar, trend.
    """
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    group_keys: list[str] = []
    if sku_col and sku_col in work.columns:
        group_keys.append(sku_col)
    if loc_col and loc_col in work.columns:
        group_keys.append(loc_col)

    def _feat(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(date_col)
        y = g[orders_col].astype(float)
        for lag in LAG_MONTHS:
            g[f"lag_{lag}"] = y.shift(lag)
        for w in ROLL_WINDOWS:
            g[f"roll_mean_{w}"] = y.shift(1).rolling(w, min_periods=1).mean()
            g[f"roll_std_{w}"] = y.shift(1).rolling(w, min_periods=1).std()
        dt = g[date_col].dt
        g["month"] = dt.month.astype(float)
        g["quarter"] = dt.quarter.astype(float)
        g["year"] = dt.year.astype(float)
        g["trend_index"] = np.arange(len(g), dtype=float)
        g["month_sin"] = np.sin(2 * np.pi * (g["month"] - 1) / 12.0)
        g["month_cos"] = np.cos(2 * np.pi * (g["month"] - 1) / 12.0)
        return g

    if group_keys:
        parts = []
        for _, g in work.groupby(group_keys, sort=False):
            parts.append(_feat(g))
        work = pd.concat(parts, ignore_index=True)
    else:
        work = _feat(work)

    return work


def feature_columns(
    df: pd.DataFrame,
    sku_col: Optional[str] = None,
    loc_col: Optional[str] = None,
) -> tuple[list[str], list[str]]:
    """Returns (numeric_features, categorical_features)."""
    num: list[str] = []
    for lag in LAG_MONTHS:
        num.append(f"lag_{lag}")
    for w in ROLL_WINDOWS:
        num.extend([f"roll_mean_{w}", f"roll_std_{w}"])
    num.extend(["month", "quarter", "year", "trend_index", "month_sin", "month_cos"])
    cats: list[str] = []
    if sku_col and sku_col in df.columns:
        cats.append(sku_col)
    if loc_col and loc_col in df.columns:
        cats.append(loc_col)
    return num, cats
