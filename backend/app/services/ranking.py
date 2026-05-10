from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _minmax_inv(scores: pd.Series) -> pd.Series:
    """Lower raw is better -> higher normalized score."""
    s = scores.astype(float)
    if s.notna().sum() < 2:
        return pd.Series(1.0, index=s.index)
    lo = float(np.nanmin(s.values))
    hi = float(np.nanmax(s.values))
    if hi - lo < 1e-12:
        return pd.Series(1.0, index=s.index)
    norm = (hi - s) / (hi - lo)
    return norm.clip(0, 1)


def _minmax_abs_inv(scores: pd.Series) -> pd.Series:
    """Closer to 0 is better: use abs then lower is better."""
    return _minmax_inv(scores.abs())


def _coverage_score(scores: pd.Series, target: float = 0.9) -> pd.Series:
    """Closer to target is better."""
    dist = (scores.astype(float) - target).abs()
    return _minmax_inv(dist)


def compute_final_scores(table: pd.DataFrame) -> pd.Series:
    """
    table rows = models, columns = metric raw values (aggregated).
    Weighted sum of normalized metrics. Diagnostic cols excluded.
    """
    idx = table.index
    parts: list[pd.Series] = []
    weights: list[float] = []

    def add(col: str, fn, w: float) -> None:
        if col not in table.columns:
            return
        raw = table[col].astype(float)
        if raw.isna().all():
            return
        s = fn(raw.fillna(raw.median()))
        parts.append(s)
        weights.append(w)

    add("wape_p50", _minmax_inv, 1.0)
    add("mae_p50", _minmax_inv, 0.8)
    add("bias_pct_p50", _minmax_abs_inv, 0.5)
    add("coverage_p90_raw", _coverage_score, 0.4)
    add("coverage_p90_cal", _coverage_score, 0.6)
    add("pinball_p90_cal", _minmax_inv, 0.8)
    add("plan_wape", _minmax_inv, 1.0)
    add("plan_bias_pct", _minmax_abs_inv, 0.5)
    add("lost_sales_lt_mae", _minmax_inv, 0.4)
    add("lost_sales_lt_bias", _minmax_abs_inv, 0.3)
    add("fill_rate_lt_mae", _minmax_inv, 0.4)
    add("fill_rate_lt_bias", _minmax_abs_inv, 0.3)

    if not parts:
        return pd.Series(0.0, index=idx)

    mat = pd.concat(parts, axis=1)
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    scores = (mat * w).sum(axis=1, min_count=1)
    return scores
