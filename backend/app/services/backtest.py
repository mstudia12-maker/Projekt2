from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from app.services.feature_engineering import add_time_features, feature_columns
from app.services.model_training import FittedModel, fit_all_quantile_models

logger = logging.getLogger(__name__)

HORIZON = 6

MODEL_DISPLAY = {
    "catboost": "CatBoost Quantile",
    "lightgbm": "LightGBM Quantile",
    "sklearn": "Gradient Boosting Quantile (sklearn)",
    "baseline": "Baseline (sezonowa naiwność)",
}


def display_name(kind_key: str) -> str:
    return MODEL_DISPLAY.get(kind_key, kind_key)


def _group_keys(sku_col: Optional[str], loc_col: Optional[str], df: pd.DataFrame) -> list[str]:
    keys: list[str] = []
    if sku_col and sku_col in df.columns:
        keys.append(sku_col)
    if loc_col and loc_col in df.columns:
        keys.append(loc_col)
    return keys


def _series_key_from_group(g: tuple[Any, ...] | str, keys: list[str]) -> str:
    if not keys:
        return "__global__"
    if len(keys) == 1:
        return str(g)
    return "|".join(str(x) for x in g)


def fit_models_for_origin(
    sup: pd.DataFrame,
    date_col: str,
    origin: pd.Timestamp,
    orders_col: str,
    numeric_cols: list[str],
    cat_cols: list[str],
    seed: int,
) -> tuple[list[FittedModel], list[FittedModel]]:
    train = sup[(sup[date_col] <= origin) & sup["y_next"].notna()].copy()
    if len(train) < 12:
        raise ValueError("Za mało danych treningowych dla wybranego okna czasowego.")
    X = train[numeric_cols + cat_cols]
    y = train["y_next"].astype(float).values
    m50 = fit_all_quantile_models(train, y, numeric_cols, cat_cols, 0.5, seed)
    m90 = fit_all_quantile_models(train, y, numeric_cols, cat_cols, 0.9, seed)
    return m50, m90


def baseline_forecast_horizon(
    hist: pd.DataFrame,
    date_col: str,
    orders_col: str,
    sku_col: Optional[str],
    loc_col: Optional[str],
    steps: int,
) -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    """Returns list of (forecast_month, last_row_meta) per step with baseline preds per series in meta."""
    out: list[tuple[pd.Timestamp, pd.DataFrame]] = []
    h = hist.copy()
    for _ in range(steps):
        feat = add_time_features(h, date_col, orders_col, sku_col, loc_col)
        keys = _group_keys(sku_col, loc_col, feat)
        last = feat.sort_values(date_col).groupby(keys, dropna=False).tail(1) if keys else feat.sort_values(date_col).tail(1)
        max_dt = h[date_col].max()
        next_m = (max_dt.to_period("M") + 1).to_timestamp()
        p50 = last[f"lag_{12}"].fillna(last[f"roll_mean_{3}"]).fillna(last[orders_col]).clip(lower=0)
        roll_max = (
            feat.sort_values(date_col).groupby(keys, dropna=False)[orders_col].transform(
                lambda s: s.shift(1).rolling(12, min_periods=1).max()
            )
            if keys
            else feat[orders_col].shift(1).rolling(12, min_periods=1).max()
        )
        p90 = np.maximum(p50.values, roll_max.reindex(last.index).fillna(p50).values)
        meta = last.copy()
        meta["next_month"] = next_m
        meta["pred_p50"] = p50.values
        meta["pred_p90_raw"] = p90
        meta = meta.reset_index(drop=True)
        out.append((next_m, meta))
        append_rows = pd.DataFrame(index=meta.index)
        for c in keys:
            append_rows[c] = meta[c].values
        append_rows[date_col] = next_m
        append_rows[orders_col] = meta["pred_p50"].values
        for col in h.columns:
            if col not in append_rows.columns:
                append_rows[col] = np.nan
        h = pd.concat([h, append_rows[h.columns]], ignore_index=True)
    return out


def ml_recursive_horizon(
    hist: pd.DataFrame,
    date_col: str,
    orders_col: str,
    sku_col: Optional[str],
    loc_col: Optional[str],
    models_p50: list[FittedModel],
    models_p90: list[FittedModel],
    numeric_cols: list[str],
    cat_cols: list[str],
    steps: int,
) -> list[tuple[pd.Timestamp, dict[str, dict[str, np.ndarray]], pd.DataFrame]]:
    """
    Returns per step: (forecast_month, preds keyed by model.kind -> arrays p50/p90, last_row_meta copy)
    """
    results: list[tuple[pd.Timestamp, dict[str, dict[str, np.ndarray]], pd.DataFrame]] = []
    h = hist.copy()
    # pair p50 and p90 models by kind
    by_kind: dict[str, tuple[FittedModel, FittedModel]] = {}
    for a, b in zip(models_p50, models_p90):
        ka = a.kind
        by_kind[ka] = (a, b)
    for _ in range(steps):
        feat = add_time_features(h, date_col, orders_col, sku_col, loc_col)
        keys = _group_keys(sku_col, loc_col, feat)
        last = feat.sort_values(date_col).groupby(keys, dropna=False).tail(1) if keys else feat.sort_values(date_col).tail(1)
        max_dt = h[date_col].max()
        next_m = (max_dt.to_period("M") + 1).to_timestamp()
        X = last[numeric_cols + cat_cols].copy()
        step_pred: dict[str, dict[str, np.ndarray]] = {}
        meta = last.copy()
        meta["next_month"] = next_m
        for kind, (m50, m90) in by_kind.items():
            p50 = m50.predict(X)
            p90 = m90.predict(X)
            p50 = np.clip(p50, 0, None)
            p90 = np.clip(np.maximum(p90, p50), 0, None)
            step_pred[kind] = {"p50": p50, "p90_raw": p90}
            meta[f"pred_{kind}_p50"] = p50
            meta[f"pred_{kind}_p90"] = p90
        meta = meta.reset_index(drop=True)
        results.append((next_m, step_pred, meta))
        # append rows using CatBoost chain? use average of preds? use first kind p50 for state
        chain_p50 = np.mean([step_pred[k]["p50"] for k in step_pred], axis=0)
        append_rows = pd.DataFrame()
        if keys:
            for i, c in enumerate(keys):
                append_rows[c] = last[c].values
        append_rows[date_col] = next_m
        append_rows[orders_col] = chain_p50
        for col in h.columns:
            if col not in append_rows.columns:
                append_rows[col] = np.nan
        h = pd.concat([h, append_rows[h.columns]], ignore_index=True)
    return results


def run_rolling_backtest(
    raw: pd.DataFrame,
    date_col: str,
    orders_col: str,
    sku_col: Optional[str],
    loc_col: Optional[str],
    min_train_months: int,
    step_months: int,
    seed: int,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Returns (long dataframe of predictions), {model_kind: p90_calibration_multiplier}
    """
    sup = build_supervised_frame(raw, date_col, orders_col, sku_col, loc_col)
    numeric_cols, cat_cols = feature_columns(sup, sku_col, loc_col)
    months = sorted(sup[date_col].drop_duplicates().tolist())
    h = HORIZON
    if len(months) < min_train_months + h + 2:
        raise ValueError(f"Potrzeba co najmniej {min_train_months + h + 2} miesięcy historii.")

    origins: list[pd.Timestamp] = []
    first = min_train_months - 1
    for idx in range(first, len(months) - h - 1, step_months):
        origins.append(pd.Timestamp(months[idx]))

    records: list[dict[str, Any]] = []
    p90_ratios: dict[str, list[float]] = {"catboost": [], "lightgbm": [], "sklearn": []}

    total = len(origins)
    for i, origin in enumerate(origins):
        if progress_cb:
            progress_cb(i / max(total, 1), f"Backtest {i+1}/{total} (origin {origin.date()})")
        hist = raw[raw[date_col] <= origin].copy()
        if hist.empty:
            continue
        try:
            m50, m90 = fit_models_for_origin(sup, date_col, origin, orders_col, numeric_cols, cat_cols, seed)
        except Exception as exc:
            logger.warning("Skip origin %s: %s", origin, exc)
            continue
        actuals = raw[(raw[date_col] > origin) & (raw[date_col] <= origin + pd.DateOffset(months=h))]
        # baseline
        bl_steps = baseline_forecast_horizon(hist, date_col, orders_col, sku_col, loc_col, h)
        ml_steps = ml_recursive_horizon(hist, date_col, orders_col, sku_col, loc_col, m50, m90, numeric_cols, cat_cols, h)
        keys = _group_keys(sku_col, loc_col, raw)

        for step_idx, ((fdate_bl, meta_bl), (fdate_ml, preds_ml, meta_ml)) in enumerate(zip(bl_steps, ml_steps)):
            assert fdate_bl == fdate_ml
            fdate = fdate_bl
            act_slice = actuals[actuals[date_col] == fdate]
            for _, act_row in act_slice.iterrows():
                sk = _series_key_from_group(
                    tuple(act_row[k] for k in keys) if keys else "__global__",
                    keys,
                )
                actual = float(act_row[orders_col])
                # baseline match row
                if keys:
                    mask = True
                    for k in keys:
                        mask = mask & (meta_bl[k] == act_row[k])
                    br = meta_bl[mask]
                else:
                    br = meta_bl
                if br.empty:
                    continue
                p50b = float(br["pred_p50"].iloc[0])
                p90b = float(br["pred_p90_raw"].iloc[0])
                rec = {
                    "origin": origin,
                    "forecast_date": fdate,
                    "series_key": sk,
                    "actual": actual,
                    "model": "baseline",
                    "p50": p50b,
                    "p90_raw": p90b,
                }
                records.append(rec)
                for kind, pv in preds_ml.items():
                    if keys:
                        m = meta_ml
                        sel = np.ones(len(m), dtype=bool)
                        for k in keys:
                            sel &= m[k].astype(str).values == str(act_row[k])
                        pos = np.where(sel)[0]
                        if pos.size == 0:
                            continue
                        ix = int(pos[0])
                    else:
                        ix = 0 if len(meta_ml) else -1
                    if ix < 0:
                        continue
                    p50 = float(pv["p50"][ix])
                    p90r = float(pv["p90_raw"][ix])
                    records.append(
                        {
                            "origin": origin,
                            "forecast_date": fdate,
                            "series_key": sk,
                            "actual": actual,
                            "model": kind,
                            "p50": p50,
                            "p90_raw": p90r,
                        }
                    )
                    if actual > 0 and p90r > 0:
                        p90_ratios[kind].append(actual / p90r)

    long_df = pd.DataFrame.from_records(records)
    # calibration: median ratio to hit ~90% - use isotonic simplified scale factor
    cal: dict[str, float] = {}
    if not long_df.empty:
        for kind in p90_ratios:
            arr = np.array(p90_ratios[kind], dtype=float)
            if len(arr) < 5:
                cal[kind] = 1.0
                continue
            grid = np.linspace(0.8, 1.5, 40)
            best = 1.0
            best_err = 1.0
            sub = long_df[long_df["model"] == kind]
            for g in grid:
                cov = (sub["actual"] <= sub["p90_raw"] * g).mean()
                err = abs(cov - 0.9)
                if err < best_err:
                    best_err = err
                    best = float(g)
            cal[kind] = best
        cal["baseline"] = 1.0
        long_df["p90_cal"] = long_df.apply(
            lambda r: r["p90_raw"] * cal.get(r["model"], 1.0),
            axis=1,
        )
    else:
        long_df["p90_cal"] = np.nan

    if progress_cb:
        progress_cb(1.0, "Backtest zakończony")
    return long_df, cal


def build_supervised_frame(
    df: pd.DataFrame,
    date_col: str,
    orders_col: str,
    sku_col: Optional[str],
    loc_col: Optional[str],
) -> pd.DataFrame:
    feat = add_time_features(df, date_col, orders_col, sku_col, loc_col)
    keys = _group_keys(sku_col, loc_col, feat)
    if keys:
        y_next = feat.groupby(keys, dropna=False)[orders_col].shift(-1)
    else:
        y_next = feat[orders_col].shift(-1)
    feat["y_next"] = y_next
    return feat
