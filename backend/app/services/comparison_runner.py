from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import pandas as pd

from app.config import settings
from app.schemas import ColumnMapping, ComparisonResult, ModelMetricRow
from app.services.backtest import display_name, run_rolling_backtest
from app.services.metrics import (
    bias_pct,
    compute_service_metrics,
    coverage,
    mae,
    pinball_mean,
    plan_array,
    wape,
)
from app.services.preprocessing import validate_and_clean
from app.services.ranking import compute_final_scores

logger = logging.getLogger(__name__)


def run_full_comparison(
    raw: pd.DataFrame,
    mapping: ColumnMapping,
    plan_forecast: str,
    rolling_step: int,
    session_id: str,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> ComparisonResult:
    date_c = mapping.date_column
    ord_c = mapping.orders_column
    sku_c = mapping.sku_column
    loc_c = mapping.location_column
    extra = [c for c in [mapping.sales_actual_column, mapping.fulfilled_column, mapping.inventory_column] if c]

    def _p(frac: float, msg: str) -> None:
        if progress_cb:
            progress_cb(frac, msg)

    _p(0.05, "Walidacja i preprocessing")
    cleaned, warns = validate_and_clean(
        raw,
        date_c,
        ord_c,
        sku_c,
        loc_c,
        extra_numeric_cols=extra,
    )

    _p(0.15, "Backtest toczący")
    long_df, _cal_map = run_rolling_backtest(
        cleaned,
        date_c,
        ord_c,
        sku_c,
        loc_c,
        settings.min_train_months,
        rolling_step,
        settings.random_state,
        progress_cb=lambda f, m: _p(0.15 + 0.55 * f, m),
    )

    if long_df.empty:
        raise ValueError("Brak wyników backtestingu — sprawdź długość historii i mapowanie kolumn.")

    models = sorted(long_df["model"].unique())
    rows: list[dict[str, Any]] = []
    missing_expl: list[str] = []

    if mapping.fulfilled_column is None or mapping.fulfilled_column not in cleaned.columns:
        missing_expl.append(
            "Metryki LostSales / FillRate wymagają kolumny zrealizowanej ilości (`fulfilled_column` w mapowaniu) obecnej w CSV."
        )
    if mapping.lead_time_months is None:
        missing_expl.append(
            "Lead time (miesiące) nie został podany — metryki LT traktujemy jako porównawcze (punktowe na horyzoncie backtestu)."
        )

    common_keys = ["origin", "forecast_date", "series_key"]
    pivot_counts: dict[str, int] = {}
    for m in models:
        sub = long_df[long_df["model"] == m].dropna(subset=["p50", "p90_raw", "actual"])
        pivot_counts[m] = len(sub)

    min_common = min(pivot_counts.values()) if pivot_counts else 0
    # Common intersection of keys where all models exist
    sets = []
    for m in models:
        s = set(
            zip(
                long_df.loc[long_df["model"] == m, "origin"],
                long_df.loc[long_df["model"] == m, "forecast_date"],
                long_df.loc[long_df["model"] == m, "series_key"],
            )
        )
        sets.append(s)
    common_set = set.intersection(*sets) if sets else set()
    n_common = len(common_set)

    mismatch_total = 0

    for m in models:
        sub = long_df[long_df["model"] == m].copy()
        if n_common > 0:
            key_tuples = sub[common_keys].apply(tuple, axis=1)
            sub_common = sub.loc[key_tuples.isin(common_set)].copy()
        else:
            sub_common = sub
        n_back = len(sub_common.dropna(subset=["p50", "actual"]))

        y = sub_common["actual"].astype(float).values
        p50 = sub_common["p50"].astype(float).values
        pr = sub_common["p90_raw"].astype(float).values
        pc = sub_common["p90_cal"].astype(float).values
        plan = plan_array(plan_forecast, p50, pr, pc)

        w50 = wape(y, p50)
        m50 = mae(y, p50)
        b50 = bias_pct(y, p50)
        covr = coverage(y, pr)
        covc = coverage(y, pc)
        pin = pinball_mean(y, pc, 0.9)
        pw = wape(y, plan)
        pb = bias_pct(y, plan)

        lost_mae, lost_bias, fill_mae, fill_bias, n_serv, mismatch = compute_service_metrics(
            sub_common,
            cleaned,
            date_c,
            ord_c,
            "series_key",
            sku_c,
            loc_c,
            mapping.fulfilled_column,
        )
        if mapping.fulfilled_column is None:
            lost_mae = lost_bias = fill_mae = fill_bias = None
            n_serv = 0

        mismatch_total += mismatch

        rows.append(
            {
                "model": m,
                "n_backtest_common": n_common if n_common > 0 else n_back,
                "wape_p50": w50,
                "mae_p50": m50,
                "bias_pct_p50": b50,
                "coverage_p90_raw": covr,
                "coverage_p90_cal": covc,
                "pinball_p90_cal": pin,
                "plan_wape": pw,
                "plan_bias_pct": pb,
                "n_service_common": n_serv,
                "lost_sales_lt_mae": lost_mae,
                "lost_sales_lt_bias": lost_bias,
                "fill_rate_lt_mae": fill_mae,
                "fill_rate_lt_bias": fill_bias,
                "actual_mismatch_count": mismatch,
            }
        )

    tdf = pd.DataFrame(rows).set_index("model")
    scores = compute_final_scores(tdf)
    tdf["final_score"] = scores
    order = scores.sort_values(ascending=False).index.tolist()
    rank_map = {mod: i + 1 for i, mod in enumerate(order)}
    best = order[0] if order else ""

    metric_rows: list[ModelMetricRow] = []
    for r in rows:
        mm = r["model"]
        metric_rows.append(
            ModelMetricRow(
                model=display_name(mm),
                n_backtest_common=int(r["n_backtest_common"]),
                wape_p50=r["wape_p50"],
                mae_p50=r["mae_p50"],
                bias_pct_p50=r["bias_pct_p50"],
                coverage_p90_raw=r["coverage_p90_raw"],
                coverage_p90_cal=r["coverage_p90_cal"],
                pinball_p90_cal=r["pinball_p90_cal"],
                plan_wape=r["plan_wape"],
                plan_bias_pct=r["plan_bias_pct"],
                n_service_common=int(r["n_service_common"]),
                lost_sales_lt_mae=r["lost_sales_lt_mae"],
                lost_sales_lt_bias=r["lost_sales_lt_bias"],
                fill_rate_lt_mae=r["fill_rate_lt_mae"],
                fill_rate_lt_bias=r["fill_rate_lt_bias"],
                actual_mismatch_count=int(r["actual_mismatch_count"]),
                final_score=float(scores.loc[mm]) if mm in scores.index else None,
                rank=int(rank_map.get(mm, 0)),
                recommendation=("Rekomendowany do wdrożenia" if mm == best else ""),
            )
        )

    interpretation = (
        f"Najwyżej oceniony model to {display_name(best)} według zbalansowanego scoringu "
        f"(WAPE P50, pokrycie P90 po kalibracji, strata pinball, plan {plan_forecast}). "
        f"Porównanie opiera się na {n_common} wspólnych punktach backtestingu. "
        + (" ".join(warns) if warns else "")
    )

    return ComparisonResult(
        session_id=session_id,
        metrics=sorted(metric_rows, key=lambda x: x.rank),
        ranking_order=[display_name(m) for m in order],
        recommended_model=display_name(best),
        interpretation=interpretation,
        missing_metric_explanations=missing_expl,
    )
