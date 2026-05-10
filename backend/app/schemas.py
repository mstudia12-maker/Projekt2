from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ColumnMapping(BaseModel):
    orders_column: str = Field(..., description="Target column, e.g. OrdersIn")
    date_column: str
    sku_column: Optional[str] = None
    location_column: Optional[str] = None
    # Optional service / supply-chain columns
    sales_actual_column: Optional[str] = None
    fulfilled_column: Optional[str] = None
    inventory_column: Optional[str] = None
    lead_time_months: Optional[int] = Field(
        default=None, description="Constant lead time in months for LT metrics"
    )


class UploadResponse(BaseModel):
    session_id: str
    columns: list[str]
    row_count: int
    preview: list[dict[str, Any]]
    warnings: list[str] = []
    detected_frequency: str = "monthly"
    has_orders_column: bool = True


class RunComparisonRequest(BaseModel):
    session_id: str
    mapping: ColumnMapping
    plan_forecast: Literal["p50", "p90_raw", "p90_cal"] = "p50"
    rolling_step_months: int = 1


class ProgressInfo(BaseModel):
    status: Literal["idle", "running", "completed", "error"]
    message: str = ""
    percent: float = 0.0


class ModelMetricRow(BaseModel):
    model: str
    n_backtest_common: int
    wape_p50: Optional[float]
    mae_p50: Optional[float]
    bias_pct_p50: Optional[float]
    coverage_p90_raw: Optional[float]
    coverage_p90_cal: Optional[float]
    pinball_p90_cal: Optional[float]
    plan_wape: Optional[float]
    plan_bias_pct: Optional[float]
    n_service_common: int
    lost_sales_lt_mae: Optional[float]
    lost_sales_lt_bias: Optional[float]
    fill_rate_lt_mae: Optional[float]
    fill_rate_lt_bias: Optional[float]
    actual_mismatch_count: int
    final_score: Optional[float]
    rank: int
    recommendation: str = ""


class ComparisonResult(BaseModel):
    session_id: str
    metrics: list[ModelMetricRow]
    ranking_order: list[str]
    recommended_model: str
    interpretation: str
    missing_metric_explanations: list[str]


class RunComparisonResponse(BaseModel):
    session_id: str
    job_status: str
    result: Optional[ComparisonResult] = None
    error: Optional[str] = None
