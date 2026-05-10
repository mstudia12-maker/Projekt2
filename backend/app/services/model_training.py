from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


@dataclass
class FittedModel:
    name: str
    predict: Callable[[pd.DataFrame], np.ndarray]
    kind: str = "custom"


def _fill_na(X: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    X = X.copy()
    for c in numeric_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    return X


def _prepare_matrix(X: pd.DataFrame, numeric_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    """CatBoost/LightGBM/sklearn: kategoryczne muszą być stringami (nie float/NaN)."""
    cols = numeric_cols + cat_cols
    Xs = X[cols].copy()
    Xs = _fill_na(Xs, [c for c in numeric_cols if c in Xs.columns])
    for c in cat_cols:
        if c in Xs.columns:
            s = Xs[c].astype(object)
            Xs[c] = [
                "missing" if pd.isna(v) or v == "" else str(v)
                for v in s.tolist()
            ]
    return Xs


def _constant_prediction_models(alpha: float, c: float) -> list[FittedModel]:
    """Gdy wszystkie wartości y są takie same (lub brak wariancji) — unikaj błędów bibliotek."""

    def predict(df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), c, dtype=float)

    kinds = ["catboost", "lightgbm", "sklearn"]
    return [FittedModel(name=f"{k}_q{alpha}", predict=predict, kind=k) for k in kinds]


def fit_catboost(
    X: pd.DataFrame,
    y: np.ndarray,
    numeric_cols: list[str],
    cat_cols: list[str],
    alpha: float,
    seed: int,
) -> FittedModel:
    X_use = _prepare_matrix(X, numeric_cols, cat_cols)
    cat_idx = [X_use.columns.get_loc(c) for c in cat_cols if c in X_use.columns]
    train_params = dict(
        loss_function=f"Quantile:alpha={alpha}",
        depth=6,
        iterations=180,
        learning_rate=0.08,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        cat_features=cat_idx if cat_idx else None,
    )
    try:
        model = CatBoostRegressor(**train_params, allow_const_label=True)
    except TypeError:
        model = CatBoostRegressor(**train_params)
    model.fit(X_use, y)

    def predict(df: pd.DataFrame) -> np.ndarray:
        Xp = _prepare_matrix(df, numeric_cols, cat_cols)
        return model.predict(Xp)

    return FittedModel(name=f"catboost_q{alpha}", predict=predict, kind="catboost")


def fit_lightgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    numeric_cols: list[str],
    cat_cols: list[str],
    alpha: float,
    seed: int,
) -> FittedModel:
    import lightgbm as lgb

    X_use = _prepare_matrix(X, numeric_cols, cat_cols)
    for c in cat_cols:
        if c in X_use.columns:
            X_use[c] = X_use[c].astype("category")
    dtrain = lgb.Dataset(X_use, label=y, categorical_feature=cat_cols if cat_cols else "auto")
    params = {
        "objective": "quantile",
        "alpha": alpha,
        "learning_rate": 0.05,
        "num_leaves": 48,
        "max_depth": -1,
        "seed": seed,
        "verbose": -1,
    }
    booster = lgb.train(params, dtrain, num_boost_round=120)

    def predict(df: pd.DataFrame) -> np.ndarray:
        Xp = _prepare_matrix(df, numeric_cols, cat_cols)
        for c in cat_cols:
            if c in Xp.columns:
                Xp[c] = Xp[c].astype("category")
        return booster.predict(Xp)

    return FittedModel(name=f"lightgbm_q{alpha}", predict=predict, kind="lightgbm")


def fit_sklearn_gbr(
    X: pd.DataFrame,
    y: np.ndarray,
    numeric_cols: list[str],
    cat_cols: list[str],
    alpha: float,
    seed: int,
) -> FittedModel:
    num_pipeline = Pipeline(steps=[("impute", SimpleImputer(strategy="median"))])
    cols = numeric_cols + cat_cols
    if cat_cols:
        cat_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        pre = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, numeric_cols),
                ("cat", cat_pipeline, cat_cols),
            ]
        )
    else:
        pre = ColumnTransformer(transformers=[("num", num_pipeline, numeric_cols)])
    model = GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=120,
        max_depth=4,
        learning_rate=0.06,
        random_state=seed,
    )
    pipe = Pipeline(steps=[("prep", pre), ("model", model)])
    X_fit = _prepare_matrix(X, numeric_cols, cat_cols)
    pipe.fit(X_fit, y)

    def predict(df: pd.DataFrame) -> np.ndarray:
        return pipe.predict(_prepare_matrix(df, numeric_cols, cat_cols))

    return FittedModel(name=f"sklearn_gbr_q{alpha}", predict=predict, kind="sklearn")


def fit_all_quantile_models(
    train_frame: pd.DataFrame,
    targets: np.ndarray,
    numeric_cols: list[str],
    cat_cols: list[str],
    alpha: float,
    seed: int,
) -> list[FittedModel]:
    """Train one quantile model per library for the same alpha."""
    y = np.asarray(targets, dtype=float)
    finite = y[np.isfinite(y)]
    if finite.size == 0 or float(np.std(finite)) < 1e-10:
        c = float(np.mean(finite)) if finite.size > 0 else 0.0
        logger.info("Brak wariancji y — modele stałe (quantile ~ %.6f)", c)
        return _constant_prediction_models(alpha, c)

    models: list[FittedModel] = []
    X = train_frame
    models.append(fit_catboost(X, y, numeric_cols, cat_cols, alpha, seed))
    models.append(fit_lightgbm(X, y, numeric_cols, cat_cols, alpha, seed))
    models.append(fit_sklearn_gbr(X, y, numeric_cols, cat_cols, alpha, seed))
    return models
