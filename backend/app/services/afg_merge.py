from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from app.config import settings

logger = logging.getLogger(__name__)

KEYS = ["ProductID", "EOM"]


def _project_root() -> Path:
    if settings.afg_project_root and str(settings.afg_project_root).strip():
        return Path(str(settings.afg_project_root)).resolve()
    # backend/app/services/afg_merge.py → parents[3] = katalog projektu (Aplikacja_4)
    return Path(__file__).resolve().parents[3]


def afg_csv_paths() -> Tuple[Path, Path]:
    root = _project_root()
    f = root / settings.afg_feature_csv
    t = root / settings.afg_train_csv
    return f, t


def _is_missing_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.isna()
    s_str = s.astype(str).str.strip()
    return s.isna() | (s_str == "") | (s_str.str.lower() == "nan")


def merge_afg_dataframes(df_feature: pd.DataFrame, df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Scala wiersze po ProductID + EOM. Wspólne kolumny (poza kluczem): wartość z TRAIN,
    jeśli jest; w przeciwnym razie z pliku feature-only. Kolumny tylko w jednym pliku
    zostają bez zmian.
    """
    for k in KEYS:
        if k not in df_feature.columns or k not in df_train.columns:
            raise ValueError(f"Oba pliki muszą zawierać kolumny {KEYS}, brakuje {k}.")

    f = df_feature.copy()
    t = df_train.copy()
    f["EOM"] = pd.to_datetime(f["EOM"], errors="coerce")
    t["EOM"] = pd.to_datetime(t["EOM"], errors="coerce")

    overlap = (set(f.columns) & set(t.columns)) - set(KEYS)
    merged = f.merge(t, on=KEYS, how="outer", suffixes=("_featureonly", "_train"))

    for col in sorted(overlap):
        c_f = f"{col}_featureonly"
        c_t = f"{col}_train"
        if c_f not in merged.columns:
            continue
        if c_t not in merged.columns:
            merged.rename(columns={c_f: col}, inplace=True)
            continue
        v_tr = merged[c_t]
        v_f = merged[c_f]
        merged[col] = v_tr.where(~_is_missing_series(v_tr), v_f)
        merged.drop(columns=[c_f, c_t], inplace=True)

    merged = merged.sort_values(KEYS).reset_index(drop=True)

    # Alias pod domyślne mapowanie aplikacji (OrdersIn)
    if "KPI_OrdersIn_Qty" in merged.columns and "OrdersIn" not in merged.columns:
        merged["OrdersIn"] = pd.to_numeric(merged["KPI_OrdersIn_Qty"], errors="coerce")

    return merged


def load_merged_afg_from_disk() -> tuple[pd.DataFrame, list[str]]:
    fp, tp = afg_csv_paths()
    if not fp.is_file():
        raise FileNotFoundError(f"Brak pliku feature-only: {fp}")
    if not tp.is_file():
        raise FileNotFoundError(f"Brak pliku train: {tp}")

    df_f = pd.read_csv(fp)
    df_t = pd.read_csv(tp)
    merged = merge_afg_dataframes(df_f, df_t)

    warns: list[str] = []
    warns.append(
        f"Polaczono AFG: {fp.name} ({len(df_f)} w.) + {tp.name} ({len(df_t)} w.) -> {len(merged)} wierszy."
    )
    if merged["EOM"].isna().any():
        n = int(merged["EOM"].isna().sum())
        warns.append(f"Uwaga: {n} wierszy bez poprawnej daty EOM — mogą zostać odrzucone w preprocesingu.")

    return merged, warns
