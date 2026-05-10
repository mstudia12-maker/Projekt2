"""
Generator sztucznych danych miesięcznych: SKU x data, OrdersIn + opcjonalnie FulfilledQty.
Uruchom: python generate_sample_data.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--months", type=int, default=48)
    p.add_argument("--skus", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data" / "sample_monthly.csv")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    start = pd.Timestamp("2020-01-01")
    rows = []
    for m in range(args.months):
        dt = (start + pd.DateOffset(months=m)).to_period("M").to_timestamp()
        for s in range(args.skus):
            sku = f"SKU-{s+1:02d}"
            seas = 80 + 40 * np.sin(2 * np.pi * (m % 12) / 12.0)
            t = 0.12 * m
            base = seas + t + rng.normal(0, 12)
            orders = float(max(0.0, base))
            fulfilled = float(max(0.0, min(orders, orders * rng.uniform(0.85, 1.05))))
            rows.append(
                {
                    "MonthStart": dt.strftime("%Y-%m-%d"),
                    "SKU": sku,
                    "Location": "PL-WAW",
                    "OrdersIn": round(orders, 2),
                    "FulfilledQty": round(fulfilled, 2),
                }
            )
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Zapisano {len(df)} wierszy -> {args.out}")


if __name__ == "__main__":
    main()
