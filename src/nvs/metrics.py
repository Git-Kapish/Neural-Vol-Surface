"""Error metrics. All vol errors are reported in vol points (1 vol point = 0.01 absolute IV)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def region_masks(k, T) -> dict[str, np.ndarray]:
    """Boolean (nT, nk) masks of named regions of the fixed grid."""
    k, T = np.asarray(k), np.asarray(T)
    K, TT = np.meshgrid(k, T)
    return {
        "all": np.ones_like(K, bool),
        "atm (|k|<=0.04)": np.abs(K) <= 0.04 + 1e-12,
        "wings (|k|>=0.14)": np.abs(K) >= 0.14 - 1e-12,
        "short T (<=0.10y)": TT <= 0.10 + 1e-12,
        "long T (>=0.75y)": TT >= 0.75 - 1e-12,
        "wings & short T": (np.abs(K) >= 0.14 - 1e-12) & (TT <= 0.10 + 1e-12),
    }


def region_error_table(pred, true, k, T) -> pd.DataFrame:
    """Mean / p95 / max absolute IV error (vol points) by region. pred, true: (N, nT, nk) decimals."""
    err = np.abs(np.asarray(pred, float) - np.asarray(true, float)) * 100
    rows = []
    for name, m in region_masks(k, T).items():
        e = err[:, m].ravel()
        rows.append({"region": name, "mean_abs_volpts": e.mean(),
                     "p95_abs_volpts": np.percentile(e, 95), "max_abs_volpts": e.max()})
    return pd.DataFrame(rows)


def rmse_volpts(pred, true, weight=None) -> float:
    """(Optionally weighted) RMSE in vol points; NaNs in `true` are ignored."""
    d = (np.asarray(pred, float) - np.asarray(true, float)) * 100
    m = np.isfinite(d)
    w = np.ones_like(d) if weight is None else np.asarray(weight, float)
    return float(np.sqrt((w[m] * d[m] ** 2).sum() / w[m].sum()))
