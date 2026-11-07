"""Static-arbitrage diagnostics on total-variance surfaces w(k, T).

Butterfly (convexity in strike): Gatheral's density condition
    g(k) = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2 >= 0
evaluated per maturity slice with finite differences (works for any surface method).
Calendar: total variance non-decreasing in T at fixed k.
Units: k dimensionless log-moneyness, w dimensionless total variance, T years.
"""
from __future__ import annotations

import numpy as np


def butterfly_g(w, k):
    """g(k) per slice. w shape (..., nk), k shape (nk,) increasing. Edge columns are NaN."""
    w, k = np.asarray(w, float), np.asarray(k, float)
    w1 = np.gradient(w, k, axis=-1)
    w2 = np.gradient(w1, k, axis=-1)
    g = (1 - k * w1 / (2 * w)) ** 2 - w1**2 / 4 * (1 / w + 0.25) + w2 / 2
    g[..., [0, -1]] = np.nan  # one-sided differences at the edges are unreliable
    return g


def butterfly_violations(w, k, tol=1e-6):
    """Count (slice, strike) points with g < -tol. Returns dict(count, total, min_g, mask)."""
    g = butterfly_g(w, k)
    mask = g < -tol
    return {"count": int(mask.sum()), "total": int(np.isfinite(g).sum()),
            "min_g": float(np.nanmin(g)), "mask": mask}


def calendar_violations(w, tol=1e-8):
    """w shape (nT, nk), T increasing along axis 0. Counts decreases larger than tol."""
    w = np.asarray(w, float)
    d = np.diff(w, axis=0)
    mask = d < -tol
    return {"count": int(mask.sum()), "total": int(d.size),
            "min_dw": float(d.min()) if d.size else 0.0, "mask": mask}


def check_surface(w, k, tol_b=1e-6, tol_c=1e-8):
    """Both checks on a (nT, nk) surface. Returns a flat dict of counts for reporting."""
    b, c = butterfly_violations(w, k, tol_b), calendar_violations(w, tol_c)
    return {"butterfly_violations": b["count"], "butterfly_points": b["total"],
            "min_g": b["min_g"], "calendar_violations": c["count"],
            "calendar_points": c["total"], "min_dw": c["min_dw"]}
