"""Raw SVI slices: w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + s^2)).

Units: k = ln(K/F) dimensionless; w = sigma^2 T total variance (dimensionless); T in years.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import least_squares

log = logging.getLogger(__name__)

PARAM_NAMES = ("a", "b", "rho", "m", "s")


def svi_w(k, a, b, rho, m, s):
    k = np.asarray(k, float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s**2))


def svi_derivs(k, a, b, rho, m, s):
    """Analytic (w, w', w'') of the raw SVI slice."""
    k = np.asarray(k, float)
    x = k - m
    r = np.sqrt(x**2 + s**2)
    return a + b * (rho * x + r), b * (rho + x / r), b * s**2 / r**3


def fit_svi(k, iv, T, ba_vol, n_starts=12, seed=0):
    """Fit one slice in IV space, weighted by 1/bid-ask half-width. Returns (params, rmse_volpts)."""
    k, iv, ba = (np.asarray(x, float) for x in (k, iv, ba_vol))
    w_obs = iv**2 * T
    rng = np.random.default_rng(seed)

    def resid(p):
        w = svi_w(k, *p)
        pen = np.maximum(-w, 0).sum() * 1e3  # keep total variance positive
        return np.append((np.sqrt(np.maximum(w, 1e-12) / T) - iv) / ba, pen)

    lo = [-w_obs.max(), 1e-4, -0.999, k.min() - 0.2, 1e-3]
    hi = [w_obs.max() * 2, 5.0, 0.999, k.max() + 0.2, 1.5]
    best = None
    for i in range(n_starts):
        p0 = [w_obs.min() * rng.uniform(0.3, 1), rng.uniform(0.05, 0.5) * np.sqrt(w_obs.max()),
              rng.uniform(-0.9, 0.3), rng.uniform(k.min(), k.max()) * 0.5, rng.uniform(0.02, 0.3)]
        p0 = np.clip(p0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
        try:
            sol = least_squares(resid, p0, bounds=(lo, hi), method="trf", max_nfev=400)
        except Exception as e:  # noqa: BLE001 - numerical failure of one start is not fatal
            log.debug("svi start %d failed: %s", i, e)
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        raise RuntimeError("SVI fit failed for all starts")
    iv_fit = np.sqrt(np.maximum(svi_w(k, *best.x), 1e-12) / T)
    return best.x, float(np.sqrt(np.mean((iv_fit - iv) ** 2)) * 100)


def fit_slices(quotes, min_quotes=8):
    """Fit SVI to each expiry of a quote table (columns expiry, T, k, iv, ba_vol).

    Returns a DataFrame with T, a, b, rho, m, s, rmse_volpts, n, kmin, kmax per expiry.
    Expiries with fewer than `min_quotes` quotes are skipped and logged.
    """
    import pandas as pd

    rows = []
    for exp, g in quotes.groupby("expiry"):
        if len(g) < min_quotes:
            log.warning("expiry %s skipped for SVI: only %d quotes", exp, len(g))
            continue
        T = float(g["T"].iloc[0])
        p, rmse = fit_svi(g["k"].values, g["iv"].values, T, g["ba_vol"].values)
        rows.append({"expiry": exp, "T": T, **dict(zip(PARAM_NAMES, p, strict=True)),
                     "rmse_volpts": rmse, "n": len(g), "kmin": g["k"].min(), "kmax": g["k"].max()})
    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)
