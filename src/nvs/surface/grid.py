"""Grids, SVI/raw surfaces over (k, T), and the market target on the fixed network grid.

Units: k dimensionless log-moneyness, T years, w = sigma^2 T, iv decimal.
"""
from __future__ import annotations

import numpy as np

from nvs.surface.svi import PARAM_NAMES, svi_w


def fixed_grid(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """The fixed (k, T) network grid from the config: returns (k (nk,), T (nT,))."""
    return np.array(cfg["grid"]["k"], float), np.array(cfg["grid"]["T"], float)


def eval_grid(cfg: dict, T_max: float, nk: int = 101, nT: int = 40):
    """Dense evaluation grid used for arbitrage counts on every surface method."""
    s = cfg["smoother"]
    return np.linspace(*s["eval_k"], nk), np.linspace(s["eval_T_min"], T_max, nT)


def _interp_T(T_nodes, W_nodes, T):
    """Linear interpolation in T of rows W_nodes (nS, nk). NaN outside [T_nodes[0], T_nodes[-1]]."""
    T = np.atleast_1d(T)
    out = np.full((T.size, W_nodes.shape[1]), np.nan)
    for i, t in enumerate(T):
        if t < T_nodes[0] - 1e-12 or t > T_nodes[-1] + 1e-12:
            continue
        j = np.clip(np.searchsorted(T_nodes, t) - 1, 0, len(T_nodes) - 2)
        lam = (t - T_nodes[j]) / (T_nodes[j + 1] - T_nodes[j])
        out[i] = (1 - lam) * W_nodes[j] + lam * W_nodes[j + 1]
    return out


def svi_surface_w(slices, k, T):
    """Total variance (nT, nk) from per-expiry SVI slices (DataFrame), linear in T between them."""
    k = np.asarray(k, float)
    W = np.array([svi_w(k, *row[list(PARAM_NAMES)].values.astype(float))
                  for _, row in slices.iterrows()])
    return _interp_T(slices["T"].values, W, T)


def raw_surface_w(quotes, k, T):
    """'Raw interpolation' baseline: piecewise-linear total variance in k per expiry (flat
    extrapolation), averaged over duplicate strikes, then linear in T. No smoothing."""
    k = np.asarray(k, float)
    Ts, W = [], []
    for _, g in quotes.groupby("expiry"):
        T0 = float(g["T"].iloc[0])
        w = (g.assign(w=g["iv"] ** 2 * T0).groupby("k", as_index=False)["w"].mean())
        if len(w) < 3:
            continue
        Ts.append(T0)
        W.append(np.interp(k, w["k"].values, w["w"].values))
    order = np.argsort(Ts)
    return _interp_T(np.array(Ts)[order], np.array(W)[order], T)


def svi_iv_at_quotes(slices, quotes):
    """Model IV (decimal) at each quote using the SVI slice of its own expiry (NaN if no slice)."""
    out = np.full(len(quotes), np.nan)
    for _, s in slices.iterrows():
        m = (quotes["expiry"] == s["expiry"]).values
        w = svi_w(quotes.loc[m, "k"].values, *s[list(PARAM_NAMES)].values.astype(float))
        out[m] = np.sqrt(np.maximum(w, 1e-12) / s["T"])
    return out


def market_grid_targets(quotes, slices, k_grid, T_grid):
    """Market IV and weights on the fixed grid, built from SVI-smoothed slices of the real quotes.

    A grid cell is valid only inside the data: T within the first/last expiry and k within the
    observed quote range of both bracketing expiries (no extrapolation). Weight = 1/half-bid-ask
    width in vol (interpolated). Returns (iv (nT,nk) with NaN outside, weight (nT,nk), mask).
    """
    w = svi_surface_w(slices, k_grid, T_grid)
    iv = np.sqrt(w / np.asarray(T_grid)[:, None])
    Tn = slices["T"].values
    ba_by_exp = {e: g.sort_values("k") for e, g in quotes.groupby("expiry")}
    ba_nodes = np.array([
        np.interp(k_grid, ba_by_exp[e]["k"], ba_by_exp[e]["ba_vol"]) for e in slices["expiry"]
    ])
    ba = _interp_T(Tn, ba_nodes, T_grid)
    kmin = _interp_T(Tn, slices[["kmin"]].values.repeat(len(k_grid), 1), T_grid)
    kmax = _interp_T(Tn, slices[["kmax"]].values.repeat(len(k_grid), 1), T_grid)
    mask = np.isfinite(iv) & (k_grid[None, :] >= kmin) & (k_grid[None, :] <= kmax)
    return np.where(mask, iv, np.nan), np.where(mask, 1.0 / ba, 0.0), mask
