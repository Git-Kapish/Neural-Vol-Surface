"""Heston (1993) pricing via the characteristic function, vectorised over parameter batches.

Model (forward measure, so rates/dividends drop out once F is given):
    dS/S = sqrt(v) dW1,  dv = kappa (vbar - v) dt + sigma sqrt(v) dW2,  d<W1,W2> = rho dt.
theta = (v0, kappa, vbar, sigma, rho): v0, vbar annualised variances (decimal^2), kappa 1/year,
sigma vol-of-vol (1/sqrt(year)), rho dimensionless. k = ln(K/F), T in years.

Characteristic function: "little Heston trap" form (Albrecher et al. 2007), which is continuous
in the principal branch. Pricing: Lewis (2001) single-integral formula,
    c(k, T) = E[(F_T - K)^+] / F = 1 - e^{k/2}/pi * int_0^inf Re[e^{-iuk} phi(u - i/2)] / (u^2+1/4) du,
with phi(u) = E[exp(i u ln(F_T/F))]. Integration: Gauss-Legendre on [0, U] with U chosen so the
envelope |phi(u - i/2)|/(u^2+1/4) is below 1e-12 (probed per parameter set, see `_cutoffs`) and
n = clip(2U + 200, 400, 6000) nodes. Accuracy is checked against a published benchmark and an
independent quadrature in tests/test_heston.py. OTM prices are returned directly; ITM by parity.
"""
from __future__ import annotations

import logging

import numpy as np

from nvs.pricing.implied_vol import OK, implied_vol

log = logging.getLogger(__name__)

PARAM_NAMES = ("v0", "kappa", "vbar", "sigma", "rho")


def heston_cf(u, T, v0, kappa, vbar, sigma, rho):
    """phi(u) = E[exp(i u ln(F_T/F))]; u may be complex. All args broadcast together."""
    iu = 1j * u
    b = kappa - rho * sigma * iu
    d = np.sqrt(b**2 + sigma**2 * (iu + u**2))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    D = (b - d) / sigma**2 * (1 - e) / (1 - g * e)
    C = kappa * vbar / sigma**2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    return np.exp(C + D * v0)


def _cutoffs(theta, T, tol=1e-12):
    """Per-parameter integration cutoff U_i: first probe point beyond which the envelope
    |phi(u - i/2)| / (u^2 + 1/4) (max over T) stays below `tol`. Probe: 200 log-spaced points on
    [0.5, 6000]. The Heston CF decays like exp(-c u), so U must grow for small c."""
    probe = np.geomspace(0.5, 6000.0, 200)
    v0, kap, vb, sg, rho = (theta[:, i][:, None, None] for i in range(5))
    env = np.abs(heston_cf(probe - 0.5j, np.asarray(T, float)[None, :, None], v0, kap, vb, sg, rho))
    env = (env / (probe**2 + 0.25)).max(axis=1)  # (N, nprobe)
    last = np.where(env > tol, np.arange(probe.size), -1).max(axis=1)
    if (last >= probe.size - 1).any():
        log.warning("Heston integrand not below %.0e at u=6000 for %d sets", tol,
                    int((last >= probe.size - 1).sum()))
    return np.maximum(probe[np.minimum(last + 1, probe.size - 1)] * 1.15, 50.0)


def _gl(U):
    n = int(np.clip(2 * U + 200, 400, 6000))
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * U * (x + 1), 0.5 * U * w


def heston_otm_price(theta, k, T, budget=2_000_000):
    """Undiscounted OTM price divided by F: call if k >= 0 else put. Shape (N, nT, nk).

    Parameters are sorted by required cutoff and processed in chunks that share one node set
    (chunk size set by `budget` complex entries), so cheap parameter sets do not pay for hard ones.
    """
    theta = np.atleast_2d(np.asarray(theta, float))
    k, T = np.atleast_1d(np.asarray(k, float)), np.atleast_1d(np.asarray(T, float))
    Ui = _cutoffs(theta, T)
    order = np.argsort(Ui)
    out = np.empty((theta.shape[0], T.size, k.size))
    s = 0
    while s < len(order):
        n_est = int(np.clip(2 * Ui[order[min(s + 63, len(order) - 1)]] + 200, 400, 6000))
        size = max(8, budget // (T.size * n_est))
        idx = order[s:s + size]
        u, wq = _gl(float(Ui[idx].max()))
        cosuk, sinuk = np.cos(np.outer(k, u)), np.sin(np.outer(k, u))  # (nk, nu)
        v0, kap, vb, sg, rho = (theta[idx, i][:, None, None] for i in range(5))
        phi = heston_cf(u - 0.5j, T[None, :, None], v0, kap, vb, sg, rho)  # (n, nT, nu)
        f = wq / (u**2 + 0.25)
        integ = np.einsum("ntu,ku->ntk", phi.real * f, cosuk) + np.einsum(
            "ntu,ku->ntk", phi.imag * f, sinuk)
        call = 1 - np.exp(k / 2) / np.pi * integ
        out[idx] = np.where(k >= 0, call, call - (1 - np.exp(k)))
        s += size
    return out


def heston_call(theta, F, K, T, df=1.0):
    """Discounted European call price for scalar/array K (same T). Convenience for tests."""
    k = np.log(np.atleast_1d(K) / F)
    otm = heston_otm_price(np.asarray(theta)[None, :], k, [T])[0, 0]
    call = np.where(k >= 0, otm, otm + (1 - np.exp(k)))
    return df * F * call


def heston_iv(theta, k, T):
    """Black-Scholes implied vol (decimal) of the Heston OTM prices; NaN where inversion fails.
    Shape (N, nT, nk). Failures are counted/logged by `implied_vol`."""
    theta = np.atleast_2d(theta)
    k, T = np.atleast_1d(np.asarray(k, float)), np.atleast_1d(np.asarray(T, float))
    price = heston_otm_price(theta, k, T)
    K = np.broadcast_to(np.exp(k), price.shape)
    Tt = np.broadcast_to(T[None, :, None], price.shape)
    res = implied_vol(price, 1.0, K, Tt, np.broadcast_to(k >= 0, price.shape), label="heston")
    return np.where(res.status == OK, res.iv, np.nan)


def sample_params(n, bounds: dict, seed=0):
    """Scrambled Sobol sample of theta in the config box. Returns (n, 5) in PARAM_NAMES order."""
    import warnings

    from scipy.stats import qmc

    lo = np.array([bounds[p][0] for p in PARAM_NAMES])
    hi = np.array([bounds[p][1] for p in PARAM_NAMES])
    with warnings.catch_warnings():  # balance warning for non-power-of-2 n is irrelevant here
        warnings.simplefilter("ignore")
        return qmc.scale(qmc.Sobol(5, scramble=True, seed=seed).random(n), lo, hi)


def build_training_set(cfg: dict, n: int, seed: int):
    """Sample theta, price the fixed (k, T) grid, drop samples with any failed IV (counted).

    Returns theta (m, 5) and iv (m, nT, nk) decimal vols, with m <= n.
    """
    k = np.array(cfg["grid"]["k"], float)
    T = np.array(cfg["grid"]["T"], float)
    theta = sample_params(n, cfg["heston"]["bounds"], seed)
    iv = heston_iv(theta, k, T)
    good = np.isfinite(iv).all(axis=(1, 2)) & (iv.max(axis=(1, 2)) < 3.0)
    log.info("training set: %d / %d parameter sets kept (%d dropped for failed/implausible IV)",
             int(good.sum()), n, int((~good).sum()))
    return theta[good], iv[good]
