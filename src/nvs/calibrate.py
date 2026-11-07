"""Calibrate Heston to the real SPY surface: frozen neural pricer vs the exact pricer.

Target: market IV on the fixed (k, T) grid, built from SVI-smoothed real quotes and masked to the
region the data covers (no extrapolation); weight = 1 / half bid-ask width (vol). Both methods fit
the *same* targets with the same random restarts and a bounded Levenberg-Marquardt-type solver
(scipy `least_squares`, trf, parameters mapped to the unit box). The neural solver uses autograd
Jacobians of the network; the exact solver uses finite differences of the exact pricer.
Reported units: IV errors in vol points; time in seconds (CPU, wall clock).
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares
from scipy.stats import qmc

from nvs import viz
from nvs.config import ROOT, load_config, setup_logging
from nvs.nn import pricer as P
from nvs.pricing.heston import PARAM_NAMES, heston_iv
from nvs.surface.grid import eval_grid, fixed_grid, market_grid_targets

log = logging.getLogger(__name__)


def _theta(u, lo, hi):
    return lo + np.asarray(u) * (hi - lo)


def calibrate_nn(net, lo, hi, iv_mkt, w, mask, starts):
    """Multi-start bounded LS with the (float64) network; returns best (theta, cost, nfev, secs)."""
    net = net.double()
    lo_t, hi_t = torch.tensor(lo), torch.tensor(hi)
    m = torch.as_tensor(mask)
    tgt = torch.tensor(iv_mkt[mask])
    wt = torch.tensor(w[mask])

    def resid_t(u):
        out = net(lo_t + u * (hi_t - lo_t))[0][m]
        return (out - tgt) * wt

    def fun(u):
        with torch.no_grad():
            return resid_t(torch.tensor(u)).numpy()

    def jac(u):
        return torch.autograd.functional.jacobian(resid_t, torch.tensor(u)).numpy()

    t0 = time.perf_counter()
    best = None
    for u0 in starts:
        sol = least_squares(fun, u0, jac=jac, bounds=(0, 1), method="trf", max_nfev=200)
        best = sol if best is None or sol.cost < best.cost else best
    return _theta(best.x, lo, hi), best.cost, best.nfev, time.perf_counter() - t0


def calibrate_exact(lo, hi, k, T, iv_mkt, w, mask, starts):
    """Same problem with the exact Heston pricer (IV via inversion) and FD Jacobians."""
    def fun(u):
        iv = heston_iv(_theta(u, lo, hi)[None, :], k, T)[0]
        iv = np.nan_to_num(iv, nan=1.0)  # failed inversion -> huge residual, steers away
        return ((iv - iv_mkt) * w)[mask]

    t0 = time.perf_counter()
    best = None
    for u0 in starts:
        sol = least_squares(fun, u0, bounds=(0, 1), method="trf", max_nfev=200, diff_step=1e-4)
        best = sol if best is None or sol.cost < best.cost else best
    return _theta(best.x, lo, hi), best.cost, best.nfev, time.perf_counter() - t0


def model_iv_at_quotes(theta, quotes) -> np.ndarray:
    """Exact Heston IV at every quote (decimal), one pricer call per expiry."""
    out = np.full(len(quotes), np.nan)
    for _, g in quotes.groupby("expiry"):
        out[quotes.index.get_indexer(g.index)] = heston_iv(theta[None, :], g["k"].values,
                                                           [g["T"].iloc[0]])[0, 0]
    return out


def main() -> None:
    setup_logging()
    logging.getLogger("nvs.pricing.implied_vol").setLevel(logging.WARNING)
    cfg = load_config()
    proc = ROOT / cfg["data"]["processed_dir"]
    quotes = pd.read_csv(proc / "quotes.csv")
    slices = pd.read_csv(proc / "svi_slices.csv")
    stamp = (proc / "meta.txt").read_text(encoding="utf-8").split()[0]
    k, T = fixed_grid(cfg)
    iv_mkt, w, mask = market_grid_targets(quotes, slices, k, T)
    log.info("market target: %d / %d grid cells covered by data", int(mask.sum()), mask.size)

    b = cfg["heston"]["bounds"]
    lo, hi = (np.array([b[n][i] for n in PARAM_NAMES]) for i in (0, 1))
    n_r = cfg["calibration"]["n_restarts"]
    starts = qmc.Sobol(5, scramble=True, seed=cfg["seed"]).random(n_r)
    net = P.load()

    th_nn, c_nn, nf_nn, t_nn = calibrate_nn(net, lo, hi, iv_mkt, w, mask, starts)
    th_ex, c_ex, nf_ex, t_ex = calibrate_exact(lo, hi, k, T, iv_mkt, w, mask, starts)

    def grid_rmse(theta):  # exact-pricer error vs market on the covered grid, vol points
        iv = heston_iv(theta[None, :], k, T)[0]
        return float(np.sqrt(np.nanmean(((iv - iv_mkt)[mask]) ** 2)) * 100)

    in_hull = ((quotes["k"].between(k.min(), k.max())) & (quotes["T"].between(T.min(), T.max()))).values

    def quote_rmse(theta):  # exact pricer vs raw quotes inside the grid's (k, T) range
        mi = model_iv_at_quotes(theta, quotes)
        return float(np.sqrt(np.nanmean((mi - quotes["iv"].values)[in_hull] ** 2)) * 100)

    # single-evaluation cost (one theta -> grid IV), exact vs network
    t0 = time.perf_counter()
    for _ in range(20):
        heston_iv(th_ex[None, :], k, T)
    ms_exact = (time.perf_counter() - t0) / 20 * 1e3
    net32 = P.load()
    th32 = torch.tensor(th_nn[None, :], dtype=torch.float32)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(2000):
            net32(th32)
    ms_nn = (time.perf_counter() - t0) / 2000 * 1e3

    rows = []
    for name, th, c, nf, t in (("neural pricer", th_nn, c_nn, nf_nn, t_nn),
                               ("exact pricer", th_ex, c_ex, nf_ex, t_ex)):
        rows.append({"method": name, **dict(zip(PARAM_NAMES, th.round(4), strict=True)),
                     "grid_rmse_volpts": grid_rmse(th), "quotes_in_grid_range_rmse_volpts": quote_rmse(th),
                     "weighted_cost": c, "nfev_best": nf, "n_restarts": n_r, "seconds": t,
                     "data": f"real {stamp}"})
    tbl = pd.DataFrame(rows)
    tbl["speedup_vs_exact"] = t_ex / tbl["seconds"]
    tbl["ms_per_eval"] = [ms_nn, ms_exact]
    out = ROOT / "results" / "tables"
    tbl.round(5).to_csv(out / "calibration.csv", index=False)
    log.info("calibration:\n%s", tbl.round(4).T.to_string())
    log.info("single-evaluation: network %.3f ms vs exact %.2f ms (x%.0f)", ms_nn, ms_exact,
             ms_exact / ms_nn)

    # Figures from the neural-calibrated theta, evaluated with the EXACT pricer.
    fig = ROOT / "results" / "figures"
    tag = f"Heston (NN-calibrated, exact pricer) vs {stamp} market"
    ke, Te = eval_grid(cfg, float(slices["T"].max()))
    iv_dense = heston_iv(th_nn[None, :], ke, Te)[0]
    viz.plot_surface_3d(ke, Te, iv_dense, quotes, fig / "surface_heston_3d.html",
                        fig / "surface_heston_3d.png", tag, "Heston")

    def model_fn(kk, t):
        return heston_iv(th_nn[None, :], kk, [t])[0, 0]

    viz.plot_smiles(quotes, model_fn, fig / "smiles_heston.png", tag)
    viz.plot_term_structure(quotes, model_fn, fig / "term_structure_heston.png", tag)
    viz.plot_residuals(quotes, model_iv_at_quotes(th_nn, quotes), fig / "residuals_heston.png",
                       "Residual (market - Heston) " + stamp)


if __name__ == "__main__":
    main()
