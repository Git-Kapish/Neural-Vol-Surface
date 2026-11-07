"""Track C: next-day surface forecasting on SYNTHETIC history (no real multi-day history exists).

Everything produced here is synthetic and labelled so. History: latent Heston parameters follow
mean-reverting (OU) dynamics, the exact Heston pricer gives each day's 6x9 IV surface, and i.i.d.
Gaussian quote noise is added. Models (all predict the next-day *observed* grid, decimal IV):
  persistence  y_hat(t+1) = y(t)                      (the hard baseline)
  AR-PCA       VAR(lags) on PCA factors (ridge)
  MLP          MLP on lagged PCA factors predicting the factor change
Evaluation: expanding-window walk-forward, refit every `refit_every` days; PCA/AR/MLP are fit only
on observations up to the refit date, test targets are strictly later. Units: IV decimals, errors
in vol points.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from torch import nn

from nvs import viz
from nvs.config import ROOT, load_config, setup_logging
from nvs.metrics import region_masks
from nvs.pricing.heston import heston_iv
from nvs.surface.arbitrage import butterfly_violations, calendar_violations

log = logging.getLogger(__name__)
DATA_LABEL = "synthetic"


def _ou(n, mean, sd, phi, rng, x0=None):
    """Discrete OU with stationary mean/sd and one-step autocorrelation phi."""
    x = np.empty(n)
    x[0] = mean + sd * rng.standard_normal() if x0 is None else x0
    for t in range(1, n):
        x[t] = mean + phi * (x[t - 1] - mean) + sd * np.sqrt(1 - phi**2) * rng.standard_normal()
    return x


def synthetic_history(cfg: dict, seed: int | None = None) -> dict[str, np.ndarray]:
    """Latent Heston parameters -> exact daily surfaces + noisy observations.

    Returns dict(theta (n,5), truth (n,nT,nk), obs (n,nT,nk)); IV decimals.
    """
    s = cfg["synthetic"]
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
    n, dt, L = s["n_days"], s["dt"], s["latent"]
    phi = lambda a: np.exp(-a * dt)  # noqa: E731
    v0 = np.exp(_ou(n, np.log(L["v0_mean"]), L["v0_sd"], phi(L["v0_speed"]), rng))
    vbar = np.exp(_ou(n, np.log(L["vbar_mean"]), L["vbar_sd"], phi(L["vbar_speed"]), rng))
    z = _ou(n, 0.0, 1.0, phi(L["rho_speed"]), rng)
    b = cfg["heston"]["bounds"]
    theta = np.column_stack([
        np.clip(v0, max(b["v0"][0], L["v0_floor"]), b["v0"][1]), np.full(n, s["kappa"]),
        np.clip(vbar, *b["vbar"]),
        np.full(n, s["sigma"]), L["rho_mean"] + L["rho_amp"] * np.tanh(z)])
    k, T = np.array(cfg["grid"]["k"]), np.array(cfg["grid"]["T"])
    truth = heston_iv(theta, k, T)
    bad = ~np.isfinite(truth).all(axis=(1, 2))
    if bad.any():  # never forward-fill: repeated days would unfairly favour persistence
        raise RuntimeError(f"{int(bad.sum())} synthetic surfaces failed to price; raise v0_floor")
    log.info("synthetic history: %d days, 0 failed surfaces", n)
    obs = truth + rng.normal(0, s["obs_noise_volpts"] / 100, truth.shape)
    return {"theta": theta, "truth": truth, "obs": obs}


# ---- splits (no leakage) -------------------------------------------------------------------
def make_xy(F: np.ndarray, lags: int):
    """Supervised pairs from a (n, q) series. X[i] = F[t-lags+1..t] flattened (oldest first),
    Y[i] = F[t+1], with t = lags-1+i. Returns X, Y, t (feature time of each row): every feature
    time is <= t and the target time t+1 is strictly later."""
    t = np.arange(lags - 1, len(F) - 1)
    X = np.concatenate([F[t - j] for j in range(lags - 1, -1, -1)], axis=1)
    return X, F[t + 1], t


def walk_forward_splits(n: int, min_train: int, step: int):
    """Yield (r, test_target_idx): fit on days [0, r) only, then predict targets r..r+step-1 from
    features at days r-1.. (all known at time r-1). Training targets are all < r <= test targets."""
    r = min_train
    while r < n:
        yield r, np.arange(r, min(r + step, n))
        r += step


# ---- models --------------------------------------------------------------------------------
class PCA:
    def __init__(self, n_components):
        self.q = n_components

    def fit(self, Y):
        self.mean = Y.mean(0)
        _, _, vt = np.linalg.svd(Y - self.mean, full_matrices=False)
        self.comp = vt[: self.q]
        return self

    def transform(self, Y):
        return (Y - self.mean) @ self.comp.T

    def inverse(self, F):
        return F @ self.comp + self.mean


def fit_var(X, Y, ridge=1e-8):
    A = np.hstack([X, np.ones((len(X), 1))])
    return np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ Y)


def predict_var(beta, X):
    return np.hstack([X, np.ones((len(X), 1))]) @ beta


def fit_mlp(X, D, cfg, seed=0):
    """MLP X -> D (factor change), standardised in/out. Returns a predict function."""
    f = cfg["forecast"]
    torch.manual_seed(seed)
    xm, xs, dm, ds = X.mean(0), X.std(0) + 1e-9, D.mean(0), D.std(0) + 1e-9
    Xt = torch.tensor((X - xm) / xs, dtype=torch.float32)
    Dt = torch.tensor((D - dm) / ds, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(X.shape[1], f["mlp_hidden"]), nn.SiLU(),
                        nn.Linear(f["mlp_hidden"], f["mlp_hidden"]), nn.SiLU(),
                        nn.Linear(f["mlp_hidden"], D.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), lr=f["mlp_lr"], weight_decay=1e-3)
    g = torch.Generator().manual_seed(seed)
    for _ in range(f["mlp_epochs"]):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), 128):
            b = perm[i:i + 128]
            loss = ((net(Xt[b]) - Dt[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    def predict(Xn):
        with torch.no_grad():
            out = net(torch.tensor((Xn - xm) / xs, dtype=torch.float32)).numpy()
        return out * ds + dm

    return predict


def walk_forward(obs: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    """Predictions of next-day observed surfaces for every test day. obs (n, nT, nk).
    Returns {model: (n, nT, nk)} (NaN before the first test day) plus the 'index' of test days."""
    f = cfg["forecast"]
    n = len(obs)
    flat = obs.reshape(n, -1)
    preds = {m: np.full_like(flat, np.nan) for m in ("persistence", "AR-PCA", "MLP")}
    for r, tgt in walk_forward_splits(n, f["min_train"], f["refit_every"]):
        pca = PCA(f["n_pca"]).fit(flat[:r])  # train window only
        F = pca.transform(flat)
        X, Y, t = make_xy(F[:r], f["lags"])  # targets <= r-1
        beta = fit_var(X, Y)
        mlp = fit_mlp(X, Y - X[:, -F.shape[1]:], cfg)  # predict the change in factors
        # features at time t = tgt-1 are known at t; build X rows for those t
        Xt, _, _ = make_xy(F[: tgt[-1] + 1], f["lags"])
        sel = (tgt - 1) - (f["lags"] - 1)  # row index of feature time tgt-1
        Xq = Xt[sel]
        last = Xq[:, -F.shape[1]:]
        preds["persistence"][tgt] = flat[tgt - 1]
        preds["AR-PCA"][tgt] = pca.inverse(predict_var(beta, Xq))
        preds["MLP"][tgt] = pca.inverse(last + mlp(Xq))
    return {m: p.reshape(obs.shape) for m, p in preds.items()}


def project_calendar(iv: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Project each surface (.., nT, nk) onto calendar-arbitrage-free: running max of w over T."""
    w = iv**2 * T[:, None]
    return np.sqrt(np.maximum.accumulate(w, axis=-2) / T[:, None])


def count_arbitrage(surfs: np.ndarray, k, T) -> dict[str, int]:
    """Violation counts over a stack of surfaces (n, nT, nk) on the coarse fixed grid (FD g on the
    9-point k grid is only indicative of butterfly arbitrage)."""
    b = c = 0
    for s in surfs:
        w = s**2 * T[:, None]
        b += butterfly_violations(w, k)["count"]
        c += calendar_violations(w)["count"]
    return {"butterfly_violations": b, "calendar_violations": c}


def rmse_by_region(pred, target, masks):
    d = (pred - target) * 100
    return {r: float(np.sqrt(np.nanmean(d[:, m] ** 2))) for r, m in masks.items()}


def main() -> None:
    setup_logging()
    logging.getLogger("nvs.pricing.implied_vol").setLevel(logging.WARNING)
    cfg = load_config()
    k, T = np.array(cfg["grid"]["k"]), np.array(cfg["grid"]["T"])
    hist = synthetic_history(cfg)
    out = ROOT / cfg["data"]["synthetic_dir"]
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "history.npz", **hist)
    obs, truth = hist["obs"], hist["truth"]
    preds = walk_forward(obs, cfg)
    test = np.arange(cfg["forecast"]["min_train"], len(obs))
    masks = region_masks(k, T)
    label = (f"{DATA_LABEL}: latent Heston OU history, {cfg['synthetic']['obs_noise_volpts']} "
             f"vol-pt quote noise")

    rows, arb = [], []
    base = None
    for m, p in preds.items():
        for tag, pp in ((m, p), (m + " + calendar projection", None)):
            if pp is None:
                if m == "persistence":
                    continue
                pp = project_calendar(p[test], T)
                full = np.full_like(p, np.nan)
                full[test] = pp
                pp = full
            r_obs = rmse_by_region(pp[test], obs[test], masks)
            r_true = rmse_by_region(pp[test], truth[test], masks)
            if tag == "persistence":
                base = r_obs
            rows.append({"model": tag, "data": label, "rmse_all_volpts": r_obs["all"],
                         "ratio_vs_persistence": r_obs["all"] / base["all"],
                         **{f"rmse_{r}": v for r, v in r_obs.items() if r != "all"},
                         "rmse_vs_latent_truth_volpts": r_true["all"]})
            arb.append({"model": tag, "data": label, **count_arbitrage(pp[test], k, T),
                        "n_surfaces": len(test)})
    for name, s in (("latent truth (reference)", truth), ("noisy observation", obs)):
        arb.append({"model": name, "data": label, **count_arbitrage(s[test], k, T),
                    "n_surfaces": len(test)})
    tbl, arb = pd.DataFrame(rows), pd.DataFrame(arb)
    tdir = ROOT / "results" / "tables"
    tbl.round(4).to_csv(tdir / "forecast_walkforward.csv", index=False)
    arb.to_csv(tdir / "forecast_arbitrage.csv", index=False)
    log.info("walk-forward (%s), %d test days:\n%s", DATA_LABEL, len(test),
             tbl[["model", "rmse_all_volpts", "ratio_vs_persistence",
                  "rmse_vs_latent_truth_volpts"]].round(4).to_string())
    log.info("arbitrage counts on test surfaces:\n%s", arb.drop(columns="data").to_string())
    viz.plot_forecast(obs, truth, preds, test, tbl, k, T,
                      ROOT / "results" / "figures" / "forecast_synthetic.png", label)


if __name__ == "__main__":
    main()
