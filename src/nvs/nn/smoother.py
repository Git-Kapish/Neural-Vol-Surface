"""Neural smoother (k, T) -> IV with butterfly and calendar penalties, after Ackerer-Tagasovska-Vatter.

Network outputs iv = softplus(raw) + 0.02 (positive), total variance w = iv^2 T. Training loss:
    bid-ask-weighted MSE of iv at the quotes (vol points^2)
  + lam_b * mean(relu(margin_b - g(k, T))^2)       butterfly: Gatheral density g >= 0
  + lam_c * mean(relu(margin_c - dw/dT)^2)         calendar: dw/dT >= 0
with g, dw/dT evaluated by autograd on fresh random collocation points over the data hull every
step. Units: k dimensionless, T years, iv decimal. The unpenalised variant (lam = 0) is trained too,
as an ablation.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as Fn

from nvs import viz
from nvs.config import ROOT, load_config, setup_logging
from nvs.metrics import rmse_volpts
from nvs.surface.arbitrage import butterfly_violations, calendar_violations, check_surface
from nvs.surface.grid import eval_grid, raw_surface_w, svi_iv_at_quotes, svi_surface_w
from nvs.surface.svi import fit_slices

log = logging.getLogger(__name__)
K_SCALE = 0.3


class SmootherNet(nn.Module):
    def __init__(self, hidden=64, layers=3):
        super().__init__()
        dims = [2] + [hidden] * layers
        mods = []
        for a, b in zip(dims[:-1], dims[1:], strict=True):
            mods += [nn.Linear(a, b), nn.SiLU()]
        self.body = nn.Sequential(*mods, nn.Linear(hidden, 1))

    def iv(self, k, T):
        x = torch.stack([k / K_SCALE, torch.sqrt(T)], dim=-1)
        return Fn.softplus(self.body(x).squeeze(-1)) + 0.02

    def w(self, k, T):
        return self.iv(k, T) ** 2 * T


def derivs(net, k, T):
    """(w, dw/dk, d2w/dk2, dw/dT) by autograd; k, T 1-D tensors."""
    k = k.clone().requires_grad_(True)
    T = T.clone().requires_grad_(True)
    w = net.w(k, T)
    w_k = torch.autograd.grad(w.sum(), k, create_graph=True)[0]
    w_kk = torch.autograd.grad(w_k.sum(), k, create_graph=True)[0]
    w_T = torch.autograd.grad(w.sum(), T, create_graph=True)[0]
    return w, w_k, w_kk, w_T


def butterfly_g(w, w_k, w_kk, k):
    return (1 - k * w_k / (2 * w)) ** 2 - w_k**2 / 4 * (1 / w + 0.25) + w_kk / 2


def train_smoother(quotes, cfg, lam_b, lam_c, margin_b=0.01, margin_c=0.001, seed=0):
    s = cfg["smoother"]
    torch.manual_seed(seed)
    net = SmootherNet(s["hidden"], s["layers"]).double()
    k = torch.tensor(quotes["k"].values)
    T = torch.tensor(quotes["T"].values)
    y = torch.tensor(quotes["iv"].values * 100)  # vol points
    wt = 1.0 / np.maximum(quotes["ba_vol"].values, 0.002) ** 2
    wt = torch.tensor(wt / wt.mean())
    k_lo, k_hi = np.clip(np.percentile(quotes["k"], [0.5, 99.5]), -0.35, 0.25)
    T_lo, T_hi = float(quotes["T"].min()), float(quotes["T"].max())
    opt = torch.optim.Adam(net.parameters(), lr=s["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, s["steps"], eta_min=1e-4)
    g = torch.Generator().manual_seed(seed)
    for step in range(s["steps"]):
        fit = (wt * (net.iv(k, T) * 100 - y) ** 2).mean()
        loss = fit
        if lam_b > 0 or lam_c > 0:
            n = s["n_colloc"]
            kc = k_lo + (k_hi - k_lo) * torch.rand(n, generator=g, dtype=torch.float64)
            Tc = T_lo + (T_hi - T_lo) * torch.rand(n, generator=g, dtype=torch.float64)
            w, w_k, w_kk, w_T = derivs(net, kc, Tc)
            pen_b = Fn.relu(margin_b - butterfly_g(w, w_k, w_kk, kc)).pow(2).mean()
            pen_c = Fn.relu(margin_c - w_T).pow(2).mean()
            loss = fit + lam_b * pen_b + lam_c * pen_c
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == s["steps"] - 1:
            log.info("smoother(lam_b=%g) step %d fit %.4f loss %.4f", lam_b, step, fit.item(),
                     loss.item())
    return net


def net_w_grid(net, k, T):
    """Total variance (nT, nk) on a grid, as numpy."""
    kk, TT = np.meshgrid(k, T)
    with torch.no_grad():
        return net.w(torch.tensor(kk.ravel()), torch.tensor(TT.ravel())).numpy().reshape(kk.shape)


def net_iv_at(net, k, T):
    with torch.no_grad():
        return net.iv(torch.tensor(np.asarray(k, float)), torch.tensor(np.asarray(T, float))).numpy()


def raw_iv_at_quotes(quotes):
    """Raw interpolation passes through the data: mean mid IV of quotes sharing (expiry, k)."""
    return quotes.groupby(["expiry", "k"])["iv"].transform("mean").values


def evaluate(quotes, slices, nets, cfg, stamp, data_label):
    """Fit-error and arbitrage table for raw / SVI / the given nets. Returns (table, plot dict, k, T)."""
    k, T = eval_grid(cfg, float(slices["T"].max()))
    surf = {"raw interpolation": raw_surface_w(quotes, k, T), "SVI (per slice)":
            svi_surface_w(slices, k, T)}
    surf.update({n: net_w_grid(m, k, T) for n, m in nets.items()})
    at_q = {"raw interpolation": raw_iv_at_quotes(quotes),
            "SVI (per slice)": svi_iv_at_quotes(slices, quotes)}
    at_q.update({n: net_iv_at(m, quotes["k"].values, quotes["T"].values) for n, m in nets.items()})
    wt = 1.0 / np.maximum(quotes["ba_vol"].values, 0.002) ** 2
    rows, plot = [], {}
    for name, w in surf.items():
        pred = at_q[name]
        rows.append({"method": name, "data": data_label,
                     "rmse_volpts": rmse_volpts(pred, quotes["iv"].values),
                     "rmse_bidask_weighted_volpts": rmse_volpts(pred, quotes["iv"].values, wt),
                     **check_surface(w, k)})
        plot[name] = (w, butterfly_violations(w, k)["mask"], calendar_violations(w)["mask"])
    return pd.DataFrame(rows), plot, k, T


def main() -> None:
    setup_logging()
    cfg = load_config()
    proc = ROOT / cfg["data"]["processed_dir"]
    quotes = pd.read_csv(proc / "quotes.csv")
    slices = pd.read_csv(proc / "svi_slices.csv")
    stamp = (proc / "meta.txt").read_text(encoding="utf-8").split()[0]
    s = cfg["smoother"]
    tbl_dir, fig_dir = ROOT / "results" / "tables", ROOT / "results" / "figures"

    def fit_both(q):
        return {"neural (no penalty)": train_smoother(q, cfg, 0.0, 0.0),
                "neural (arb. penalties)": train_smoother(q, cfg, s["lambda_butterfly"],
                                                          s["lambda_calendar"])}

    nets = fit_both(quotes)
    torch.save({n: m.state_dict() for n, m in nets.items()}, ROOT / "results" / "models" / "smoother.pt")
    tbl, plot, k, T = evaluate(quotes, slices, nets, cfg, stamp, f"real {stamp}")
    tbl.round(5).to_csv(tbl_dir / "smoother_comparison.csv", index=False)
    log.info("smoother vs baselines (k window %s):\n%s", s["eval_k"], tbl.round(4).to_string())
    viz.plot_arbitrage_compare(k, T, plot, fig_dir / "arbitrage_compare.png",
                               f"IV surfaces with arbitrage violations marked - SPY real {stamp}")
    net = nets["neural (arb. penalties)"]
    tag = f"Neural smoother (arb. penalties) vs market - SPY real {stamp}"
    viz.plot_surface_3d(k, T, np.sqrt(plot["neural (arb. penalties)"][0] / T[:, None]), quotes,
                        fig_dir / "surface_neural_3d.html", fig_dir / "surface_neural_3d.png", tag,
                        "neural")
    viz.plot_smiles(quotes, lambda kk, t: net_iv_at(net, kk, np.full_like(kk, t)),
                    fig_dir / "smiles_neural.png", tag)

    # Stress test: the real quotes are benign, so also add synthetic Gaussian noise to the mid IVs
    # (documented in the table's `data` column) to see whether the penalties earn their keep.
    sd = s["stress_noise_volpts"]
    noisy = quotes.copy()
    noisy["iv"] = noisy["iv"] + np.random.default_rng(cfg["seed"]).normal(0, sd / 100, len(noisy))
    slices_n = fit_slices(noisy)
    label = f"real {stamp} + synthetic noise {sd} vol pts"
    tbl_n, plot_n, k, T = evaluate(noisy, slices_n, fit_both(noisy), cfg, stamp, label)
    tbl_n.round(5).to_csv(tbl_dir / "smoother_stress.csv", index=False)
    log.info("stress test (%s):\n%s", label, tbl_n.round(4).to_string())
    viz.plot_arbitrage_compare(k, T, plot_n, fig_dir / "arbitrage_compare_stress.png",
                               f"Stress test: {label}")


if __name__ == "__main__":
    main()
