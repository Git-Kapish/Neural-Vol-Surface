"""Neural pricer: Heston parameters theta -> implied-vol grid on the fixed (k, T) grid.

Two-step approach of Horvath-Muguruza-Tomas: (1) learn theta -> IV(k_j, T_i) offline from exact
Heston prices; (2) calibrate with the frozen network (see calibrate.py). Units: theta as in
nvs.pricing.heston, outputs are annualised IVs (decimals) of shape (B, nT, nk).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from nvs.config import ROOT, load_config, setup_logging
from nvs.metrics import region_error_table
from nvs.pricing.heston import PARAM_NAMES, build_training_set

log = logging.getLogger(__name__)
MODEL_PATH = ROOT / "results" / "models" / "pricer.pt"


class PricerNet(nn.Module):
    def __init__(self, lo, hi, out_mean, out_std, hidden=256, layers=4):
        super().__init__()
        self.register_buffer("lo", torch.as_tensor(lo, dtype=torch.float32))
        self.register_buffer("hi", torch.as_tensor(hi, dtype=torch.float32))
        self.register_buffer("out_mean", torch.as_tensor(out_mean, dtype=torch.float32))
        self.register_buffer("out_std", torch.as_tensor(out_std, dtype=torch.float32))
        dims = [5] + [hidden] * layers
        mods = []
        for a, b in zip(dims[:-1], dims[1:], strict=True):
            mods += [nn.Linear(a, b), nn.SiLU()]
        self.body = nn.Sequential(*mods, nn.Linear(hidden, out_mean.shape[0] * out_mean.shape[1]))
        self.shape = tuple(out_mean.shape)

    def forward(self, theta):
        """theta (B, 5) in natural units -> IV (B, nT, nk) decimals."""
        x = 2 * (theta - self.lo) / (self.hi - self.lo) - 1
        y = self.body(x).view(-1, *self.shape)
        return self.out_mean + self.out_std * y


def _dataset(cfg, cache: Path):
    p = cfg["pricer"]
    n_all = p["n_train"] + p["n_val"] + p["n_test"]
    if cache.exists():
        d = np.load(cache)
        if d["n_requested"] == n_all:
            log.info("loaded cached training set %s", cache)
            return d["theta"], d["iv"]
    theta, iv = build_training_set(cfg, n_all, cfg["seed"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, theta=theta, iv=iv, n_requested=n_all)
    return theta, iv


def split(theta, iv, cfg):
    """Random (seeded) train/val/test split of the generated samples; test is never trained on."""
    p = cfg["pricer"]
    idx = np.random.default_rng(cfg["seed"] + 1).permutation(len(theta))
    n_val, n_test = p["n_val"], p["n_test"]
    te, va, tr = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]
    return (theta[tr], iv[tr]), (theta[va], iv[va]), (theta[te], iv[te])


def build_net(cfg, iv_train) -> PricerNet:
    b = cfg["heston"]["bounds"]
    lo = [b[n][0] for n in PARAM_NAMES]
    hi = [b[n][1] for n in PARAM_NAMES]
    return PricerNet(lo, hi, iv_train.mean(0), iv_train.std(0) + 1e-6,
                     cfg["pricer"]["hidden"], cfg["pricer"]["layers"])


def predict(net: PricerNet, theta) -> np.ndarray:
    net.eval()
    with torch.no_grad():
        return net(torch.as_tensor(theta, dtype=torch.float32)).numpy().astype(np.float64)


def train(cfg, tr, va, verbose=True) -> PricerNet:
    p = cfg["pricer"]
    torch.manual_seed(cfg["seed"])
    net = build_net(cfg, tr[1])
    th = torch.as_tensor(tr[0], dtype=torch.float32)
    y = torch.as_tensor(tr[1], dtype=torch.float32)
    th_v = torch.as_tensor(va[0], dtype=torch.float32)
    y_v = torch.as_tensor(va[1], dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=p["lr"])
    steps_per = int(np.ceil(len(th) / p["batch"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, p["lr"], total_steps=p["epochs"] * steps_per)
    scale = (1.0 / net.out_std).detach()  # loss in standardised vol units
    g = torch.Generator().manual_seed(cfg["seed"])
    t0 = time.time()
    for ep in range(p["epochs"]):
        net.train()
        perm = torch.randperm(len(th), generator=g)
        for i in range(steps_per):
            b = perm[i * p["batch"]:(i + 1) * p["batch"]]
            loss = (((net(th[b]) - y[b]) * scale) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if verbose and (ep % 25 == 0 or ep == p["epochs"] - 1):
            net.eval()
            with torch.no_grad():
                mae = ((net(th_v) - y_v).abs().mean() * 100).item()
            log.info("epoch %3d  train loss %.2e  val MAE %.4f vol pts  (%.0fs)", ep, loss.item(),
                     mae, time.time() - t0)
    return net


def save(net: PricerNet, cfg, path=MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "config": cfg, "out_shape": net.shape}, path)


def load(path=MODEL_PATH) -> PricerNet:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    shape = ck["out_shape"]
    net = build_net(cfg, np.zeros((2, *shape)) + np.array([0.0, 1.0])[:, None, None])
    net.load_state_dict(ck["state"])
    return net.eval()


def main() -> None:
    setup_logging()
    cfg = load_config()
    theta, iv = _dataset(cfg, ROOT / cfg["data"]["synthetic_dir"] / "pricer_train.npz")
    tr, va, te = split(theta, iv, cfg)
    log.info("train %d / val %d / test %d", len(tr[0]), len(va[0]), len(te[0]))
    net = train(cfg, tr, va)
    save(net, cfg)
    k, T = np.array(cfg["grid"]["k"]), np.array(cfg["grid"]["T"])
    pred = predict(net, te[0])
    table = region_error_table(pred, te[1], k, T)
    table.insert(0, "data", "synthetic (exact Heston labels)")
    out = ROOT / "results" / "tables" / "pricer_errors.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out, index=False)
    log.info("held-out test error (vol points):\n%s", table.round(4).to_string())

    from nvs import viz

    viz.plot_error_grid(k, T, np.abs(pred - te[1]).mean(0) * 100,
                        ROOT / "results" / "figures" / "pricer_error_grid.png",
                        "Neural pricer held-out mean |error| (vol points), synthetic Heston")


if __name__ == "__main__":
    main()
