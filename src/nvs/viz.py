"""Figures. Plotly HTML for interactive 3D (CDN plotly.js, small files) + matplotlib PNGs.

Axes: log-moneyness k = ln(K/F), maturity in years, IV in %. Every function takes explicit output
paths and a title that carries the data tag (real snapshot / synthetic).
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

log = logging.getLogger(__name__)


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)


def plot_surface_3d(k, T, iv, quotes, html_path, png_path, title, name="model"):
    """3D IV surface (iv decimal, shape (nT, nk)) with market quotes as markers."""
    K, TT = np.meshgrid(k, T)
    fig = go.Figure(go.Surface(x=K, y=TT, z=iv * 100, colorscale="Viridis", opacity=0.9, name=name))
    if quotes is not None:
        fig.add_trace(go.Scatter3d(x=quotes["k"], y=quotes["T"], z=quotes["iv"] * 100,
                                   mode="markers", marker=dict(size=2, color="red"),
                                   name="market quotes"))
    fig.update_layout(title=title, scene=dict(xaxis_title="log-moneyness k",
                                              yaxis_title="maturity T (years)",
                                              zaxis_title="implied vol (%)"))
    Path(html_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_path, include_plotlyjs="cdn")
    f = plt.figure(figsize=(8, 6))
    ax = f.add_subplot(111, projection="3d")
    ax.plot_surface(K, TT, iv * 100, cmap="viridis", alpha=0.85, linewidth=0)
    if quotes is not None:
        ax.scatter(quotes["k"], quotes["T"], quotes["iv"] * 100, s=2, c="red")
    ax.set(xlabel="log-moneyness k", ylabel="maturity T (years)", zlabel="implied vol (%)",
           title=title)
    ax.view_init(25, -60)
    _save(f, png_path)


def plot_smiles(quotes, model_fn, path, title, n=6):
    """Smile slices for `n` expiries. model_fn(k_array, T_scalar) -> iv decimal array."""
    exps = sorted(quotes["expiry"].unique())
    pick = [exps[i] for i in np.unique(np.linspace(0, len(exps) - 1, n).astype(int))]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, e in zip(axes.ravel(), pick, strict=False):
        g = quotes[quotes["expiry"] == e].sort_values("k")
        T = g["T"].iloc[0]
        ax.fill_between(g["k"], g["iv_bid"] * 100, g["iv_ask"] * 100, color="0.85", label="bid-ask")
        ax.plot(g["k"], g["iv"] * 100, ".", ms=3, c="k", label="mid")
        kk = np.linspace(g["k"].min(), g["k"].max(), 200)
        ax.plot(kk, model_fn(kk, T) * 100, c="C3", label="model")
        ax.set(title=f"{e}  (T={T:.3f}y)", xlabel="k", ylabel="IV (%)")
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle(title)
    _save(fig, path)


def plot_term_structure(quotes, model_fn, path, title):
    """ATM (k=0) implied vol vs maturity: model curve and near-ATM market mids."""
    near = quotes[quotes["k"].abs() < 0.02].groupby("T")["iv"].mean()
    Ts = np.linspace(quotes["T"].min(), quotes["T"].max(), 100)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(Ts, [model_fn(np.array([0.0]), t)[0] * 100 for t in Ts], c="C3", label="model")
    ax.plot(near.index, near.values * 100, "ko", ms=4, label="market (|k|<0.02)")
    ax.set(xlabel="maturity T (years)", ylabel="ATM IV (%)", title=title)
    ax.legend()
    _save(fig, path)


def plot_residuals(quotes, model_iv, path, title):
    """Residual (market - model) in vol points, scattered over (k, T)."""
    r = (quotes["iv"].values - model_iv) * 100
    lim = np.nanpercentile(np.abs(r), 95)
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(quotes["k"], quotes["T"], c=r, cmap="coolwarm", vmin=-lim, vmax=lim, s=12)
    fig.colorbar(sc, label="market - model (vol points)")
    ax.set(xlabel="log-moneyness k", ylabel="maturity T (years)", title=title)
    _save(fig, path)


def plot_error_grid(k, T, err_volpts, path, title):
    """Heatmap of mean absolute IV error (vol points) over the fixed (k, T) grid (nT, nk)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(err_volpts, origin="lower", aspect="auto", cmap="magma")
    ax.set_xticks(range(len(k)), [f"{x:.2f}" for x in k])
    ax.set_yticks(range(len(T)), [f"{x:.2f}" for x in T])
    for i in range(len(T)):
        for j in range(len(k)):
            ax.text(j, i, f"{err_volpts[i, j]:.3f}", ha="center", va="center", fontsize=6,
                    color="w")
    fig.colorbar(im, label="vol points")
    ax.set(xlabel="log-moneyness k", ylabel="maturity T (years)", title=title)
    _save(fig, path)


def plot_arbitrage_compare(k, T, surfaces: dict, path, title):
    """Heatmap of IV per method with butterfly (x) and calendar (+) violations marked.

    surfaces: name -> (w (nT,nk), butterfly mask (nT,nk), calendar mask (nT-1,nk)).
    """
    fig, axes = plt.subplots(1, len(surfaces), figsize=(5.2 * len(surfaces), 4.5), squeeze=False)
    for ax, (name, (w, bm, cm)) in zip(axes[0], surfaces.items(), strict=True):
        iv = np.sqrt(w / T[:, None]) * 100
        im = ax.pcolormesh(k, T, iv, shading="auto", cmap="viridis")
        fig.colorbar(im, ax=ax, label="IV (%)")
        ti, ki = np.nonzero(bm)
        ax.plot(k[ki], T[ti], "rx", ms=4, label=f"butterfly ({bm.sum()})")
        ti, ki = np.nonzero(cm)
        ax.plot(k[ki], T[ti + 1], "w+", ms=5, label=f"calendar ({cm.sum()})")
        ax.set(title=name, xlabel="k", ylabel="T (years)")
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(title)
    _save(fig, path)
