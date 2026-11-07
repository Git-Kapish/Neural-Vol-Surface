"""Quote filtering, per-expiry forward extraction (put-call parity), OTM mid prices, IV.

Units: strikes/prices in USD, T in years (ACT/365.25 from the valuation time to 20:00 UTC on the
expiry date, i.e. 16:00 New York), rates and vols as decimals, k = ln(K/F) dimensionless.
Valuation time = capture time, rolled back to Friday 20:00 UTC when captured on a weekend (the
delayed quotes are then Friday's close).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from nvs.config import ROOT, load_config
from nvs.pricing.implied_vol import implied_vol

log = logging.getLogger(__name__)


def latest_snapshot_dir(cfg: dict) -> Path:
    dirs = sorted(p for p in (ROOT / cfg["data"]["raw_dir"]).iterdir() if p.is_dir())
    if not dirs:
        raise FileNotFoundError("no snapshots in data/raw; run `make fetch`")
    return dirs[-1]


def load_snapshot(cfg: dict, snap_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    snap_dir = snap_dir or latest_snapshot_dir(cfg)
    t = cfg["data"]["ticker"]
    chain = pd.read_csv(snap_dir / f"{t}_chain.csv")
    meta = json.loads((snap_dir / f"{t}_meta.json").read_text(encoding="utf-8"))
    return chain, meta


def valuation_time(captured_utc: str) -> pd.Timestamp:
    ts = pd.Timestamp(captured_utc).tz_convert("UTC")
    if ts.weekday() >= 5:  # weekend: quotes are Friday's close
        ts = (ts - pd.Timedelta(days=ts.weekday() - 4)).normalize() + pd.Timedelta(hours=20)
    return ts


def year_frac(asof: pd.Timestamp, expiry: str) -> float:
    exp = pd.Timestamp(expiry, tz="UTC") + pd.Timedelta(hours=20)
    return (exp - asof).total_seconds() / (365.25 * 86400)


def filter_quotes(chain: pd.DataFrame, cfg: dict, asof: pd.Timestamp) -> pd.DataFrame:
    """Apply liquidity/staleness filters sequentially; log how many quotes each rule drops."""
    c = cfg["clean"]
    df = chain.copy()
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df["last_trade"] = pd.to_datetime(df["last_trade"], utc=True, errors="coerce")
    df["volume"] = df["volume"].fillna(0)
    rules = [
        ("bid<=0 or ask<=bid", lambda d: (d["bid"] > 0) & (d["ask"] > d["bid"])),
        ("mid price < min_price", lambda d: d["mid"] >= c["min_price"]),
        ("relative spread too wide",
         lambda d: (d["ask"] - d["bid"]) / d["mid"] <= c["max_rel_spread"]),
        ("open interest too low", lambda d: d["open_interest"].fillna(0) >= c["min_open_interest"]),
        ("volume too low", lambda d: d["volume"] >= c["min_volume"]),
        ("stale last trade and no volume today",
         lambda d: ~(((asof - d["last_trade"]).dt.days > c["max_stale_days"]) & (d["volume"] == 0))),
    ]
    counts = {"start": len(df)}
    for name, rule in rules:  # applied in order; each count is what that rule newly rejected
        keep = rule(df)
        counts[name] = int((~keep).sum())
        df = df[keep]
    counts["kept"] = len(df)
    log.info("quote filter: %s", counts)
    return df.reset_index(drop=True)


def extract_forward(strikes, call_mid, put_mid, spot, T, r_fallback, band=0.05, n_max=12):
    """Forward F and discount factor DF from put-call parity C - P = DF (F - K).

    Linear fit of C - P on K using up to `n_max` pairs closest to spot within +-`band`.
    Falls back to DF = exp(-r_fallback T) and the median of K + (C - P)/DF when the fit is
    unresolvable (<3 pairs, T <= 0.15y, or implied rate more than 3% from the fallback); last resort F = spot e^{rT} (flagged).
    Returns (F, DF, method, n_pairs).
    """
    K, d = np.asarray(strikes, float), np.asarray(call_mid, float) - np.asarray(put_mid, float)
    near = np.abs(K / spot - 1) <= band
    K, d = K[near], d[near]
    order = np.argsort(np.abs(K - spot))[:n_max]
    K, d = K[order], d[order]
    df_fb = np.exp(-r_fallback * T)
    if K.size >= 3 and np.ptp(K) > 0:
        slope, icpt = np.polyfit(K, d, 1)
        DF = -slope
        # the slope is only resolvable at longer T; require an implied rate near the fallback
        if T > 0.15 and DF > 0 and abs(-np.log(DF) / T - r_fallback) < 0.03:
            return icpt / DF, DF, "parity_fit", K.size
    if K.size >= 1:
        return float(np.median(K + d / df_fb)), df_fb, "parity_median", K.size
    return spot / df_fb, df_fb, "spot_fallback", 0


def build_quotes(chain: pd.DataFrame, meta: dict, cfg: dict) -> pd.DataFrame:
    """Clean chain -> OTM quote table with forward, k, T, mid/bid/ask IVs and bid-ask half-width.

    Output columns: expiry, T, F, DF, fwd_method, K, k, type (OTM leg), bid, ask, mid,
    call_equiv_mid (mid converted to a call price by parity, USD), iv, iv_bid, iv_ask (decimals),
    ba_vol (half bid-ask width in vol, decimal, floored at 0.001), vega weight inputs.
    """
    asof = valuation_time(meta["captured_utc"])
    spot = meta["spot"]
    r_fb = cfg["data"]["risk_free_rate"]
    df = filter_quotes(chain, cfg, asof)
    rows, fwd_log = [], []
    for exp, g in df.groupby("expiry"):
        T = year_frac(asof, exp)
        if T <= 0:
            continue
        calls, puts = g[g["type"] == "C"].set_index("strike"), g[g["type"] == "P"].set_index("strike")
        both = calls.index.intersection(puts.index)
        F, DF, method, npairs = extract_forward(
            both.values, calls.loc[both, "mid"].values, puts.loc[both, "mid"].values, spot, T, r_fb
        )
        fwd_log.append((exp, round(T, 4), round(F, 2), round(DF, 5), method, npairs))
        g = g.assign(T=T, F=F, DF=DF, fwd_method=method)
        otm = g[((g["type"] == "C") & (g["strike"] >= F)) | ((g["type"] == "P") & (g["strike"] < F))]
        rows.append(otm)
    q = pd.concat(rows, ignore_index=True).rename(columns={"strike": "K"})
    log.info("forwards (expiry, T, F, DF, method, n_pairs): %s", fwd_log)
    q["k"] = np.log(q["K"] / q["F"])
    is_call = (q["type"] == "C").values
    args = (q["F"].values, q["K"].values, q["T"].values, is_call, q["DF"].values)
    iv = implied_vol(q["mid"].values, *args, label="mid")
    ivb = implied_vol(q["bid"].values, *args, label="bid")
    iva = implied_vol(q["ask"].values, *args, label="ask")
    q["iv"], q["iv_bid"], q["iv_ask"] = iv.iv, ivb.iv, iva.iv
    q["call_equiv_mid"] = np.where(is_call, q["mid"], q["mid"] + q["DF"] * (q["F"] - q["K"]))
    ba = 0.5 * (q["iv_ask"] - q["iv_bid"])
    q["ba_vol"] = np.maximum(ba.fillna(ba.median()), 1e-3)
    failed = int(np.isnan(q["iv"]).sum())
    q = q[np.isfinite(q["iv"]) & (q["iv"] > 0.02) & (q["iv"] < 3.0)].reset_index(drop=True)
    log.info("OTM quotes with valid IV: %d (inversion failures dropped and counted: %d)",
             len(q), failed)
    return q


def main() -> pd.DataFrame:
    cfg = load_config()
    chain, meta = load_snapshot(cfg)
    return build_quotes(chain, meta, cfg)
