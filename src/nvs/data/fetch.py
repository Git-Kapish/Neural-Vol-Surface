"""Download an options-chain snapshot via yfinance and save it to data/raw/YYYY-MM-DD/.

Output columns (units): expiry (date), type ('C'/'P'), strike (USD), bid/ask/last (USD),
volume/open_interest (contracts), last_trade (UTC timestamp), yf_iv (decimal, informational only).
meta.json stores spot (USD) and the capture timestamp (UTC).
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from nvs.config import ROOT, load_config

log = logging.getLogger(__name__)


def fetch_chain(
    ticker: str, max_expiries: int, min_dte_days: int, max_dte_days: int
) -> tuple[pd.DataFrame, float]:
    """Return (chain, spot). Chain has one row per option quote."""
    tk = yf.Ticker(ticker)
    hist = tk.history(period="5d")
    if hist.empty:
        raise RuntimeError(f"no price history for {ticker}")
    spot = float(hist["Close"].iloc[-1])
    now = pd.Timestamp.now(tz="UTC").normalize()
    frames = []
    dte = {e: (pd.Timestamp(e, tz="UTC") - now).days for e in tk.options}
    keep = [e for e, d in dte.items() if min_dte_days <= d <= max_dte_days]
    if len(keep) > max_expiries:  # spread evenly so long maturities are covered
        keep = [keep[i] for i in np.unique(np.linspace(0, len(keep) - 1, max_expiries).astype(int))]
    for exp in keep:
        oc = tk.option_chain(exp)
        for kind, df in (("C", oc.calls), ("P", oc.puts)):
            d = df.rename(
                columns={
                    "openInterest": "open_interest",
                    "lastTradeDate": "last_trade",
                    "impliedVolatility": "yf_iv",
                }
            )
            d = d[["strike", "bid", "ask", "lastPrice", "volume", "open_interest",
                   "last_trade", "yf_iv"]].rename(columns={"lastPrice": "last"})
            d.insert(0, "type", kind)
            d.insert(0, "expiry", exp)
            frames.append(d)
    if not frames:
        raise RuntimeError(f"no expiries fetched for {ticker}")
    return pd.concat(frames, ignore_index=True), spot


def save_snapshot(cfg: dict | None = None) -> Path:
    """Fetch and write one snapshot; returns the output directory."""
    cfg = cfg or load_config()
    d = cfg["data"]
    chain, spot = fetch_chain(
        d["ticker"], d["max_expiries"], d["min_dte_days"], d["max_dte_days"]
    )
    ts = datetime.now(UTC)
    out = ROOT / d["raw_dir"] / ts.strftime("%Y-%m-%d")
    out.mkdir(parents=True, exist_ok=True)
    chain.to_csv(out / f"{d['ticker']}_chain.csv", index=False)
    meta = {"ticker": d["ticker"], "spot": spot, "captured_utc": ts.isoformat(),
            "n_quotes": len(chain), "source": "yfinance (delayed)"}
    (out / f"{d['ticker']}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("saved %d quotes, spot=%.2f -> %s", len(chain), spot, out)
    return out
